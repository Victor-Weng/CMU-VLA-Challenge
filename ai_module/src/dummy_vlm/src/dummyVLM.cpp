#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ros/ros.h>

#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/Pose2D.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/String.h>
#include <std_msgs/Int32.h>

#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

#include <algorithm> // transform
#include <cctype>    // to lower
#include <sstream>   // space separated numbers

// ADD: we use PoseStamped for frontier goal bridging
#include <geometry_msgs/PoseStamped.h>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

using namespace std;

string waypoint_file_dir;
string object_list_file_dir;
double waypointReachDis = 0.3; // how far from waypont is considered "reached"

vector<float> waypointX, waypointY, waypointHeading;

int objID;
float objMidX, objMidY, objMidZ, objL, objW, objH, objHeading;
string objLabel;

float vehicleX = 0, vehicleY = 0;

string question;

// global cancel flag for stopping ongoing navigation
volatile bool g_cancelNav = false;

// global publisher used by frontier bridge callback
static ros::Publisher g_waypointPub;

// gate to enable/disable frontier bridging from chat
static bool g_frontierEnabled = false;
static ros::Time g_lastNextGoal;      // time we last saw a /next_goal
static ros::Time g_lastBridgedGoal;   // time we last bridged to /way_point_with_heading
static nav_msgs::Odometry g_lastOdom; // to check pose freshness
static ros::Time g_lastMapTime;       // latest /map stamp observed (if we subscribe to map)
// frames for diagnostics (configurable via params)
static std::string g_mapFrame = "map";
static std::string g_robotFrame = "base_link";

// libcurl write callback to collect HTTP response into a std::string
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
  size_t totalSize = size * nmemb;
  if (!userp || !contents)
  {
    return totalSize;
  }
  std::string *s = static_cast<std::string *>(userp);
  s->append(static_cast<const char *>(contents), totalSize);
  return totalSize;
}

// --- Stagnation detection & recovery state ---
static geometry_msgs::Pose2D g_lastWp;         // last published waypoint
static ros::Time g_lastWpStamp;                // time we last published any waypoint
static geometry_msgs::Pose2D g_lastDistinctWp; // last waypoint that changed significantly
static ros::Time g_lastDistinctStamp;          // time of the last distinct waypoint
static float g_refX = 0.0f, g_refY = 0.0f;     // reference pose for movement window
static ros::Time g_refStamp;                   // when ref pose was set
static ros::Time g_lastRecovery;               // last time we triggered recovery
static ros::Time g_recoveryHoldUntil;          // suppress bridging until this time
// Recovery step state
static int g_recoveryPhase = 0; // 0=idle, 1=backing, 2=turning
static geometry_msgs::Pose2D g_recoveryBackupWp;
static geometry_msgs::Pose2D g_recoveryTurnWp;
static ros::Time g_recoveryStepDeadline;

// Tunables (can be overridden via params)
static double g_stagnationSec = 10.0;     // how long similar goals tolerated
static double g_minGoalDelta = 0.20;      // meters; threshold to consider goals different
static double g_minMove = 0.20;           // meters moved within window to consider progress
static double g_recoveryCooldown = 10.0;  // seconds between recovery actions
static double g_turnRadians = M_PI / 1.0; // 160 degrees left turn
static double g_recoveryHoldSec = 3.0;    // seconds to suppress bridging after recovery
// Back-up then turn parameters
static double g_recoveryBackupDist = 0.30;  // meters to back up before turning
static double g_recoveryStepWaitSec = 1.5;  // seconds to wait for backup step
static double g_recoveryTurnWaitSec = 1.0;  // seconds to wait for the turn step
static double g_recoveryReachThresh = 0.15; // meters to consider backup reached

// --- Alternative recovery: go home (0,0) and pause ---
static bool g_goHomeOnLongStagnation = true; // enable go-home recovery
static double g_longStagnationSec = 30.0;    // how long of no progress triggers go-home
static double g_homeX = 0.0, g_homeY = 0.0;  // home waypoint in map frame
static double g_homePauseSec = 1.0;          // pause duration at home before resuming
static ros::Time g_lastProgress;             // last time we detected movement >= g_minMove
// Go-home sequence state
static int g_goHomePhase = 0;              // 0=idle, 1=going_home, 2=push_out
static geometry_msgs::Pose2D g_homeWp;     // home waypoint (x=g_homeX, y=g_homeY, theta=opposite yaw)
static geometry_msgs::Pose2D g_homePushWp; // push-out waypoint from home along opposite yaw
static double g_homeReachThresh = 0.30;    // meters to consider home reached
static double g_homePushDist = 3.0;        // meters to push from home along opposite heading
static double g_homeGoTimeoutSec = 20.0;   // seconds timeout to reach home
static double g_homePushTimeoutSec = 10.0; // seconds timeout to complete push-out
static ros::Time g_goHomeStepDeadline;     // current step deadline

// Helper: record (and de-dup) waypoint events
static inline void recordWaypoint(const geometry_msgs::Pose2D &wp)
{
  g_lastWp = wp;
  g_lastWpStamp = ros::Time::now();
  double dx = wp.x - g_lastDistinctWp.x;
  double dy = wp.y - g_lastDistinctWp.y;
  double d = std::sqrt(dx * dx + dy * dy);
  if (g_lastDistinctStamp.isZero() || d >= g_minGoalDelta)
  {
    g_lastDistinctWp = wp;
    g_lastDistinctStamp = g_lastWpStamp;
  }
}

// forward declarations
// Receives PoseStamped on /next_goal and republishes Pose2D to /way_point_with_heading
void nextGoalCb(const geometry_msgs::PoseStamped::ConstPtr &msg);

bool handleNavCommand(const std::string &q,                // raw question text
                      ros::Publisher &waypointPub,         // publisher for /way_point_with_heading
                      geometry_msgs::Pose2D &waypointMsgs, // reusable message object
                      ros::Rate &rate);

void pubPathWaypoints(ros::Publisher &waypointPub, geometry_msgs::Pose2D &waypointMsgs, ros::Rate &rate);

// frontier bridge callback implementation
void nextGoalCb(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
  // Only bridge when explicitly enabled by user command (prevents conflicts with manual nav)
  if (!g_frontierEnabled)
    return;

  // Suppress bridging if a recovery just triggered and we want to hold the turn-in-place
  if (!g_recoveryHoldUntil.isZero() && ros::Time::now() < g_recoveryHoldUntil)
  {
    ROS_INFO_THROTTLE(1.0, "Frontier bridge suppressed during recovery hold");
    return;
  }

  // Build a simple Pose2D from the incoming PoseStamped (map frame assumed)
  geometry_msgs::Pose2D out;
  out.x = msg->pose.position.x; // copy X in meters
  out.y = msg->pose.position.y; // copy Y in meters
  out.theta = 0.0;              // keep heading neutral (change if you want yaw from quaternion)

  // Publish to the local planner’s expected topic
  g_waypointPub.publish(out);
  g_lastNextGoal = ros::Time::now();
  g_lastBridgedGoal = g_lastNextGoal;

  // Track for stagnation detection
  recordWaypoint(out);

  // Log for visibility
  ROS_INFO("Frontier bridge: sent goal to /way_point_with_heading (%.2f, %.2f)", out.x, out.y);
}

// reading waypoints from file function
void readWaypointFile()
{
  FILE *waypoint_file = fopen(waypoint_file_dir.c_str(), "r");
  if (waypoint_file == NULL)
  {
    printf("\nCannot read input files, exit.\n\n");
    exit(1);
  }

  char str[50];
  int val, pointNum;
  string strCur, strLast;
  while (strCur != "end_header")
  {
    val = fscanf(waypoint_file, "%s", str);
    if (val != 1)
    {
      printf("\nError reading input files, exit.\n\n");
      exit(1);
    }

    strLast = strCur;
    strCur = string(str);

    if (strCur == "vertex" && strLast == "element")
    {
      val = fscanf(waypoint_file, "%d", &pointNum);
      if (val != 1)
      {
        printf("\nError reading input files, exit.\n\n");
        exit(1);
      }
    }
  }

  float x, y, heading;
  int val1, val2, val3;
  for (int i = 0; i < pointNum; i++)
  {
    val1 = fscanf(waypoint_file, "%f", &x);
    val2 = fscanf(waypoint_file, "%f", &y);
    val3 = fscanf(waypoint_file, "%f", &heading);

    if (val1 != 1 || val2 != 1 || val3 != 1)
    {
      printf("\nError reading input files, exit.\n\n");
      exit(1);
    }

    waypointX.push_back(x);
    waypointY.push_back(y);
    waypointHeading.push_back(heading);
  }

  fclose(waypoint_file);
}

// reading objects from file function
void readObjectListFile()
{
  FILE *object_list_file = fopen(object_list_file_dir.c_str(), "r");
  if (object_list_file == NULL)
  {
    printf("\n[WARNING] dummyVLM: object_list file '%s' not found; skipping read.\n\n", object_list_file_dir.c_str());
    return;
  }

  char s[100], s2[100];
  int val1, val2, val3, val4, val5, val6, val7, val8, val9;
  val1 = fscanf(object_list_file, "%d", &objID);
  val2 = fscanf(object_list_file, "%f", &objMidX);
  val3 = fscanf(object_list_file, "%f", &objMidY);
  val4 = fscanf(object_list_file, "%f", &objMidZ);
  val5 = fscanf(object_list_file, "%f", &objL);
  val6 = fscanf(object_list_file, "%f", &objW);
  val7 = fscanf(object_list_file, "%f", &objH);
  val8 = fscanf(object_list_file, "%f", &objHeading);
  val9 = fscanf(object_list_file, "%s", s);

  if (val1 != 1 || val2 != 1 || val3 != 1 || val4 != 1 || val5 != 1 || val6 != 1 || val7 != 1 || val8 != 1 || val9 != 1)
  {
    printf("\n[WARNING] dummyVLM: object_list file '%s' is malformed or empty; skipping read.\n\n", object_list_file_dir.c_str());
    fclose(object_list_file);
    return;
  }

  while (s[strlen(s) - 1] != '"')
  {
    val9 = fscanf(object_list_file, "%s", s2);

    if (val9 != 1)
      break;

    strcat(s, " ");
    strcat(s, s2);
  }

  for (int i = 1; s[i] != '"' && i < 100; i++)
    objLabel += s[i];
}

// Mistral API Function
//  Callback for curl to write response data
std::string askMistral(const std::string &question)
{
  std::string response;

  CURL *curl = curl_easy_init();
  if (!curl)
  {
    std::cerr << "Failed to initialize curl." << std::endl;
    return "";
  }

  std::string url = "https://api.mistral.ai/v1/chat/completions";

  // Add your instructions here
  std::string instructions = R"(You are going to be asked one of three types of questions and are provided a datasheet of information.  
The datasheet provides the time, object name, x/y/z coordinates in space, and reference image in that order. Use it to answer the questions as accurately as possible.  

The first question type is "Numerical" and it takes the form of a question like "How many blue chairs are between the table and the wall?" or "How many black trash cans are near the window?". You must go through the datasheet to find an integer value as the answer. Respond with only a singular integer, without extraneous information.  

The second question type is "Object Reference" and it takes the form of a question like "Find the potted plant on the kitchen island that is closest to the fridge." or "Find the orange chair between the table and sink that is closest to the window.". You must go through the spreadsheet to find the unique object that most closely matches this description. Respond with only the image file corresponding to the correct object. Respond with only a single image file name, without extraneous information.  

The third question type is "Instruction-Following" and it takes the form of a question like "Take the path near the window to the fridge." or "Avoid the path between the two tables and go near the blue trash can near the window.". You start at (0,0). You can only move up, down, left, or right in this grid in single integer-valued squares. Write out the coordinates of the steps for you to accomplish the task provided. Provide only a sequence of coordinates in the form (x,y) - each on a new line.)";

  std::string payload = R"({
        "model":"mistral-large-latest",
        "messages":[
            {"role":"system","content":")" +
                        instructions + R"("},
            {"role":"user","content":")" +
                        question + R"("}
        ]
    })";

  struct curl_slist *headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");
  headers = curl_slist_append(headers, "Authorization: Bearer OFgLAX07gC1v65hDAfeU4AoZNI9ehcMa");

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK)
  {
    std::cerr << "Curl failed: " << curl_easy_strerror(res) << std::endl;
    response = "";
  }

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  // Parse JSON to get assistant content
  try
  {
    auto j = json::parse(response);
    return j["choices"][0]["message"]["content"];
  }
  catch (...)
  {
    std::cerr << "Failed to parse response." << std::endl;
    return "";
  }
}

// handle navigator commands (to be replaced by LLM later)
// Parse and execute simple nav commands
//   1) "goto x y [theta]"  -> publish a single waypoint to (x, y) with optional theta
//   2) "explore [radius] [N]" -> generate N waypoints on a circle around the current pose
// return true : handled false : not
bool handleNavCommand(const std::string &q, ros::Publisher &waypointPub, geometry_msgs::Pose2D &waypointMsgs, ros::Rate &rate)
{
  // trim leading spaces
  size_t start = q.find_first_not_of(" \t\r\n");
  if (start == std::string::npos)
  { // all empty or whitespace
    return false;
  }

  // trimmed q
  std::string s = q.substr(start);

  // lower case for handling goto / explore
  std::string lower = s;
  // transform over the start and end of the string and output at the start, (change in place). apply ::tolower to each char
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

  // STOP: cancel any ongoing navigation promptly
  if (lower.rfind("stop", 0) == 0)
  {
    g_cancelNav = true;
    ROS_WARN("Stop command received; canceling navigation.");
    return true;
  }

  // COMMAND: frontier control commands
  // "frontier start" -> enable bridging from /next_goal
  // "frontier stop"  -> disable bridging
  if (lower.rfind("frontier", 0) == 0)
  {
    // parse "frontier <start|stop>"
    std::istringstream iss(s);
    std::string cmd, arg;
    iss >> cmd >> arg; // cmd="frontier", arg may be empty/start/stop

    if (arg == "start")
    {
      g_frontierEnabled = true;
      ROS_INFO("Frontier exploration ENABLED (bridging /next_goal -> /way_point_with_heading)");
    }
    else if (arg == "stop")
    {
      g_frontierEnabled = false;
      ROS_INFO("Frontier exploration DISABLED");
    }
    else
    {
      ROS_WARN("Usage: frontier <start|stop>");
    }
    return true;
  }

  // COMMAND 1: GOTO
  if (lower.rfind("goto", 0) == 0)
  { /// starts with goto
    // Use an input string stream to parse numbers after the command word.
    std::istringstream iss(s);

    std::string cmd; // will hold "goto"
    float x = 0.0f;  // target x
    float y = 0.0f;  // target y
    float th = 0.0f; // optional heading (theta), default 0

    // Parse: first token (cmd), then x and y
    // iss: is string stream
    iss >> cmd >> x >> y;

    // show proper form if not form
    if (iss.fail())
    {
      ROS_WARN("Usage: goto <x> <y> [theta]");
      return true;
    }

    // parse for 0 but if not available then default to 0 rad
    iss >> th;
    if (iss.fail())
    {
      th = 0.0f;
    }

    // Log for visibility.
    ROS_INFO("Goto command: (x=%.2f, y=%.2f, theta=%.2f)", x, y, th);

    // Use pubPathWaypoints with a single temporary waypoint, preserving existing path
    const auto oldX = waypointX;
    const auto oldY = waypointY;
    const auto oldH = waypointHeading;

    waypointX.clear();
    waypointY.clear();
    waypointHeading.clear();
    waypointX.push_back(x);
    waypointY.push_back(y);
    waypointHeading.push_back(th);

    pubPathWaypoints(waypointPub, waypointMsgs, rate);

    waypointX = oldX;
    waypointY = oldY;
    waypointHeading = oldH;
    return true;
  }

  // COMMAND 2: EXPLORE for radius (default 2) and number of points on the circle (default 8)
  // Sequentially generates then publishes waypoints in the circle formation
  if (lower.rfind("explore", 0) == 0)
  {
    // parse paramters after explore
    std::istringstream iss(s);

    std::string cmd;     // holds "explore"
    float radius = 2.0f; // rad
    int N = 8;

    // read format:
    iss >> cmd >> radius >> N;
    if (iss.fail())
    {
      // reset defaults
      float radius = 2.0f; // rad
      int N = 8;
    }

    // check params are positive
    if (radius <= 0.0f)
      radius = 2.0f;
    if (N <= 0)
      N = 8;

    // log
    ROS_INFO("Explore command: center=(%.2f, %.2f), radius=%.2f, N=%d", vehicleX, vehicleY, radius, N);

    // Build a temporary ring of waypoints and run pubPathWaypoints
    const auto oldX = waypointX;
    const auto oldY = waypointY;
    const auto oldH = waypointHeading;

    waypointX.clear();
    waypointY.clear();
    waypointHeading.clear();

    for (int i = 0; i < N && ros::ok(); ++i)
    {
      float ang = 2.0f * static_cast<float>(M_PI) * (static_cast<float>(i) / static_cast<float>(N));
      float wx = vehicleX + radius * cos(ang);
      float wy = vehicleY + radius * sin(ang);
      float wth = ang;
      waypointX.push_back(wx);
      waypointY.push_back(wy);
      waypointHeading.push_back(wth);
    }

    pubPathWaypoints(waypointPub, waypointMsgs, rate);

    waypointX = oldX;
    waypointY = oldY;
    waypointHeading = oldH;
    return true;
  }
  // Mistral API
  // Print the mistral api output with ROS-INFO

  std::string mistralOutput = askMistral(s);
  if (!mistralOutput.empty())
  {
    ROS_INFO("Mistral says: %s", mistralOutput.c_str());
    return true;
  }
  return false;
}

// publishing helpers:

void pubPathWaypoints(ros::Publisher &waypointPub, geometry_msgs::Pose2D &waypointMsgs, ros::Rate &rate)
{
  // Timeout safeguard per waypoint (seconds) to prevent indefinite waiting
  const double waypointTimeoutSec = 30.0;

  int waypointID = 0;
  int waypointNum = waypointX.size();

  if (waypointNum == 0)
  {
    printf("\nNo waypoint available, exit.\n\n");
    exit(1);
  }

  // publish fist waypoint
  waypointMsgs.x = waypointX[waypointID];
  waypointMsgs.y = waypointY[waypointID];
  waypointMsgs.theta = waypointHeading[waypointID];
  if (waypointPub.getNumSubscribers() == 0)
    ROS_WARN("No subscribers on /way_point_with_heading (planner not running?)");
  waypointPub.publish(waypointMsgs);
  recordWaypoint(waypointMsgs);

  // Start timer for timeout handling
  ros::Time wpStart = ros::Time::now();

  bool status = ros::ok();
  while (status)
  {
    if (g_cancelNav)
    {
      ROS_WARN("Navigation canceled by STOP command.");
      g_cancelNav = false;
      break;
    }

    ros::spinOnce();

    float disX = vehicleX - waypointX[waypointID];
    float disY = vehicleY - waypointY[waypointID];

    // move to the next waypoint and publish
    if (sqrt(disX * disX + disY * disY) < waypointReachDis)
    {
      if (waypointID == waypointNum - 1)
        break; // break out at least waypoint
      waypointID++;

      waypointMsgs.x = waypointX[waypointID];
      waypointMsgs.y = waypointY[waypointID];
      waypointMsgs.theta = waypointHeading[waypointID];
      waypointPub.publish(waypointMsgs);
      recordWaypoint(waypointMsgs);
      wpStart = ros::Time::now(); // reset timeout timer on advance
    }

    // If planner rejected goal or cannot reach, avoid hanging forever
    if ((ros::Time::now() - wpStart).toSec() > waypointTimeoutSec)
    {
      ROS_WARN("Timeout waiting to reach waypoint %d/%d; skipping", waypointID + 1, waypointNum);
      if (waypointID == waypointNum - 1)
        break;
      waypointID++;
      waypointMsgs.x = waypointX[waypointID];
      waypointMsgs.y = waypointY[waypointID];
      waypointMsgs.theta = waypointHeading[waypointID];
      waypointPub.publish(waypointMsgs);
      recordWaypoint(waypointMsgs);
      wpStart = ros::Time::now();
    }

    status = ros::ok();
    rate.sleep();
  }
}

void pubObjectWaypoint(ros::Publisher &waypointPub, geometry_msgs::Pose2D &waypointMsgs)
{
  waypointMsgs.x = objMidX;
  waypointMsgs.y = objMidY;
  waypointMsgs.theta = 0;
  waypointPub.publish(waypointMsgs);
}

void pubObjectMarker(ros::Publisher &objectMarkerPub, visualization_msgs::Marker &objectMarkerMsgs)
{
  objectMarkerMsgs.header.frame_id = "map";
  objectMarkerMsgs.header.stamp = ros::Time().now();
  objectMarkerMsgs.ns = objLabel;
  objectMarkerMsgs.id = objID;
  objectMarkerMsgs.action = visualization_msgs::Marker::ADD;
  objectMarkerMsgs.type = visualization_msgs::Marker::CUBE;
  objectMarkerMsgs.pose.position.x = objMidX;
  objectMarkerMsgs.pose.position.y = objMidY;
  objectMarkerMsgs.pose.position.z = objMidZ;
  objectMarkerMsgs.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(0, 0, objHeading);
  objectMarkerMsgs.scale.x = objL;
  objectMarkerMsgs.scale.y = objW;
  objectMarkerMsgs.scale.z = objH;
  objectMarkerMsgs.color.a = 0.5;
  objectMarkerMsgs.color.r = 0;
  objectMarkerMsgs.color.g = 0;
  objectMarkerMsgs.color.b = 1.0;
  objectMarkerPub.publish(objectMarkerMsgs);
}

void delObjectMarker(ros::Publisher &objectMarkerPub, visualization_msgs::Marker &objectMarkerMsgs)
{
  objectMarkerMsgs.header.frame_id = "map";
  objectMarkerMsgs.header.stamp = ros::Time().now();
  objectMarkerMsgs.ns = objLabel;
  objectMarkerMsgs.id = objID;
  objectMarkerMsgs.action = visualization_msgs::Marker::DELETE;
  objectMarkerMsgs.type = visualization_msgs::Marker::CUBE;
  objectMarkerPub.publish(objectMarkerMsgs);
}

void pubNumericalAnswer(ros::Publisher &numericalAnswerPub, std_msgs::Int32 &numericalResponseMsg, int32_t numericalResponse)
{
  numericalResponseMsg.data = numericalResponse;
  numericalAnswerPub.publish(numericalResponseMsg);
}

// vehicle pose callback function
void poseHandler(const nav_msgs::Odometry::ConstPtr &pose)
{
  vehicleX = pose->pose.pose.position.x;
  vehicleY = pose->pose.pose.position.y;
  g_lastOdom = *pose;
  if (g_refStamp.isZero())
  {
    g_refX = vehicleX;
    g_refY = vehicleY;
    g_refStamp = ros::Time::now();
    g_lastProgress = g_refStamp;
  }
  ROS_INFO_THROTTLE(1.0, "Odom x=%.2f y=%.2f (frame=%s child=%s)", vehicleX, vehicleY,
                    pose->header.frame_id.c_str(), pose->child_frame_id.c_str());
}

// optional: map stamp tracker for diagnostics
void mapStampHandler(const nav_msgs::OccupancyGrid::ConstPtr &msg)
{
  g_lastMapTime = msg->header.stamp;
}

void questionHandler(const std_msgs::String::ConstPtr &msg)
{
  ROS_INFO("Received question");
  question = msg->data;
}

// Periodic diagnostics: prints status about frontier bridging and required topics/TF
void diagnosticsTimerCb(const ros::TimerEvent &)
{
  if (!g_frontierEnabled)
  {
    // We still run stagnation recovery even if frontier is off when navigation is active.
    // But keep logs quiet otherwise.
  }

  ros::Time now = ros::Time::now();

  // (Optional) Could query ROS master for topic presence, but keep diagnostics lightweight here

  // Planner subscriber count on /way_point_with_heading
  int plannerSubs = g_waypointPub.getNumSubscribers();

  // Odom age
  double odomAge = (now - g_lastOdom.header.stamp).toSec();
  // Map age (if we saw any)
  double mapAge = g_lastMapTime.isZero() ? -1.0 : (now - g_lastMapTime).toSec();
  // Last bridged goal age
  double bridgedAge = g_lastBridgedGoal.isZero() ? -1.0 : (now - g_lastBridgedGoal).toSec();

  // TF availability: try a quick lookup to see if map->base_link exists
  static tf::TransformListener tfListener;
  bool tfOk = true;
  try
  {
    tf::StampedTransform t;
    tfListener.lookupTransform(g_mapFrame, g_robotFrame, ros::Time(0), t);
  }
  catch (const tf::TransformException &)
  {
    tfOk = false;
  }

  ROS_INFO_THROTTLE(2.0,
                    "Frontier diag: plannerSubs=%d mapAge=%.1fs odomAge=%.1fs lastBridged=%.1fs tf=%s",
                    plannerSubs,
                    mapAge,
                    odomAge,
                    bridgedAge,
                    tfOk ? "OK" : "MISSING");

  if (plannerSubs == 0)
  {
    ROS_WARN_THROTTLE(2.0, "No subscribers on /way_point_with_heading (planner not running?)");
  }
  if (!tfOk)
  {
    ROS_WARN_THROTTLE(5.0, "TF %s->%s not available; frontier goals cannot be selected reliably",
                      g_mapFrame.c_str(), g_robotFrame.c_str());
  }
  if (mapAge < 0.0)
  {
    ROS_WARN_THROTTLE(5.0, "No /map seen yet; frontier_explorer will wait");
  }
  if (bridgedAge < 0.0)
  {
    ROS_WARN_THROTTLE(5.0, "No /next_goal bridged yet; is frontier_explorer running?");
  }

  // --- Stagnation detection & recovery ---
  bool navActive = g_frontierEnabled || (!g_lastWpStamp.isZero() && (now - g_lastWpStamp).toSec() < 5.0);
  if (navActive)
  {
    // Goal similarity over time
    bool goalsStagnant = (!g_lastDistinctStamp.isZero() && (now - g_lastDistinctStamp).toSec() > g_stagnationSec);

    // Movement check in a sliding window
    bool notMoving = false;
    if (!g_refStamp.isZero() && (now - g_refStamp).toSec() > g_stagnationSec)
    {
      double dx = vehicleX - g_refX;
      double dy = vehicleY - g_refY;
      double moved = std::sqrt(dx * dx + dy * dy);
      if (moved < g_minMove)
        notMoving = true;
      else
        g_lastProgress = now;
      // reset window
      g_refX = vehicleX;
      g_refY = vehicleY;
      g_refStamp = now;
    }

    // Start a recovery sequence if not already in one
    if (g_recoveryPhase == 0 && (goalsStagnant || notMoving) && (g_lastRecovery.isZero() || (now - g_lastRecovery).toSec() > g_recoveryCooldown))
    {
      // If configured and stagnation is long, prefer go-home instead of backup-turn
      bool longStagnation = g_goHomeOnLongStagnation && !g_lastProgress.isZero() && ((now - g_lastProgress).toSec() > g_longStagnationSec);
      if (longStagnation)
      {
        // Compute opposite heading from the last observed yaw
        tf::Quaternion qh;
        tf::quaternionMsgToTF(g_lastOdom.pose.pose.orientation, qh);
        double r_h, p_h, y_h;
        tf::Matrix3x3(qh).getRPY(r_h, p_h, y_h);
        double oppositeYaw = y_h + M_PI;
        while (oppositeYaw > M_PI)
          oppositeYaw -= 2.0 * M_PI;
        while (oppositeYaw < -M_PI)
          oppositeYaw += 2.0 * M_PI;

        // Initialize go-home sequence
        g_goHomePhase = 1; // going_home
        g_homeWp.x = g_homeX;
        g_homeWp.y = g_homeY;
        g_homeWp.theta = static_cast<float>(oppositeYaw);
        // Push-out waypoint from home along opposite heading
        g_homePushWp.x = g_homeWp.x + g_homePushDist * std::cos(oppositeYaw);
        g_homePushWp.y = g_homeWp.y + g_homePushDist * std::sin(oppositeYaw);
        g_homePushWp.theta = static_cast<float>(oppositeYaw);

        ROS_WARN("Recovery: long stagnation detected (%.0fs) -> going to home (%.2f, %.2f) facing opposite (yaw=%.2f)",
                 (now - g_lastProgress).toSec(), g_homeWp.x, g_homeWp.y, g_homeWp.theta);
        g_waypointPub.publish(g_homeWp);
        recordWaypoint(g_homeWp);
        g_lastRecovery = now;
        g_goHomeStepDeadline = now + ros::Duration(g_homeGoTimeoutSec);
        // Hold frontier bridging off during the go-home sequence
        g_recoveryHoldUntil = now + ros::Duration(std::max(g_homePauseSec, 1.0));
        // Skip backup-turn path this time
        return;
      }

      // Compute yaw from odometry
      tf::Quaternion q;
      tf::quaternionMsgToTF(g_lastOdom.pose.pose.orientation, q);
      double roll, pitch, yaw;
      tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
      double newYaw = yaw + g_turnRadians;
      // normalize to [-pi, pi]
      while (newYaw > M_PI)
        newYaw -= 2.0 * M_PI;
      while (newYaw < -M_PI)
        newYaw += 2.0 * M_PI;
      // Step 1: compute and publish a backup waypoint behind the robot
      g_recoveryBackupWp.x = vehicleX - g_recoveryBackupDist * std::cos(yaw);
      g_recoveryBackupWp.y = vehicleY - g_recoveryBackupDist * std::sin(yaw);
      g_recoveryBackupWp.theta = static_cast<float>(yaw); // keep current heading while backing up

      // Step 2: turn in place at the backed-up position
      g_recoveryTurnWp.x = g_recoveryBackupWp.x;
      g_recoveryTurnWp.y = g_recoveryBackupWp.y;
      g_recoveryTurnWp.theta = static_cast<float>(newYaw);

      // Start the recovery sequence with backing up
      g_recoveryPhase = 1;
      g_recoveryStepDeadline = now + ros::Duration(g_recoveryStepWaitSec);
      // Hold bridging long enough to perform the backup step
      g_recoveryHoldUntil = now + ros::Duration(std::max(g_recoveryHoldSec, g_recoveryStepWaitSec));

      ROS_WARN("Recovery: detected %s; backing up %.2fm then turning left to theta=%.2f rad",
               goalsStagnant && notMoving ? "stagnant goals and no movement"
                                          : (goalsStagnant ? "stagnant goals" : "no movement"),
               g_recoveryBackupDist, newYaw);
      ROS_INFO("Step 1/2: publish backup waypoint (%.2f, %.2f); suppressing bridge for %.1fs",
               g_recoveryBackupWp.x, g_recoveryBackupWp.y,
               (g_recoveryHoldUntil - now).toSec());
      g_waypointPub.publish(g_recoveryBackupWp);
      recordWaypoint(g_recoveryBackupWp);
      g_lastRecovery = now;
    }

    // Drive recovery sequence steps
    if (g_goHomePhase == 1)
    {
      // Driving to home: check distance or timeout
      double dx = vehicleX - g_homeWp.x;
      double dy = vehicleY - g_homeWp.y;
      double dist = std::sqrt(dx * dx + dy * dy);
      if (dist < g_homeReachThresh || now >= g_goHomeStepDeadline)
      {
        // Arrived (or give up), pause briefly, then push out
        ROS_INFO("Go-Home: reached home within %.2fm (or timeout). Pausing %.1fs, then pushing out %.1fm",
                 g_homeReachThresh, g_homePauseSec, g_homePushDist);
        g_goHomePhase = 2;
        g_goHomeStepDeadline = now + ros::Duration(g_homePushTimeoutSec);
        g_recoveryHoldUntil = now + ros::Duration(g_homePauseSec);
        // Publish the push-out waypoint after the pause window; for simplicity publish now
        g_waypointPub.publish(g_homePushWp);
        recordWaypoint(g_homePushWp);
      }
    }
    else if (g_goHomePhase == 2)
    {
      // Pushing out: finish on timeout or when near target
      double dx = vehicleX - g_homePushWp.x;
      double dy = vehicleY - g_homePushWp.y;
      double dist = std::sqrt(dx * dx + dy * dy);
      if (dist < g_recoveryReachThresh || now >= g_goHomeStepDeadline)
      {
        ROS_INFO("Go-Home: push-out complete. Resuming normal operation.");
        g_goHomePhase = 0; // done
      }
    }
    else if (g_recoveryPhase == 1)
    {
      // Check if backup reached or deadline passed
      double dx = vehicleX - g_recoveryBackupWp.x;
      double dy = vehicleY - g_recoveryBackupWp.y;
      double dist = std::sqrt(dx * dx + dy * dy);
      if (dist < g_recoveryReachThresh || now >= g_recoveryStepDeadline)
      {
        // Proceed to turning step
        g_recoveryPhase = 2;
        g_recoveryStepDeadline = now + ros::Duration(g_recoveryTurnWaitSec);
        // Extend hold to cover turn step
        if (g_recoveryHoldUntil < g_recoveryStepDeadline)
          g_recoveryHoldUntil = g_recoveryStepDeadline;
        ROS_INFO("Step 2/2: publish turn-in-place waypoint theta=%.2f; holding bridge for %.1fs",
                 g_recoveryTurnWp.theta, (g_recoveryHoldUntil - now).toSec());
        g_waypointPub.publish(g_recoveryTurnWp);
        recordWaypoint(g_recoveryTurnWp);
      }
    }
    else if (g_recoveryPhase == 2)
    {
      // Finish recovery after turn wait window
      if (now >= g_recoveryStepDeadline)
      {
        g_recoveryPhase = 0; // done
        ROS_INFO("Recovery sequence completed");
      }
    }
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "dummyVLM");
  ros::NodeHandle nh;
  ros::NodeHandle nhPrivate = ros::NodeHandle("~");

  nhPrivate.getParam("waypoint_file_dir", waypoint_file_dir);
  nhPrivate.getParam("object_list_file_dir", object_list_file_dir);
  nhPrivate.getParam("waypointReachDis", waypointReachDis);
  nhPrivate.getParam("map_frame", g_mapFrame);
  nhPrivate.getParam("robot_frame", g_robotFrame);
  nhPrivate.param("stagnation_time", g_stagnationSec, g_stagnationSec);
  nhPrivate.param("min_goal_delta", g_minGoalDelta, g_minGoalDelta);
  nhPrivate.param("min_movement", g_minMove, g_minMove);
  nhPrivate.param("recovery_cooldown", g_recoveryCooldown, g_recoveryCooldown);
  nhPrivate.param("turn_radians", g_turnRadians, g_turnRadians);
  nhPrivate.param("recovery_hold_sec", g_recoveryHoldSec, g_recoveryHoldSec);
  nhPrivate.param("recovery_backup_dist", g_recoveryBackupDist, g_recoveryBackupDist);
  nhPrivate.param("recovery_step_wait_sec", g_recoveryStepWaitSec, g_recoveryStepWaitSec);
  nhPrivate.param("recovery_turn_wait_sec", g_recoveryTurnWaitSec, g_recoveryTurnWaitSec);
  nhPrivate.param("recovery_reach_thresh", g_recoveryReachThresh, g_recoveryReachThresh);
  nhPrivate.param("go_home_on_long_stagnation", g_goHomeOnLongStagnation, g_goHomeOnLongStagnation);
  nhPrivate.param("long_stagnation_sec", g_longStagnationSec, g_longStagnationSec);
  nhPrivate.param("home_x", g_homeX, g_homeX);
  nhPrivate.param("home_y", g_homeY, g_homeY);
  nhPrivate.param("home_pause_sec", g_homePauseSec, g_homePauseSec);
  nhPrivate.param("home_reach_thresh", g_homeReachThresh, g_homeReachThresh);
  nhPrivate.param("home_push_dist", g_homePushDist, g_homePushDist);
  nhPrivate.param("home_go_timeout_sec", g_homeGoTimeoutSec, g_homeGoTimeoutSec);
  nhPrivate.param("home_push_timeout_sec", g_homePushTimeoutSec, g_homePushTimeoutSec);

  ros::Subscriber subPose = nh.subscribe<nav_msgs::Odometry>("/state_estimation", 5, poseHandler);

  ros::Subscriber subQuestion = nh.subscribe<std_msgs::String>("/challenge_question", 5, questionHandler);

  // advertise the waypoint publisher before creating the bridge subscriber
  g_waypointPub = nh.advertise<geometry_msgs::Pose2D>("/way_point_with_heading", 5);
  geometry_msgs::Pose2D waypointMsgs;

  // subscribe to /next_goal (from frontier_explorer) and republish to /way_point_with_heading
  ros::Subscriber nextGoalSub = nh.subscribe<geometry_msgs::PoseStamped>("/next_goal", 1, nextGoalCb);

  ros::Publisher objectMarkerPub = nh.advertise<visualization_msgs::Marker>("selected_object_marker", 5);
  visualization_msgs::Marker objectMarkerMsgs;

  ros::Publisher numericalAnswerPub = nh.advertise<std_msgs::Int32>("/numerical_response", 5);
  std_msgs::Int32 numericalResponseMsg;

  // local alias for publishing waypoints inside this function
  ros::Publisher &waypointPub = g_waypointPub;

  // optional: map subscriber for diagnostics (stamp only)
  ros::Subscriber mapSubDiag = nh.subscribe<nav_msgs::OccupancyGrid>("/map", 1, mapStampHandler);

  // diagnostics timer
  ros::Timer diagTimer = nh.createTimer(ros::Duration(2.0), diagnosticsTimerCb);

  // read waypoints from file
  readWaypointFile();

  // read objects from file
  readObjectListFile();

  ros::Rate rate(100);

  ROS_INFO("Awaiting question...");

  bool status = ros::ok();
  while (status)
  {
    ros::spinOnce();
    if (question.empty())
    {
      continue;
    }
    if (question.rfind("Find", 0) == 0 || question.rfind("find", 0) == 0)
    {
      ROS_INFO("Marking and navigating to object.");
      printf("[dummyVLM] Received 'Find' question: %s\n", question.c_str());
      pubObjectMarker(objectMarkerPub, objectMarkerMsgs);
      pubObjectWaypoint(waypointPub, waypointMsgs);
    }
    else if (question.rfind("How many", 0) == 0 || question.rfind("how many", 0) == 0)
    {
      delObjectMarker(objectMarkerPub, objectMarkerMsgs);
      int32_t number = (rand() % 10) + 1;
      printf("[dummyVLM] Received 'How many' question: %s\n", question.c_str());
      ROS_INFO("%d", number);
      pubNumericalAnswer(numericalAnswerPub, numericalResponseMsg, number);
    }
    // custom navigation
    else if (handleNavCommand(question, waypointPub, waypointMsgs, rate))
    {
      ROS_INFO("Navigation command handled.");
    }
    else
    {
      delObjectMarker(objectMarkerPub, objectMarkerMsgs);
      printf("[dummyVLM] Received other question: %s\n", question.c_str());
      ROS_INFO("Navigation starts.");
      pubPathWaypoints(waypointPub, waypointMsgs, rate);
      ROS_INFO("Navigation ends.");
    }
    question.clear();
    ROS_INFO("Awaiting question...");
  }

  return 0;
}
