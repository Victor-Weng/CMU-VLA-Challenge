#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <ros/ros.h>

#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/Pose2D.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/String.h>
#include <std_msgs/Int32.h>

#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <algorithm> // transform
#include <cctype>    // to lower
#include <sstream>   // space separated numbers

using namespace std;

string waypoint_file_dir;
string object_list_file_dir;
double waypointReachDis = 1.0; // how far from waypont is considered "reached"

vector<float> waypointX, waypointY, waypointHeading;

int objID;
float objMidX, objMidY, objMidZ, objL, objW, objH, objHeading;
string objLabel;

float vehicleX = 0, vehicleY = 0;

string question;

// global cancel flag for stopping ongoing navigation
volatile bool g_cancelNav = false;

// latest occupancy grid
nav_msgs::OccupancyGrid g_map;
bool g_hasMap = false;

// forward declarations

bool handleNavCommand(const std::string &q,                // raw question text
                      ros::Publisher &waypointPub,         // publisher for /way_point_with_heading
                      geometry_msgs::Pose2D &waypointMsgs, // reusable message object
                      ros::Rate &rate);

void pubPathWaypoints(ros::Publisher &waypointPub, geometry_msgs::Pose2D &waypointMsgs, ros::Rate &rate);

// frontier-based target selection
bool findBestFrontierTarget(const nav_msgs::OccupancyGrid &map,
                            float rx, float ry,
                            float &tx, float &ty, float &th);

// /map callback
void mapHandler(const nav_msgs::OccupancyGrid::ConstPtr &msg);

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
    printf("\nCannot read input files, exit.\n\n");
    exit(1);
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
    exit(1);
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

  // COMMAND 1: GOTO

  // STOP: cancel any ongoing navigation promptly
  if (lower.rfind("stop", 0) == 0)
  {
    g_cancelNav = true;
    ROS_WARN("Stop command received; canceling navigation.");
    return true;
  }

  if (lower.rfind("goto", 0) == 0)
  { /// starts with goto
    // starting a new navigation; ensure cancel flag is cleared
    g_cancelNav = false;
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
    // starting a new navigation; ensure cancel flag is cleared
    g_cancelNav = false;
    // parse paramters after explore
    std::istringstream iss(s);

    std::string cmd;     // holds "explore"
    float radius = 2.0f; // rad
    int N = 8;

    // read format:
    iss >> cmd >> radius >> N;
    if (iss.fail())
    {
      // keep defaults
      iss.clear();
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

  // COMMAND 3: FRONTIER exploration (auto-explore via occupancy grid)
  if (lower.rfind("frontier", 0) == 0 || lower.rfind("autoexplore", 0) == 0)
  {
    // starting a new navigation; ensure cancel flag is cleared
    g_cancelNav = false;
    std::istringstream iss(s);
    std::string cmd;
    int maxSteps = 10; // default hops
    iss >> cmd >> maxSteps;
    if (maxSteps <= 0)
      maxSteps = 10;

    if (!g_hasMap)
    {
      ROS_WARN("No /map received yet; cannot run frontier exploration.");
      return true;
    }

    ROS_INFO("Frontier exploration: maxSteps=%d", maxSteps);
    for (int step = 0; step < maxSteps && ros::ok() && !g_cancelNav; ++step)
    {
      float tx, ty, th;
      if (!findBestFrontierTarget(g_map, vehicleX, vehicleY, tx, ty, th))
      {
        ROS_INFO("No valid frontier found; exploration complete.");
        break;
      }

      const auto oldX = waypointX;
      const auto oldY = waypointY;
      const auto oldH = waypointHeading;
      waypointX.clear();
      waypointY.clear();
      waypointHeading.clear();
      waypointX.push_back(tx);
      waypointY.push_back(ty);
      waypointHeading.push_back(th);
      ROS_INFO("Frontier step %d: target (%.2f, %.2f) th=%.2f", step + 1, tx, ty, th);
      pubPathWaypoints(waypointPub, waypointMsgs, rate);
      waypointX = oldX;
      waypointY = oldY;
      waypointHeading = oldH;

      ros::spinOnce();
      rate.sleep();
    }
    return true;
  }

  std::cout << "Nothing read";
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
  ROS_INFO_THROTTLE(1.0, "Odom x=%.2f y=%.2f", vehicleX, vehicleY);
}

void questionHandler(const std_msgs::String::ConstPtr &msg)
{
  ROS_INFO("Received question");
  question = msg->data;
}

// /map callback: store latest occupancy grid
void mapHandler(const nav_msgs::OccupancyGrid::ConstPtr &msg)
{
  g_map = *msg;
  g_hasMap = true;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "dummyVLM");
  ros::NodeHandle nh;
  ros::NodeHandle nhPrivate = ros::NodeHandle("~");

  nhPrivate.getParam("waypoint_file_dir", waypoint_file_dir);
  nhPrivate.getParam("object_list_file_dir", object_list_file_dir);
  nhPrivate.getParam("waypointReachDis", waypointReachDis);

  ros::Subscriber subPose = nh.subscribe<nav_msgs::Odometry>("/state_estimation", 5, poseHandler);

  ros::Subscriber subQuestion = nh.subscribe<std_msgs::String>("/challenge_question", 5, questionHandler);
  ros::Subscriber subMap = nh.subscribe<nav_msgs::OccupancyGrid>("/map", 1, mapHandler);

  ros::Publisher waypointPub = nh.advertise<geometry_msgs::Pose2D>("/way_point_with_heading", 5);
  geometry_msgs::Pose2D waypointMsgs;

  ros::Publisher objectMarkerPub = nh.advertise<visualization_msgs::Marker>("selected_object_marker", 5);
  visualization_msgs::Marker objectMarkerMsgs;

  ros::Publisher numericalAnswerPub = nh.advertise<std_msgs::Int32>("/numerical_response", 5);
  std_msgs::Int32 numericalResponseMsg;

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

// --- Frontier selection ---
static inline bool worldToMap(float wx, float wy, const nav_msgs::OccupancyGrid &map, int &mx, int &my)
{
  const float res = map.info.resolution;
  const float ox = map.info.origin.position.x;
  const float oy = map.info.origin.position.y;
  mx = static_cast<int>(floor((wx - ox) / res));
  my = static_cast<int>(floor((wy - oy) / res));
  return (mx >= 0 && my >= 0 && mx < static_cast<int>(map.info.width) && my < static_cast<int>(map.info.height));
}

static inline void mapToWorld(int mx, int my, const nav_msgs::OccupancyGrid &map, float &wx, float &wy)
{
  const float res = map.info.resolution;
  const float ox = map.info.origin.position.x;
  const float oy = map.info.origin.position.y;
  wx = ox + (mx + 0.5f) * res;
  wy = oy + (my + 0.5f) * res;
}

static inline int idxOf(int x, int y, int width) { return y * width + x; }

bool findBestFrontierTarget(const nav_msgs::OccupancyGrid &map,
                            float rx, float ry,
                            float &tx, float &ty, float &th)
{
  const int W = static_cast<int>(map.info.width);
  const int H = static_cast<int>(map.info.height);
  if (W <= 2 || H <= 2 || map.data.empty())
    return false;

  int rmx = 0, rmy = 0;
  if (!worldToMap(rx, ry, map, rmx, rmy))
  {
    rmx = W / 2;
    rmy = H / 2;
  }

  const auto &data = map.data;
  const int stride = 2;      // subsample for speed
  const int occThresh = 50;  // >=50 -> occupied
  const int clearVal = 0;    // 0 -> free
  const int unknownVal = -1; // -1 -> unknown
  const int clearance = 2;   // cells to avoid around obstacles
  const int infoRadius = 4;  // radius for information gain

  double bestScore = -1.0;
  int bestX = -1, bestY = -1;

  auto isOcc = [&](int x, int y)
  { return data[idxOf(x, y, W)] >= occThresh; };
  auto isFree = [&](int x, int y)
  { return data[idxOf(x, y, W)] == clearVal; };
  auto isUnknown = [&](int x, int y)
  { return data[idxOf(x, y, W)] == unknownVal; };

  for (int y = 1; y < H - 1; y += stride)
  {
    for (int x = 1; x < W - 1; x += stride)
    {
      if (!isFree(x, y))
        continue;
      if (!(isUnknown(x + 1, y) || isUnknown(x - 1, y) || isUnknown(x, y + 1) || isUnknown(x, y - 1)))
        continue; // not a frontier

      bool nearOcc = false;
      for (int dy = -clearance; dy <= clearance && !nearOcc; ++dy)
        for (int dx = -clearance; dx <= clearance && !nearOcc; ++dx)
        {
          int nx = x + dx, ny = y + dy;
          if (nx < 0 || nx >= W || ny < 0 || ny >= H)
            continue;
          nearOcc = nearOcc || isOcc(nx, ny);
        }
      if (nearOcc)
        continue;

      int info = 0;
      for (int dy = -infoRadius; dy <= infoRadius; ++dy)
        for (int dx = -infoRadius; dx <= infoRadius; ++dx)
        {
          int nx = x + dx, ny = y + dy;
          if (nx < 0 || nx >= W || ny < 0 || ny >= H)
            continue;
          if (isUnknown(nx, ny))
            ++info;
        }

      double dist = hypot(static_cast<double>(x - rmx), static_cast<double>(y - rmy));
      if (dist < 1.0)
        dist = 1.0;
      double score = static_cast<double>(info) / dist;
      if (score > bestScore)
      {
        bestScore = score;
        bestX = x;
        bestY = y;
      }
    }
  }

  if (bestScore <= 0.0 || bestX < 0)
    return false;
  mapToWorld(bestX, bestY, map, tx, ty);
  th = atan2f(ty - ry, tx - rx);
  return true;
}
