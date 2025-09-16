#include <ros/ros.h>                   // ros lib
#include <nav_msgs/OccupancyGrid.h>    // occupancy grid
#include <geometry_msgs/PoseStamped.h> // to publish next goal (pos + header)
#include <geometry_msgs/Point.h>       // store cell coordinates in frontier structs
#include <tf/transform_listener.h>     // TF Listener to query the robot pose
// Include standard headers for utilities (vectors, queues, limits).
#include <vector>
#include <queue>
#include <limits>
#include <cmath>
#include <string>

using std::queue;
using std::size_t;
using std::string;
using std::vector;

// define struct for a frontier

struct Frontier
{
    // list of cells in this frontier
    vector<geometry_msgs::Point> cells;
    // Centroid of the frontier in world coordinates, avg of cell points
    geometry_msgs::Point centroid;
    // size as area = cell_count*res^2
    double size;
};

// convert 2D grid indices of row,col into a flat array index
// grind cells stored in OccupancyGrid::data as a 1D vector of int8_t
inline int idx(int row, int col, int width)
{ // inline makes global var safe
    // linear index = row*width + col
    return row * width + col;
}

// class for frontier detection, clustering, and goal selection logic
class FrontierExplorer
{
public: // accessible outside the class
    // constructor: set up ROS interfaces and parameters
    FrontierExplorer(ros::NodeHandle &nh, ros::NodeHandle &pnh)
        : nh_(nh), pnh_(pnh),
          tf_listener_(),            // transform frames from camera and lidar to our map
          have_map_(false),          // no map by default
          map_frame_("map"),         // default map frame name
          robot_frame_("base_link"), // default robot frame name
          free_threshold_(20),       // occupancy <= 20 = free, change this later
          occ_threshold_(65),        // >= 65 = occupied
          min_frontier_cells_(10),   // minimum cluster size to consider a valid frontier
          goal_topic_("/next_goal"), // topic to publish PoseStamped goals
          map_topic_("/map")         // topic to subscribe for occupany grid
    {
        // Allow overriding defaults via private parameters (from launch file / param server)
        pnh_.param<string>("map_frame", map_frame_, map_frame_);
        pnh_.param<string>("robot_frame", robot_frame_, robot_frame_);
        pnh_.param<int>("free_threshold", free_threshold_, free_threshold_);
        pnh_.param<int>("occ_threshold", occ_threshold_, occ_threshold_);
        pnh_.param<int>("min_frontier_cells", min_frontier_cells_, min_frontier_cells_);
        pnh_.param<string>("goal_topic", goal_topic_, goal_topic_);
        pnh_.param<string>("map_topic", map_topic_, map_topic_);

        // Subscribe to the occupancy grid map; queue size 1 since we only need the latest map.
        map_sub_ = nh_.subscribe<nav_msgs::OccupancyGrid>(
            map_topic_, 1, &FrontierExplorer::mapCallback, this);

        // Advertise the next-goal topic where we publish the selected frontier as a PoseStamped.
        goal_pub_ = nh_.advertise<geometry_msgs::PoseStamped>(goal_topic_, 1);

        // Create a periodic timer to run the frontier pipeline at a fixed frequency (e.g., 1 Hz).
        timer_ = nh_.createTimer(ros::Duration(1.0), &FrontierExplorer::onTimer, this);

        // Log configuration so the operator knows what frames and thresholds are active.
        ROS_INFO("FrontierExplorer: map_frame=%s robot_frame=%s map_topic=%s goal_topic=%s",
                 map_frame_.c_str(), robot_frame_.c_str(), map_topic_.c_str(), goal_topic_.c_str());
    }

private:
    // Callback when a new map (OccupancyGrid) message arrives.
    void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr &msg)
    {
        // Store the received map message (copy) so we can process it in the timer callback.
        latest_map_ = *msg;
        // Mark that we now have at least one valid map.
        have_map_ = true;
    }
    // Timer callback executed periodically to detect frontiers and publish a goal.
    void onTimer(const ros::TimerEvent &)
    {
        // If we have no map yet, do nothing.
        if (!have_map_)
        {
            ROS_WARN_THROTTLE(5.0, "FrontierExplorer: waiting for map on %s", map_topic_.c_str());
            return;
        }

        // Attempt to get the current robot pose in the map frame.
        double rx = 0.0, ry = 0.0, ryaw = 0.0;
        if (!getRobotPose(rx, ry, ryaw))
        {
            // If pose lookup fails (e.g., TF not ready), skip this cycle.
            ROS_WARN_THROTTLE(2.0, "FrontierExplorer: could not get robot pose (%s -> %s)",
                              map_frame_.c_str(), robot_frame_.c_str());
            return;
        }

        // Run frontier detection to get a vector of individual frontier cell coordinates (world frame).
        vector<geometry_msgs::Point> frontier_cells = detectFrontiers(latest_map_);

        // Cluster the frontier cells into connected regions (frontiers) using BFS/DFS.
        vector<Frontier> frontiers = clusterFrontiers(latest_map_, frontier_cells);

        // If no usable frontiers were found, log and return.
        if (frontiers.empty())
        {
            ROS_WARN_THROTTLE(2.0, "FrontierExplorer: no frontiers detected");
            return;
        }

        // Select the best frontier (closest centroid to robot in Euclidean distance).
        Frontier best = selectBestFrontier(frontiers, rx, ry);

        // Publish the chosen frontier centroid as a PoseStamped goal in the map frame.
        publishGoal(best.centroid);
    }

    // Look up the robot pose (x, y, yaw) in the map frame using TF.
    bool getRobotPose(double &x, double &y, double &yaw)
    {
        // Define the source and target frames for the transform.
        const string &target_frame = map_frame_;
        const string &source_frame = robot_frame_;
        // Create a stamped transform object to receive the transform data.
        tf::StampedTransform transform;
        try
        {
            // Wait up to 0.2 seconds for the transform to become available.
            tf_listener_.waitForTransform(target_frame, source_frame, ros::Time(0), ros::Duration(0.2));
            // Retrieve the latest available transform from source_frame to target_frame.
            tf_listener_.lookupTransform(target_frame, source_frame, ros::Time(0), transform);
        }
        catch (const tf::TransformException &ex)
        {
            // If TF throws, log the reason and return false.
            ROS_WARN("TF lookup failed: %s", ex.what());
            return false;
        }
        // Extract translation (x, y) from the transform origin.
        x = transform.getOrigin().x();
        y = transform.getOrigin().y();
        // Extract yaw (rotation about Z) from the quaternion.
        double roll, pitch;
        tf::Matrix3x3(transform.getRotation()).getRPY(roll, pitch, yaw);
        // Return success.
        return true;
    }

    // Detect individual frontier cells in the map and return them as world-frame points.
    vector<geometry_msgs::Point> detectFrontiers(const nav_msgs::OccupancyGrid &map)
    {
        // Prepare an output vector to accumulate frontier points.
        vector<geometry_msgs::Point> cells;
        // Cache map metadata for convenience.
        const int width = static_cast<int>(map.info.width);   // number of columns
        const int height = static_cast<int>(map.info.height); // number of rows
        const double res = map.info.resolution;               // cell resolution (meters per cell)
        const double ox = map.info.origin.position.x;         // world origin x (map frame)
        const double oy = map.info.origin.position.y;         // world origin y (map frame)

        // Lambda that checks whether a grid coordinate is inside bounds.
        auto inBounds = [&](int r, int c) -> bool
        {
            // Valid rows: [0, height), valid cols: [0, width).
            return (r >= 0 && r < height && c >= 0 && c < width);
        };

        // Iterate over every grid cell to test if it’s a frontier.
        for (int r = 0; r < height; ++r)
        {
            for (int c = 0; c < width; ++c)
            {
                // Read the occupancy value at (r, c). Values: -1 unknown, 0 free, 100 occupied (or scaled).
                int8_t occ = map.data[idx(r, c, width)];
                // Frontier rule prerequisite: current cell must be "free enough".
                if (occ < 0 || occ > free_threshold_)
                {
                    // Skip unknown or not-free cells.
                    continue;
                }
                // Check the 8-neighborhood for at least one unknown neighbor.
                bool adjacent_to_unknown = false;
                for (int dr = -1; dr <= 1 && !adjacent_to_unknown; ++dr)
                {
                    for (int dc = -1; dc <= 1 && !adjacent_to_unknown; ++dc)
                    {
                        // Skip the center cell itself (dr=0, dc=0).
                        if (dr == 0 && dc == 0)
                            continue;
                        // Compute neighbor row and column.
                        int nr = r + dr;
                        int nc = c + dc;
                        // Only consider neighbors inside bounds.
                        if (!inBounds(nr, nc))
                            continue;
                        // If a neighbor is unknown (< 0), current free cell is a frontier cell.
                        if (map.data[idx(nr, nc, width)] < 0)
                        {
                            adjacent_to_unknown = true;
                        }
                    }
                }
                // If not adjacent to unknown, this free cell is not a frontier; continue.
                if (!adjacent_to_unknown)
                    continue;

                // Convert this grid cell to world coordinates at cell center:
                // world_x = ox + (c + 0.5) * res, world_y = oy + (r + 0.5) * res.
                geometry_msgs::Point p;
                p.x = ox + (static_cast<double>(c) + 0.5) * res;
                p.y = oy + (static_cast<double>(r) + 0.5) * res;
                p.z = 0.0; // 2D map, z = 0
                // Append to the list of frontier cells.
                cells.push_back(p);
            }
        }
        // Return the collected frontier cells in world frame.
        return cells;
    }

    // Cluster the frontier cells into connected components (frontiers) using BFS connectivity.
    vector<Frontier> clusterFrontiers(const nav_msgs::OccupancyGrid &map,
                                      const vector<geometry_msgs::Point> &cells_world)
    {
        // Prepare output vector for clusters (frontiers).
        vector<Frontier> clusters;
        // Quick exit: if no cells, return empty.
        if (cells_world.empty())
            return clusters;

        // We'll need to map world points back to grid coords to perform connectivity on the grid.
        // Cache map metadata for conversions.
        const int width = static_cast<int>(map.info.width);
        const int height = static_cast<int>(map.info.height);
        const double res = map.info.resolution;
        const double ox = map.info.origin.position.x;
        const double oy = map.info.origin.position.y;

        // Frontier mask grid: mark which cells are frontier (1) vs not (0).
        vector<uint8_t> is_frontier(static_cast<size_t>(width * height), 0);
        // For each world point in cells_world, mark the corresponding grid cell as frontier.
        for (const auto &wp : cells_world)
        {
            // Convert world coords back to integer grid indices: col = floor((x - ox)/res), row similarly.
            int c = static_cast<int>(std::floor((wp.x - ox) / res));
            int r = static_cast<int>(std::floor((wp.y - oy) / res));
            // Bounds check to be safe.
            if (r >= 0 && r < height && c >= 0 && c < width)
            {
                is_frontier[idx(r, c, width)] = 1;
            }
        }

        // Visited mask to avoid reprocessing cells during BFS.
        vector<uint8_t> visited(static_cast<size_t>(width * height), 0);

        // Lambda to test in-bounds grid coordinates.
        auto inBounds = [&](int r, int c) -> bool
        {
            return (r >= 0 && r < height && c >= 0 && c < width);
        };

        // Iterate over all grid cells; when we find an unvisited frontier cell, grow a cluster from it.
        for (int r = 0; r < height; ++r)
        {
            for (int c = 0; c < width; ++c)
            {
                // Skip if not a frontier cell or already visited.
                int linear = idx(r, c, width);
                if (!is_frontier[linear] || visited[linear])
                    continue;

                // Start a BFS from (r, c) to gather all connected frontier cells (8-connectivity).
                queue<std::pair<int, int> > q;
                q.push({r, c});
                visited[linear] = 1;

                // Accumulate cells (in world coords) into this cluster.
                vector<geometry_msgs::Point> cluster_cells;

                // Run BFS until the queue is empty.
                while (!q.empty())
                {
                    // Pop the front cell from the queue.
                    auto rc = q.front();
                    q.pop();
                    // Extract row and col of the current cell.
                    int cr = rc.first;
                    int cc = rc.second;

                    // Convert current grid cell to world coordinates (cell center) and store in cluster.
                    geometry_msgs::Point wp;
                    wp.x = ox + (static_cast<double>(cc) + 0.5) * res;
                    wp.y = oy + (static_cast<double>(cr) + 0.5) * res;
                    wp.z = 0.0;
                    cluster_cells.push_back(wp);

                    // Explore 8-connected neighbors to grow the cluster.
                    for (int dr = -1; dr <= 1; ++dr)
                    {
                        for (int dc = -1; dc <= 1; ++dc)
                        {
                            // Skip center.
                            if (dr == 0 && dc == 0)
                                continue;
                            // Neighbor coordinates.
                            int nr = cr + dr;
                            int nc = cc + dc;
                            // Bounds check.
                            if (!inBounds(nr, nc))
                                continue;
                            // Compute linear index.
                            int nlin = idx(nr, nc, width);
                            // Enqueue neighbor if it is an unvisited frontier cell.
                            if (is_frontier[nlin] && !visited[nlin])
                            {
                                visited[nlin] = 1;
                                q.push({nr, nc});
                            }
                        }
                    }
                } // end BFS

                // If the cluster is large enough, compute centroid and size and add it to the list.
                if (static_cast<int>(cluster_cells.size()) >= min_frontier_cells_)
                {
                    Frontier f;
                    f.cells = std::move(cluster_cells);
                    // Compute centroid by averaging all world points in the cluster.
                    double sx = 0.0, sy = 0.0;
                    for (const auto &p : f.cells)
                    {
                        sx += p.x;
                        sy += p.y;
                    }
                    const double inv_n = 1.0 / static_cast<double>(f.cells.size());
                    f.centroid.x = sx * inv_n;
                    f.centroid.y = sy * inv_n;
                    f.centroid.z = 0.0;
                    // Define size as physical area covered by frontier cells (approx cell_count * res^2).
                    f.size = static_cast<double>(f.cells.size()) * (res * res);
                    // Append this frontier cluster to the output list.
                    clusters.push_back(std::move(f));
                }
                // Else (cluster too small), discard it silently.
            }
        }

        // Return all detected frontier clusters.
        return clusters;
    }

    // Choose the best frontier: here we pick the centroid closest to the robot.
    Frontier selectBestFrontier(const vector<Frontier> &frontiers, double rx, double ry)
    {
        // Initialize best index and best distance with sentinel values.
        int best_idx = -1;
        double best_dist = std::numeric_limits<double>::infinity();
        // Iterate over all frontiers and compute robot-to-centroid distance.
        for (int i = 0; i < static_cast<int>(frontiers.size()); ++i)
        {
            const auto &c = frontiers[i].centroid;
            const double dx = c.x - rx;
            const double dy = c.y - ry;
            const double d2 = dx * dx + dy * dy; // squared distance (avoids sqrt)
            // Keep the smallest distance encountered.
            if (d2 < best_dist)
            {
                best_dist = d2;
                best_idx = i;
            }
        }
        // If no valid frontier found (shouldn't happen if frontiers non-empty), return the first.
        if (best_idx < 0)
            return frontiers.front();
        // Return the best frontier by index.
        return frontiers[best_idx];
    }

    // Publish a PoseStamped at the given world point (map frame) as the next goal.
    void publishGoal(const geometry_msgs::Point &p)
    {
        // Prepare a PoseStamped message (position + header).
        geometry_msgs::PoseStamped goal;
        // Fill header frame so downstream knows the frame of reference (map).
        goal.header.frame_id = map_frame_;
        // Stamp with current time for freshness.
        goal.header.stamp = ros::Time::now();
        // Set position to the chosen centroid.
        goal.pose.position = p;
        // Set a neutral orientation (yaw=0) using a unit quaternion (w=1 means no rotation).
        goal.pose.orientation.x = 0.0;
        goal.pose.orientation.y = 0.0;
        goal.pose.orientation.z = 0.0;
        goal.pose.orientation.w = 1.0;
        // Publish on the configured topic for the planner to consume.
        goal_pub_.publish(goal);
        // Log for visibility.
        ROS_INFO("FrontierExplorer: published next goal at (%.2f, %.2f)", p.x, p.y);
    }

private:
    // Node handles (global and private) for parameters and topic setup.
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    // TF listener to query transforms between frames (map -> base_link).
    tf::TransformListener tf_listener_;

    // Subscriber for the occupancy grid map.
    ros::Subscriber map_sub_;
    // Publisher for the next frontier goal.
    ros::Publisher goal_pub_;
    // Timer for periodic processing of frontier detection and goal selection.
    ros::Timer timer_;

    // Latest map and a flag indicating whether we have received one.
    nav_msgs::OccupancyGrid latest_map_;
    bool have_map_;

    // Frames and topics configurable via parameters.
    string map_frame_;
    string robot_frame_;
    string goal_topic_;
    string map_topic_;

    // Frontier detection parameters (thresholds and cluster size filter).
    int free_threshold_;
    int occ_threshold_;
    int min_frontier_cells_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "frontier_explorer");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    FrontierExplorer explorer(nh, pnh);

    ros::spin();
    return 0;
}
