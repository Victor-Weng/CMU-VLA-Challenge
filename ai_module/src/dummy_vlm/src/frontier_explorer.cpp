// ROS
#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <tf/transform_listener.h>

// STL
#include <vector>
#include <queue>
#include <deque>
#include <string>
#include <algorithm>
#include <limits>
#include <cmath>
#include <random>

using std::deque;
using std::queue;
using std::string;
using std::vector;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Frontier
{
    vector<geometry_msgs::Point> cells;
    geometry_msgs::Point centroid;
    double size; // approx area in m^2
};

inline int idx(int row, int col, int width) { return row * width + col; }

class FrontierExplorer
{
public:
    FrontierExplorer(ros::NodeHandle &nh, ros::NodeHandle &pnh)
        : nh_(nh), pnh_(pnh), tf_listener_(), have_map_(false),
          map_frame_("map"), robot_frame_("base_link"), goal_topic_("/next_goal"), map_topic_("/map"), robot_pose_topic_("/state_estimation"),
          free_threshold_(20), occ_threshold_(65), min_frontier_cells_(10),
          fallback_enabled_(true), fb_angle_(0.0), fb_radius_(1.0), fb_radius_max_(5.0), fb_angle_step_(M_PI / 6.0), fb_radius_step_(0.2),
          robot_radius_(0.30), safety_margin_(0.20),
          min_goal_dist_m_(3.0), min_publish_separation_m_(0.5), min_publish_period_s_(3.0),
          unknown_radius_m_(2.0), weight_unknown_(1.0), weight_distance_(0.6),
          current_goal_active_(false),
          switch_improvement_(5.0), goal_timeout_s_(25.0), progress_window_s_(6.0), min_progress_m_(0.30),
          recent_goal_block_s_(12.0), recent_goal_radius_m_(0.6),
          random_step_m_(3.0), random_trials_(8), temp_hold_s_(3.0), temp_goal_active_(false)
    {
        pnh_.param<string>("map_frame", map_frame_, map_frame_);
        pnh_.param<string>("robot_frame", robot_frame_, robot_frame_);
        pnh_.param<string>("robot_pose_topic", robot_pose_topic_, robot_pose_topic_);
        pnh_.param<int>("free_threshold", free_threshold_, free_threshold_);
        pnh_.param<int>("occ_threshold", occ_threshold_, occ_threshold_);
        pnh_.param<int>("min_frontier_cells", min_frontier_cells_, min_frontier_cells_);
        pnh_.param<string>("goal_topic", goal_topic_, goal_topic_);
        pnh_.param<string>("map_topic", map_topic_, map_topic_);
        pnh_.param<bool>("fallback_enabled", fallback_enabled_, fallback_enabled_);
        pnh_.param<double>("fallback_radius_max", fb_radius_max_, fb_radius_max_);
        pnh_.param<double>("robot_radius", robot_radius_, robot_radius_);
        pnh_.param<double>("safety_margin", safety_margin_, safety_margin_);
        pnh_.param<double>("min_goal_dist", min_goal_dist_m_, min_goal_dist_m_);
        pnh_.param<double>("min_publish_separation", min_publish_separation_m_, min_publish_separation_m_);
        pnh_.param<double>("min_publish_period", min_publish_period_s_, min_publish_period_s_);
        pnh_.param<double>("unknown_radius", unknown_radius_m_, unknown_radius_m_);
        pnh_.param<double>("weight_unknown", weight_unknown_, weight_unknown_);
        pnh_.param<double>("weight_distance", weight_distance_, weight_distance_);
        pnh_.param<double>("switch_improvement", switch_improvement_, switch_improvement_);
        pnh_.param<double>("goal_timeout", goal_timeout_s_, goal_timeout_s_);
        pnh_.param<double>("progress_window", progress_window_s_, progress_window_s_);
        pnh_.param<double>("min_progress", min_progress_m_, min_progress_m_);
        pnh_.param<double>("recent_goal_block_s", recent_goal_block_s_, recent_goal_block_s_);
        pnh_.param<double>("recent_goal_radius_m", recent_goal_radius_m_, recent_goal_radius_m_);
        pnh_.param<double>("random_step_m", random_step_m_, random_step_m_);
        pnh_.param<int>("random_trials", random_trials_, random_trials_);
        pnh_.param<double>("random_hold_s", temp_hold_s_, temp_hold_s_);

        map_sub_ = nh_.subscribe<nav_msgs::OccupancyGrid>(map_topic_, 1, &FrontierExplorer::mapCallback, this);
        odom_sub_ = nh_.subscribe<nav_msgs::Odometry>(robot_pose_topic_, 1, &FrontierExplorer::odomCallback, this);
        goal_pub_ = nh_.advertise<geometry_msgs::PoseStamped>(goal_topic_, 1);
        timer_ = nh_.createTimer(ros::Duration(1.0), &FrontierExplorer::onTimer, this);

        rng_.seed(static_cast<unsigned>(ros::Time::now().toNSec() & 0xffffffffu));

        ROS_INFO("FrontierExplorer: map_frame=%s robot_frame=%s map_topic=%s goal_topic=%s",
                 map_frame_.c_str(), robot_frame_.c_str(), map_topic_.c_str(), goal_topic_.c_str());
    }

private:
    // Callbacks
    void odomCallback(const nav_msgs::Odometry::ConstPtr &msg) { last_odom_ = *msg; }
    void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr &msg)
    {
        latest_map_ = *msg;
        have_map_ = true;
    }

    void onTimer(const ros::TimerEvent &)
    {
        // Keep publishing temp random goal briefly
        if (temp_goal_active_)
        {
            if (ros::Time::now() < temp_goal_until_)
            {
                if (shouldPublish(temp_goal_))
                    publishTempGoal(temp_goal_);
                return;
            }
            temp_goal_active_ = false;
        }

        // No map yet? Fallback circle exploration using TF pose
        if (!have_map_)
        {
            if (!fallback_enabled_)
            {
                ROS_WARN_THROTTLE(5.0, "FrontierExplorer: waiting for map on %s", map_topic_.c_str());
                return;
            }
            double rx = 0, ry = 0, yaw = 0;
            if (!getRobotPose(rx, ry, yaw))
            {
                ROS_WARN_THROTTLE(2.0, "FrontierExplorer: fallback waiting for TF (%s -> %s)", map_frame_.c_str(), robot_frame_.c_str());
                return;
            }
            geometry_msgs::Point p;
            p.x = rx + fb_radius_ * std::cos(fb_angle_);
            p.y = ry + fb_radius_ * std::sin(fb_angle_);
            p.z = 0.0;
            publishGoal(p);
            fb_angle_ += fb_angle_step_;
            if (fb_angle_ > 2.0 * M_PI)
                fb_angle_ -= 2.0 * M_PI;
            fb_radius_ = std::min(fb_radius_ + fb_radius_step_, fb_radius_max_);
            ROS_INFO_THROTTLE(2.0, "FrontierExplorer(fallback): exploratory goal r=%.1f", fb_radius_);
            return;
        }

        // Get robot pose
        double rx = 0, ry = 0, yaw = 0;
        if (!getRobotPose(rx, ry, yaw))
        {
            ROS_WARN_THROTTLE(2.0, "FrontierExplorer: could not get robot pose (%s -> %s)", map_frame_.c_str(), robot_frame_.c_str());
            return;
        }

        // Detect and cluster frontiers
        vector<geometry_msgs::Point> frontier_cells = detectFrontiers(latest_map_);
        vector<Frontier> frontiers = clusterFrontiers(latest_map_, frontier_cells);
        if (frontiers.empty())
        {
            ROS_WARN_THROTTLE(2.0, "FrontierExplorer: no frontiers; trying random exploration");
            geometry_msgs::Point rp;
            double cx = current_goal_active_ ? current_goal_.x : rx;
            double cy = current_goal_active_ ? current_goal_.y : ry;
            if (computeRandomGoal(latest_map_, cx, cy, rx, ry, rp))
            {
                temp_goal_ = rp;
                temp_goal_until_ = ros::Time::now() + ros::Duration(temp_hold_s_);
                temp_goal_active_ = true;
                publishTempGoal(temp_goal_);
            }
            return;
        }

        // Choose a goal
        geometry_msgs::Point goal;
        bool ok = selectGoal(latest_map_, frontiers, rx, ry, goal);
        if (!ok)
        {
            ROS_WARN_THROTTLE(2.0, "FrontierExplorer: no suitable frontier goal; trying random exploration");
            geometry_msgs::Point rp;
            double cx = current_goal_active_ ? current_goal_.x : rx;
            double cy = current_goal_active_ ? current_goal_.y : ry;
            if (computeRandomGoal(latest_map_, cx, cy, rx, ry, rp))
            {
                temp_goal_ = rp;
                temp_goal_until_ = ros::Time::now() + ros::Duration(temp_hold_s_);
                temp_goal_active_ = true;
                publishTempGoal(temp_goal_);
            }
            return;
        }

        // If no current goal, set and publish
        if (!current_goal_active_)
        {
            setCurrentGoal(goal, rx, ry, latest_map_);
            publishGoal(goal);
            return;
        }

        // Commitment and progress checks
        double now_s = ros::Time::now().toSec();
        bool timeout = (now_s - current_goal_time_.toSec()) > goal_timeout_s_;
        bool no_progress = false;
        if ((now_s - last_progress_check_.toSec()) > progress_window_s_)
        {
            double dist_now = std::hypot(current_goal_.x - rx, current_goal_.y - ry);
            if ((last_progress_dist_ - dist_now) < min_progress_m_)
                no_progress = true;
            last_progress_dist_ = dist_now;
            last_progress_check_ = ros::Time::now();
        }

        double cand_score = scorePoint(latest_map_, rx, ry, goal);
        double curr_score = scorePoint(latest_map_, rx, ry, current_goal_);
        bool significantly_better = (cand_score + switch_improvement_) < curr_score;

        if (timeout || no_progress || significantly_better)
        {
            if (isRecentlyUsed(goal))
            {
                ROS_INFO_THROTTLE(2.0, "FrontierExplorer: candidate near recent goal; publishing random temp goal");
                geometry_msgs::Point rp;
                double cx = current_goal_active_ ? current_goal_.x : rx;
                double cy = current_goal_active_ ? current_goal_.y : ry;
                if (computeRandomGoal(latest_map_, cx, cy, rx, ry, rp))
                {
                    temp_goal_ = rp;
                    temp_goal_until_ = ros::Time::now() + ros::Duration(temp_hold_s_);
                    temp_goal_active_ = true;
                    publishTempGoal(temp_goal_);
                    return;
                }
            }
            else
            {
                setCurrentGoal(goal, rx, ry, latest_map_);
                publishGoal(goal);
                return;
            }
        }

        if (!shouldPublish(current_goal_))
        {
            ROS_INFO_THROTTLE(2.0, "FrontierExplorer: keeping current goal (stable)");
            return;
        }
        publishGoal(current_goal_);
    }

    bool getRobotPose(double &x, double &y, double &yaw)
    {
        const string &target_frame = map_frame_;
        const string &source_frame = robot_frame_;
        tf::StampedTransform transform;
        try
        {
            tf_listener_.waitForTransform(target_frame, source_frame, ros::Time(0), ros::Duration(0.2));
            tf_listener_.lookupTransform(target_frame, source_frame, ros::Time(0), transform);
        }
        catch (const tf::TransformException &ex)
        {
            vector<string> candidates = {robot_frame_, string("base_link"), string("base_footprint"), string("base")};
            bool tf_ok = false;
            for (const auto &cand : candidates)
            {
                try
                {
                    tf_listener_.waitForTransform(target_frame, cand, ros::Time(0), ros::Duration(0.1));
                    tf_listener_.lookupTransform(target_frame, cand, ros::Time(0), transform);
                    if (cand != robot_frame_)
                    {
                        ROS_WARN("FrontierExplorer: using alternative robot_frame '%s' (param was '%s')", cand.c_str(), robot_frame_.c_str());
                        robot_frame_ = cand;
                    }
                    tf_ok = true;
                    break;
                }
                catch (const tf::TransformException &)
                {
                }
            }
            if (!tf_ok)
            {
                ROS_WARN("TF lookup failed: %s", ex.what());
                if (!last_odom_.header.frame_id.empty())
                {
                    const string &odom_frame = last_odom_.header.frame_id;
                    if (odom_frame == map_frame_ || (!latest_map_.header.frame_id.empty() && odom_frame == latest_map_.header.frame_id))
                    {
                        x = last_odom_.pose.pose.position.x;
                        y = last_odom_.pose.pose.position.y;
                        double roll, pitch;
                        tf::Quaternion q;
                        tf::quaternionMsgToTF(last_odom_.pose.pose.orientation, q);
                        tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
                        ROS_INFO_THROTTLE(2.0, "FrontierExplorer: using Odometry pose (frame=%s) as fallback", odom_frame.c_str());
                        return true;
                    }
                }
                return false;
            }
        }
        x = transform.getOrigin().x();
        y = transform.getOrigin().y();
        double roll, pitch;
        tf::Matrix3x3(transform.getRotation()).getRPY(roll, pitch, yaw);
        return true;
    }

    vector<geometry_msgs::Point> detectFrontiers(const nav_msgs::OccupancyGrid &map)
    {
        vector<geometry_msgs::Point> cells;
        const int width = static_cast<int>(map.info.width);
        const int height = static_cast<int>(map.info.height);
        const double res = map.info.resolution;
        const double ox = map.info.origin.position.x;
        const double oy = map.info.origin.position.y;

        auto inBounds = [&](int r, int c)
        { return (r >= 0 && r < height && c >= 0 && c < width); };
        for (int r = 0; r < height; ++r)
        {
            for (int c = 0; c < width; ++c)
            {
                int8_t occ = map.data[idx(r, c, width)];
                if (occ < 0 || occ > free_threshold_)
                    continue;
                bool adjacent_to_unknown = false;
                for (int dr = -1; dr <= 1 && !adjacent_to_unknown; ++dr)
                {
                    for (int dc = -1; dc <= 1 && !adjacent_to_unknown; ++dc)
                    {
                        if (dr == 0 && dc == 0)
                            continue;
                        int nr = r + dr, nc = c + dc;
                        if (!inBounds(nr, nc))
                            continue;
                        if (map.data[idx(nr, nc, width)] < 0)
                            adjacent_to_unknown = true;
                    }
                }
                if (!adjacent_to_unknown)
                    continue;
                geometry_msgs::Point p;
                p.x = ox + (static_cast<double>(c) + 0.5) * res;
                p.y = oy + (static_cast<double>(r) + 0.5) * res;
                p.z = 0.0;
                cells.push_back(p);
            }
        }
        return cells;
    }

    vector<Frontier> clusterFrontiers(const nav_msgs::OccupancyGrid &map, const vector<geometry_msgs::Point> &cells_world)
    {
        vector<Frontier> clusters;
        if (cells_world.empty())
            return clusters;
        const int width = static_cast<int>(map.info.width);
        const int height = static_cast<int>(map.info.height);
        const double res = map.info.resolution;
        const double ox = map.info.origin.position.x;
        const double oy = map.info.origin.position.y;
        vector<uint8_t> is_frontier(static_cast<size_t>(width * height), 0);
        for (const auto &wp : cells_world)
        {
            int c = static_cast<int>(std::floor((wp.x - ox) / res));
            int r = static_cast<int>(std::floor((wp.y - oy) / res));
            if (r >= 0 && r < height && c >= 0 && c < width)
                is_frontier[idx(r, c, width)] = 1;
        }
        vector<uint8_t> visited(static_cast<size_t>(width * height), 0);
        auto inBounds = [&](int r, int c)
        { return (r >= 0 && r < height && c >= 0 && c < width); };
        for (int r = 0; r < height; ++r)
        {
            for (int c = 0; c < width; ++c)
            {
                int linear = idx(r, c, width);
                if (!is_frontier[linear] || visited[linear])
                    continue;
                queue<std::pair<int, int> > q;
                q.push({r, c});
                visited[linear] = 1;
                vector<geometry_msgs::Point> cluster_cells;
                while (!q.empty())
                {
                    auto rc = q.front();
                    q.pop();
                    int cr = rc.first, cc = rc.second;
                    geometry_msgs::Point wp;
                    wp.x = ox + (static_cast<double>(cc) + 0.5) * res;
                    wp.y = oy + (static_cast<double>(cr) + 0.5) * res;
                    wp.z = 0.0;
                    cluster_cells.push_back(wp);
                    for (int dr = -1; dr <= 1; ++dr)
                    {
                        for (int dc = -1; dc <= 1; ++dc)
                        {
                            if (dr == 0 && dc == 0)
                                continue;
                            int nr = cr + dr, nc = cc + dc;
                            if (!inBounds(nr, nc))
                                continue;
                            int nlin = idx(nr, nc, width);
                            if (is_frontier[nlin] && !visited[nlin])
                            {
                                visited[nlin] = 1;
                                q.push({nr, nc});
                            }
                        }
                    }
                }
                if (static_cast<int>(cluster_cells.size()) >= min_frontier_cells_)
                {
                    Frontier f;
                    f.cells = std::move(cluster_cells);
                    double sx = 0, sy = 0;
                    for (const auto &p : f.cells)
                    {
                        sx += p.x;
                        sy += p.y;
                    }
                    double inv_n = 1.0 / static_cast<double>(f.cells.size());
                    f.centroid.x = sx * inv_n;
                    f.centroid.y = sy * inv_n;
                    f.centroid.z = 0.0;
                    f.size = static_cast<double>(f.cells.size()) * (res * res);
                    clusters.push_back(std::move(f));
                }
            }
        }
        return clusters;
    }

    bool hasClearance(const nav_msgs::OccupancyGrid &map, int r, int c, int clearance_cells) const
    {
        const int w = static_cast<int>(map.info.width);
        const int h = static_cast<int>(map.info.height);
        for (int dr = -clearance_cells; dr <= clearance_cells; ++dr)
        {
            for (int dc = -clearance_cells; dc <= clearance_cells; ++dc)
            {
                int nr = r + dr, nc = c + dc;
                if (nr < 0 || nr >= h || nc < 0 || nc >= w)
                    continue;
                if (dr * dr + dc * dc > clearance_cells * clearance_cells)
                    continue;
                int8_t occ = map.data[idx(nr, nc, w)];
                if (occ >= occ_threshold_)
                    return false;
            }
        }
        return true;
    }

    bool lineClear(const nav_msgs::OccupancyGrid &map, int r0, int c0, int r1, int c1, int clearance_cells) const
    {
        int dr = std::abs(r1 - r0);
        int dc = std::abs(c1 - c0);
        int sr = (r0 < r1) ? 1 : -1;
        int sc = (c0 < c1) ? 1 : -1;
        int err = (dr > dc ? dr : -dc) / 2;
        int r = r0, c = c0;
        while (true)
        {
            if (!hasClearance(map, r, c, clearance_cells))
                return false;
            if (r == r1 && c == c1)
                break;
            int e2 = err;
            if (e2 > -dr)
            {
                err -= dc;
                r += sr;
            }
            if (e2 < dc)
            {
                err += dr;
                c += sc;
            }
        }
        return true;
    }

    geometry_msgs::Point chooseSafeGoal(const nav_msgs::OccupancyGrid &map, const Frontier &frontier, double rx, double ry)
    {
        geometry_msgs::Point best_p = frontier.centroid;
        double best_score = std::numeric_limits<double>::infinity();
        const double res = map.info.resolution;
        const double ox = map.info.origin.position.x;
        const double oy = map.info.origin.position.y;
        const int w = static_cast<int>(map.info.width);
        const int h = static_cast<int>(map.info.height);
        const int clearance_cells = std::max(1, static_cast<int>(std::floor((robot_radius_ + safety_margin_) / res)));
        const int unknown_rad_cells = std::max(1, static_cast<int>(std::floor(unknown_radius_m_ / res)));
        const double min_goal_dist2 = min_goal_dist_m_ * min_goal_dist_m_;
        for (const auto &p : frontier.cells)
        {
            int c = static_cast<int>(std::floor((p.x - ox) / res));
            int r = static_cast<int>(std::floor((p.y - oy) / res));
            if (r < 0 || r >= h || c < 0 || c >= w)
                continue;
            int8_t occ = map.data[idx(r, c, w)];
            if (occ < 0 || occ > free_threshold_)
                continue;
            if (!hasClearance(map, r, c, clearance_cells))
                continue;
            double dx = p.x - rx, dy = p.y - ry;
            double d2 = dx * dx + dy * dy;
            if (d2 < min_goal_dist2)
                continue;
            int unknown_cnt = 0;
            for (int dr = -unknown_rad_cells; dr <= unknown_rad_cells; ++dr)
                for (int dc = -unknown_rad_cells; dc <= unknown_rad_cells; ++dc)
                {
                    int nr = r + dr, nc = c + dc;
                    if (nr < 0 || nr >= h || nc < 0 || nc >= w)
                        continue;
                    if (dr * dr + dc * dc > unknown_rad_cells * unknown_rad_cells)
                        continue;
                    if (map.data[idx(nr, nc, w)] < 0)
                        unknown_cnt++;
                }
            int r0 = static_cast<int>(std::floor((ry - oy) / res));
            int c0 = static_cast<int>(std::floor((rx - ox) / res));
            if (r0 < 0 || r0 >= h || c0 < 0 || c0 >= w)
                continue;
            if (!lineClear(map, r0, c0, r, c, clearance_cells))
                continue;
            double score = weight_distance_ * std::sqrt(d2) - weight_unknown_ * static_cast<double>(unknown_cnt);
            if (score < best_score)
            {
                best_score = score;
                best_p = p;
            }
        }
        if (best_score != std::numeric_limits<double>::infinity())
            return best_p;
        geometry_msgs::Point p = frontier.centroid;
        double vx = p.x - rx, vy = p.y - ry;
        double vnorm = std::hypot(vx, vy);
        if (vnorm < 1e-3)
        {
            vx = 1.0;
            vy = 0.0;
            vnorm = 1.0;
        }
        vx /= vnorm;
        vy /= vnorm;
        int max_steps = std::max(3, clearance_cells * 3);
        for (int k = 1; k <= max_steps; ++k)
        {
            geometry_msgs::Point q;
            q.x = rx + vx * res * (min_goal_dist_m_ / res + k);
            q.y = ry + vy * res * (min_goal_dist_m_ / res + k);
            q.z = 0.0;
            int c = static_cast<int>(std::floor((q.x - ox) / res));
            int r = static_cast<int>(std::floor((q.y - oy) / res));
            if (r < 0 || r >= h || c < 0 || c >= w)
                continue;
            int8_t occ = map.data[idx(r, c, w)];
            if (occ >= 0 && occ <= free_threshold_ && hasClearance(map, r, c, clearance_cells))
                return q;
        }
        return frontier.centroid;
    }

    void publishGoal(const geometry_msgs::Point &p)
    {
        geometry_msgs::PoseStamped goal;
        goal.header.frame_id = map_frame_;
        goal.header.stamp = ros::Time::now();
        goal.pose.position = p;
        goal.pose.orientation.x = 0.0;
        goal.pose.orientation.y = 0.0;
        goal.pose.orientation.z = 0.0;
        goal.pose.orientation.w = 1.0;
        goal_pub_.publish(goal);
        last_goal_ = p;
        last_goal_time_ = goal.header.stamp;
        recent_goals_.push_back({p, goal.header.stamp});
        while (!recent_goals_.empty())
        {
            if ((goal.header.stamp - recent_goals_.front().second).toSec() > recent_goal_block_s_)
                recent_goals_.pop_front();
            else
                break;
        }
        ROS_INFO("FrontierExplorer: published next goal at (%.2f, %.2f)", p.x, p.y);
    }

    void publishTempGoal(const geometry_msgs::Point &p)
    {
        geometry_msgs::PoseStamped goal;
        goal.header.frame_id = map_frame_;
        goal.header.stamp = ros::Time::now();
        goal.pose.position = p;
        goal.pose.orientation.x = 0.0;
        goal.pose.orientation.y = 0.0;
        goal.pose.orientation.z = 0.0;
        goal.pose.orientation.w = 1.0;
        goal_pub_.publish(goal);
        last_goal_ = p;
        last_goal_time_ = goal.header.stamp;
        ROS_INFO("FrontierExplorer: published TEMP random goal at (%.2f, %.2f)", p.x, p.y);
    }

    bool shouldPublish(const geometry_msgs::Point &p)
    {
        if (last_goal_time_.isZero())
            return true;
        double dt = (ros::Time::now() - last_goal_time_).toSec();
        if (dt < min_publish_period_s_)
            return false;
        double dx = p.x - last_goal_.x, dy = p.y - last_goal_.y;
        double d2 = dx * dx + dy * dy;
        return d2 >= (min_publish_separation_m_ * min_publish_separation_m_);
    }

    bool selectGoal(const nav_msgs::OccupancyGrid &map, const vector<Frontier> &frontiers, double rx, double ry, geometry_msgs::Point &out)
    {
        struct Cand
        {
            int idx;
            double score;
        };
        vector<Cand> order;
        order.reserve(frontiers.size());
        for (int i = 0; i < static_cast<int>(frontiers.size()); ++i)
        {
            const auto &c = frontiers[i].centroid;
            double dx = c.x - rx, dy = c.y - ry;
            double dist = std::hypot(dx, dy);
            double score = dist - 0.5 * std::sqrt(std::max(0.0, frontiers[i].size));
            order.push_back({i, score});
        }
        std::sort(order.begin(), order.end(), [](const Cand &a, const Cand &b)
                  { return a.score < b.score; });
        for (const auto &c : order)
        {
            geometry_msgs::Point g = chooseSafeGoal(map, frontiers[c.idx], rx, ry);
            out = g;
            return true;
        }
        return false;
    }

    // Sample a random reachable goal around a center (cx,cy) while validating from robot (rx,ry)
    bool computeRandomGoal(const nav_msgs::OccupancyGrid &map, double cx, double cy, double rx, double ry, geometry_msgs::Point &out)
    {
        const double res = map.info.resolution;
        const double ox = map.info.origin.position.x;
        const double oy = map.info.origin.position.y;
        const int w = static_cast<int>(map.info.width);
        const int h = static_cast<int>(map.info.height);
        const int clearance_cells = std::max(1, static_cast<int>(std::floor((robot_radius_ + safety_margin_) / res)));
        std::uniform_real_distribution<double> ang_dist(0.0, 2.0 * M_PI);
        std::uniform_real_distribution<double> jitter_dist(0.0, M_PI / 2.0);
        std::uniform_int_distribution<int> sign_dist(0, 1);
        int r0 = static_cast<int>(std::floor((ry - oy) / res));
        int c0 = static_cast<int>(std::floor((rx - ox) / res));
        if (r0 < 0 || r0 >= h || c0 < 0 || c0 >= w)
            return false;
        for (int t = 0; t < random_trials_; ++t)
        {
            double theta = ang_dist(rng_);
            for (int j = 0; j < 2; ++j)
            {
                double qx = cx + random_step_m_ * std::cos(theta);
                double qy = cy + random_step_m_ * std::sin(theta);
                int r1 = static_cast<int>(std::floor((qy - oy) / res));
                int c1 = static_cast<int>(std::floor((qx - ox) / res));
                if (r1 >= 0 && r1 < h && c1 >= 0 && c1 < w)
                {
                    int8_t occ = map.data[idx(r1, c1, w)];
                    if (occ >= 0 && occ <= free_threshold_ && hasClearance(map, r1, c1, clearance_cells) && lineClear(map, r0, c0, r1, c1, clearance_cells))
                    {
                        out.x = qx;
                        out.y = qy;
                        out.z = 0.0;
                        return true;
                    }
                }
                double delta = jitter_dist(rng_);
                theta += (sign_dist(rng_) ? delta : -delta);
            }
        }
        return false;
    }

    // Backward-compatible helper: sample around the robot position
    bool computeRandomGoal(const nav_msgs::OccupancyGrid &map, double rx, double ry, geometry_msgs::Point &out)
    {
        return computeRandomGoal(map, rx, ry, rx, ry, out);
    }

    double scorePoint(const nav_msgs::OccupancyGrid &map, double rx, double ry, const geometry_msgs::Point &p) const
    {
        const double res = map.info.resolution;
        const double ox = map.info.origin.position.x;
        const double oy = map.info.origin.position.y;
        const int w = static_cast<int>(map.info.width);
        const int h = static_cast<int>(map.info.height);
        int c = static_cast<int>(std::floor((p.x - ox) / res));
        int r = static_cast<int>(std::floor((p.y - oy) / res));
        if (r < 0 || r >= h || c < 0 || c >= w)
            return std::numeric_limits<double>::infinity();
        double dx = p.x - rx, dy = p.y - ry;
        double dist = std::hypot(dx, dy);
        const int unknown_rad_cells = std::max(1, static_cast<int>(std::floor(unknown_radius_m_ / res)));
        int unknown_cnt = 0;
        for (int dr = -unknown_rad_cells; dr <= unknown_rad_cells; ++dr)
            for (int dc = -unknown_rad_cells; dc <= unknown_rad_cells; ++dc)
            {
                int nr = r + dr, nc = c + dc;
                if (nr < 0 || nr >= h || nc < 0 || nc >= w)
                    continue;
                if (dr * dr + dc * dc > unknown_rad_cells * unknown_rad_cells)
                    continue;
                if (map.data[idx(nr, nc, w)] < 0)
                    unknown_cnt++;
            }
        return weight_distance_ * dist - weight_unknown_ * static_cast<double>(unknown_cnt);
    }

    bool isRecentlyUsed(const geometry_msgs::Point &p) const
    {
        double r2 = recent_goal_radius_m_ * recent_goal_radius_m_;
        for (const auto &pr : recent_goals_)
        {
            double dx = p.x - pr.first.x, dy = p.y - pr.first.y;
            if (dx * dx + dy * dy <= r2)
                return true;
        }
        return false;
    }

    void setCurrentGoal(const geometry_msgs::Point &p, double rx, double ry, const nav_msgs::OccupancyGrid &)
    {
        current_goal_ = p;
        current_goal_time_ = ros::Time::now();
        current_goal_active_ = true;
        last_progress_check_ = current_goal_time_;
        last_progress_dist_ = std::hypot(p.x - rx, p.y - ry);
    }

private:
    ros::NodeHandle nh_, pnh_;
    tf::TransformListener tf_listener_;
    ros::Subscriber map_sub_, odom_sub_;
    ros::Publisher goal_pub_;
    ros::Timer timer_;

    nav_msgs::OccupancyGrid latest_map_;
    nav_msgs::Odometry last_odom_;
    bool have_map_;

    string map_frame_, robot_frame_, goal_topic_, map_topic_, robot_pose_topic_;
    int free_threshold_, occ_threshold_, min_frontier_cells_;
    bool fallback_enabled_;
    double fb_angle_, fb_radius_, fb_radius_max_, fb_angle_step_, fb_radius_step_;
    double robot_radius_, safety_margin_;
    double min_goal_dist_m_, min_publish_separation_m_, min_publish_period_s_;
    double unknown_radius_m_, weight_unknown_, weight_distance_;

    deque<std::pair<geometry_msgs::Point, ros::Time> > recent_goals_;
    bool current_goal_active_;
    geometry_msgs::Point current_goal_;
    ros::Time current_goal_time_, last_progress_check_;
    double last_progress_dist_;
    double switch_improvement_, goal_timeout_s_, progress_window_s_, min_progress_m_;
    double recent_goal_block_s_, recent_goal_radius_m_;

    geometry_msgs::Point last_goal_;
    ros::Time last_goal_time_;

    std::mt19937 rng_;
    double random_step_m_;
    int random_trials_;
    double temp_hold_s_;
    bool temp_goal_active_;
    geometry_msgs::Point temp_goal_;
    ros::Time temp_goal_until_;
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
