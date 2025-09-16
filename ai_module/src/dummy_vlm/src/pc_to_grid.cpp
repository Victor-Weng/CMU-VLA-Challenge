#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <nav_msgs/OccupancyGrid.h>
#include <cmath>
#include <algorithm>

// Very simple 2.5D height-projection to a 2D occupancy grid.
// Not a full SLAM; just marks grid cells near observed points as free/occupied.
// Parameters:
//  - input_cloud: topic for terrain / registered point cloud (default: /terrain_map)
//  - map_frame: frame id for the published grid (default: map)
//  - resolution: grid cell size (m)
//  - size_x, size_y: grid size in meters (centered around 0,0)
//  - z_min, z_max: accepted point height range (filtering)
//  - occ_height_threshold: height above ground to consider occupied
//  - free_height_threshold: height range close to ground to mark free

struct Params
{
    std::string input_cloud{"/terrain_map"};
    std::string map_frame{"map"};
    double resolution{0.2};
    double size_x{40.0};
    double size_y{40.0};
    double z_min{-1.0};
    double z_max{2.0};
    double occ_height_threshold{0.25};
    double free_height_threshold{0.15};
    double downsample{0.1};
};

class PcToGrid
{
public:
    PcToGrid(ros::NodeHandle &nh, ros::NodeHandle &pnh) : nh_(nh), pnh_(pnh)
    {
        loadParams();
        pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("/map", 1, true);
        sub_ = nh_.subscribe(params_.input_cloud, 1, &PcToGrid::cloudCb, this);
        setupGrid();
        ROS_INFO("pc_to_grid: subscribing to %s, publishing /map (OccupancyGrid) frame=%s res=%.2f size=%.1fx%.1f m",
                 params_.input_cloud.c_str(), params_.map_frame.c_str(), params_.resolution, params_.size_x, params_.size_y);
    }

private:
    void loadParams()
    {
        pnh_.param<std::string>("input_cloud", params_.input_cloud, params_.input_cloud);
        pnh_.param<std::string>("map_frame", params_.map_frame, params_.map_frame);
        pnh_.param<double>("resolution", params_.resolution, params_.resolution);
        pnh_.param<double>("size_x", params_.size_x, params_.size_x);
        pnh_.param<double>("size_y", params_.size_y, params_.size_y);
        pnh_.param<double>("z_min", params_.z_min, params_.z_min);
        pnh_.param<double>("z_max", params_.z_max, params_.z_max);
        pnh_.param<double>("occ_height_threshold", params_.occ_height_threshold, params_.occ_height_threshold);
        pnh_.param<double>("free_height_threshold", params_.free_height_threshold, params_.free_height_threshold);
        pnh_.param<double>("downsample", params_.downsample, params_.downsample);
    }

    void setupGrid()
    {
        int w = static_cast<int>(std::round(params_.size_x / params_.resolution));
        int h = static_cast<int>(std::round(params_.size_y / params_.resolution));
        grid_.info.resolution = params_.resolution;
        grid_.info.width = w;
        grid_.info.height = h;
        grid_.info.origin.position.x = -0.5 * params_.size_x;
        grid_.info.origin.position.y = -0.5 * params_.size_y;
        grid_.info.origin.position.z = 0.0;
        grid_.info.origin.orientation.w = 1.0;
        grid_.header.frame_id = params_.map_frame;
        grid_.data.assign(w * h, -1);
    }

    inline int idx(int r, int c) const { return r * grid_.info.width + c; }

    void clearGridUnknown()
    {
        std::fill(grid_.data.begin(), grid_.data.end(), -1);
    }

    void cloudCb(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        clearGridUnknown();

        const double res = grid_.info.resolution;
        const double ox = grid_.info.origin.position.x;
        const double oy = grid_.info.origin.position.y;
        const int w = grid_.info.width;
        const int h = grid_.info.height;

        // Optional crude stride based on requested downsample (tuned heuristically)
        int stride = 1;
        if (params_.downsample > 0.19) // ~>= 0.2m
            stride = std::min(8, std::max(2, static_cast<int>(std::floor(params_.downsample / 0.2))));

        // First pass: mark free (near ground) where unknown
        int i = 0;
        for (sensor_msgs::PointCloud2ConstIterator<float> x_it(*msg, "x"), y_it(*msg, "y"), z_it(*msg, "z");
             x_it != x_it.end(); ++x_it, ++y_it, ++z_it, ++i)
        {
            if ((i % stride) != 0)
                continue;
            const float x = *x_it;
            const float y = *y_it;
            const float z = *z_it;
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
                continue;
            if (z < params_.z_min || z > params_.z_max)
                continue;
            int c = static_cast<int>(std::floor((x - ox) / res));
            int r = static_cast<int>(std::floor((y - oy) / res));
            if (r < 0 || r >= h || c < 0 || c >= w)
                continue;
            if (std::fabs(z) <= params_.free_height_threshold)
            {
                int id = idx(r, c);
                if (grid_.data[id] == -1)
                    grid_.data[id] = 0; // mark free if unknown
            }
        }

        // Second pass: mark occupied (wins over free)
        i = 0;
        for (sensor_msgs::PointCloud2ConstIterator<float> x_it(*msg, "x"), y_it(*msg, "y"), z_it(*msg, "z");
             x_it != x_it.end(); ++x_it, ++y_it, ++z_it, ++i)
        {
            if ((i % stride) != 0)
                continue;
            const float x = *x_it;
            const float y = *y_it;
            const float z = *z_it;
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
                continue;
            if (z < params_.z_min || z > params_.z_max)
                continue;
            int c = static_cast<int>(std::floor((x - ox) / res));
            int r = static_cast<int>(std::floor((y - oy) / res));
            if (r < 0 || r >= h || c < 0 || c >= w)
                continue;
            if (z > params_.occ_height_threshold)
            {
                grid_.data[idx(r, c)] = 100; // occupied
            }
        }

        grid_.header.stamp = msg->header.stamp;
        grid_.header.frame_id = params_.map_frame;
        pub_.publish(grid_);
    }

    ros::NodeHandle nh_, pnh_;
    Params params_;
    ros::Subscriber sub_;
    ros::Publisher pub_;
    nav_msgs::OccupancyGrid grid_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pc_to_grid");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    PcToGrid node(nh, pnh);
    ros::spin();
    return 0;
}
