#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <string>
#include <cmath>
#include <boost/bind.hpp>

// This node synchronizes camera images and LiDAR point clouds and republishes
// paired messages when either a min period elapses or the robot has moved
// farther than a threshold since the last publish.
// Params:
//  - image_topic (default: /camera/image)
//  - lidar_topic (default: /registered_scan)
//  - odom_topic  (default: /state_estimation)
//  - min_period_sec (default: 0.5) minimal seconds between outputs (0 disables)
//  - min_distance_m (default: 0.0) distance movement trigger (0 disables)
// Outputs:
//  - /paired/image (sensor_msgs/Image)
//  - /paired/points (sensor_msgs/PointCloud2)
// The two outputs share the stamp of the synchronized input pair.

class ImageLidarCapture
{
public:
    ImageLidarCapture(ros::NodeHandle &nh, ros::NodeHandle &pnh)
        : nh_(nh), pnh_(pnh)
    {
        pnh_.param<std::string>("image_topic", image_topic_, std::string("/camera/image"));
        pnh_.param<std::string>("lidar_topic", lidar_topic_, std::string("/registered_scan"));
        pnh_.param<std::string>("odom_topic", odom_topic_, std::string("/state_estimation"));
        pnh_.param<double>("min_period_sec", min_period_sec_, 0.5);
        pnh_.param<double>("min_distance_m", min_distance_m_, 0.0);

        pub_image_ = nh_.advertise<sensor_msgs::Image>("/paired/image", 2, true);
        pub_points_ = nh_.advertise<sensor_msgs::PointCloud2>("/paired/points", 2, true);

        // Odometry for distance gating
        odom_sub_ = nh_.subscribe<nav_msgs::Odometry>(odom_topic_, 50, &ImageLidarCapture::odomCb, this);

        // Synchronized inputs
        img_sub_.subscribe(nh_, image_topic_, 10);
        pc_sub_.subscribe(nh_, lidar_topic_, 10);

        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> SyncPolicy;
        sync_.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(50), img_sub_, pc_sub_));
        sync_->registerCallback(boost::bind(&ImageLidarCapture::pairCb, this, _1, _2));

        last_pub_time_ = ros::Time(0);
        have_last_pose_ = false;
    }

private:
    void odomCb(const nav_msgs::Odometry::ConstPtr &msg)
    {
        last_odom_ = *msg;
    }

    static double dist2D(const nav_msgs::Odometry &a, const nav_msgs::Odometry &b)
    {
        const double dx = a.pose.pose.position.x - b.pose.pose.position.x;
        const double dy = a.pose.pose.position.y - b.pose.pose.position.y;
        return std::sqrt(dx * dx + dy * dy);
    }

    void pairCb(const sensor_msgs::ImageConstPtr &img, const sensor_msgs::PointCloud2ConstPtr &pc)
    {
        const ros::Time now_stamp = img->header.stamp;

        // Time gating
        const bool period_ok = (min_period_sec_ <= 0.0) || ((now_stamp - last_pub_time_).toSec() >= min_period_sec_);

        // Distance gating: only evaluate if we have odom newer than last publish
        bool distance_ok = (min_distance_m_ <= 0.0);
        if (!distance_ok && last_pub_time_.toSec() > 0.0)
        {
            // Consider odometry at/after this pair's stamp if available
            if (!have_last_pose_)
            {
                last_pose_ = last_odom_;
                have_last_pose_ = true;
            }
            // If last_odom_ is sufficiently recent, compute distance since last_pose_
            const double d = dist2D(last_odom_, last_pose_);
            if (d >= min_distance_m_)
            {
                distance_ok = true;
            }
        }
        else if (!distance_ok && last_pub_time_.toSec() == 0.0)
        {
            // First publish allowed
            distance_ok = true;
        }

        if (!(period_ok || distance_ok))
        {
            return;
        }

        // Publish the synchronized pair using the same timestamp
        sensor_msgs::Image img_out = *img;
        sensor_msgs::PointCloud2 pc_out = *pc;

        // Optionally set frame consistency; keep original frames as they are meaningful
        img_out.header.stamp = now_stamp;
        pc_out.header.stamp = now_stamp;

        pub_image_.publish(img_out);
        pub_points_.publish(pc_out);

        last_pub_time_ = now_stamp;
        last_pose_ = last_odom_;
        have_last_pose_ = true;
    }

    ros::NodeHandle nh_, pnh_;
    std::string image_topic_;
    std::string lidar_topic_;
    std::string odom_topic_;
    double min_period_sec_;
    double min_distance_m_;

    message_filters::Subscriber<sensor_msgs::Image> img_sub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> pc_sub_;
    boost::shared_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> > > sync_;

    ros::Subscriber odom_sub_;
    nav_msgs::Odometry last_odom_;
    nav_msgs::Odometry last_pose_;
    bool have_last_pose_;
    ros::Time last_pub_time_;

    ros::Publisher pub_image_;
    ros::Publisher pub_points_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "image_lidar_capture");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    ImageLidarCapture node(nh, pnh);
    ros::spin();
    return 0;
}
