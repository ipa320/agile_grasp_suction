// imports & includes
#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/String.h>
#include <std_srvs/Trigger.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>

/**
 * \brief function called back when trigger is received.
 * calls the crop_PC and voxelize_PC functions
 * \sa voxelize_PC
 * \sa crop_PC
 */
void messgage_callback(const sensor_msgs::PointCloud2ConstPtr& msg);
/**
 * \brief downsample the point cloud
 */
void voxelize_PC (pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud);
/**
 * \brief crops the received PC to the dimensions defined by workspace_
 * \sa workspace_
 */
void crop_PC (pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud);
/**
 * \brief sets the execution flag trigger_flag_
 * this is the function called when the service is called
 * \sa trigger_flag_
 */
bool service_callback(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res);

// internal variables
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_recieved_(new pcl::PointCloud<pcl::PointXYZRGB>); ///<  the point cloud received
std::string message_frame_id_;///< frame ID of the recieved cloud
bool trigger_flag_ = false; ///< the flag denoting if processing should occur
std::string cloud_frame_; ///< cloud frame name
std::vector<double> workspace_; ///<  x min to x max y min to y max z min to z max
double cell_size_; ///<  the cell size of each voxel
int number_of_PC_sampels_; ///<  the number of point clouds to be filtered and published per trigger
ros::Publisher PC_publisher_; ///<  the publisher that publishes the PC after processing
