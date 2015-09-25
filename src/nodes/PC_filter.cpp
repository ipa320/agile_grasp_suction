#include <agile_grasp/PC_filter.h>


void crop_PC (pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud)
{
	ROS_DEBUG("inside crop function");
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr croped_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	ROS_DEBUG("created croped pointcloud empty");
	std::cout<<"the size of the PCL is " <<input_cloud->points.size()<<"\n";
	std::cout<<"the workspace size is " << workspace_.size()<<"\n";
	std::cout<<"the workspace's last entry is " << workspace_[5]<<"\n";
	for (int i = 0; i < input_cloud->points.size(); i++)
		{
			const pcl::PointXYZRGB& p = input_cloud->points[i];
			if (p.x >= workspace_[0] && p.x <= workspace_[1] && p.y >= workspace_[2] && p.y <= workspace_[3]
					&& p.z >= workspace_[4] && p.z <= workspace_[5])
			{
				croped_cloud->points.push_back(p);
			}
		}
//	sensor_msgs::PointCloud2 msg_out;
//	pcl::toROSMsg(*croped_cloud , msg_out);
//		ROS_DEBUG("conversion successful");
//		msg_out.header.frame_id = message_frame_id_;
//		PC_publisher_.publish(msg_out);
	input_cloud = croped_cloud; // overwrite the input clouds pointer
}

void voxelize_PC (pcl::PointCloud<pcl::PointXYZRGB>::Ptr& input_cloud, double cell_size)
{
	ROS_DEBUG("inside Voxelize function");
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr voxelized_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::VoxelGrid<pcl::PointXYZRGB> voxelizer;
	voxelizer.setInputCloud (input_cloud);
	voxelizer.setLeafSize (cell_size, cell_size, cell_size);
	voxelizer.filter (*voxelized_cloud);
	input_cloud = voxelized_cloud;
	ROS_DEBUG("I have sucsessfuly filltered voxelized the cloud");
}


void message_callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
	if(trigger_flag_){
    double t_filter_start = omp_get_wtime();
	ROS_WARN("In message call back, I have recieved the PC_filtration trigger");
	for (int i = 0; i< number_of_PC_sampels_; i++){
	if (cloud_frame_.compare(msg->header.frame_id) != 0
				&& cloud_frame_.compare("/" + msg->header.frame_id) != 0)
	  {
		std::cout << "Input cloud frame " << msg->header.frame_id << " is not equal to parameter " << cloud_frame_
					<< std::endl;
		    std::exit(EXIT_FAILURE);
	  }
	else
	{
		ROS_DEBUG("The cloud has the proper frame ID");
		pcl::fromROSMsg(*msg, *cloud_recieved_); // converting the topic to a PC2 obj
		ROS_DEBUG("The cloud has been properly converted to a cloudobj");
//	    message_stamp_ = msg->header.stamp;
	    message_frame_id_ = msg->header.frame_id;
		// remove NAN points from the cloud
		std::vector<int> nan_indices;
		pcl::removeNaNFromPointCloud(*cloud_recieved_, *cloud_recieved_, nan_indices);
		int size_cloud = cloud_recieved_->size();
		ROS_DEBUG("The NAN points have been removed from the cloud ");
		if(size_cloud == 0)
		{
			ROS_ERROR("The cloud recieved by the PC_filter is empty...");
		}
		/*
		* part used for processing
		*/
		crop_PC (cloud_recieved_);
//		voxelize_PC (cloud_recieved_, cell_size_);
			// do the voxelization
	}
	}
	sensor_msgs::PointCloud2 msg_out;
	ROS_DEBUG("Created sensor_mesg that will be published");
	ROS_DEBUG("Converting PC to sensor msg");
	pcl::toROSMsg(*cloud_recieved_ , msg_out);
	ROS_DEBUG("conversion successful");
	msg_out.header.frame_id = message_frame_id_;
	PC_publisher_.publish(msg_out);
	ROS_DEBUG("publish successful");
	trigger_flag_ = false;
	double t_filter_end = omp_get_wtime();
	double time_elapsed = t_filter_end-t_filter_start;
	std::cout << "The filtration took " << time_elapsed << " seconds \n";
	}
	return;
//	cloud_recieved_ , trigger_flag_
}

bool service_callback(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res)
{
	ROS_WARN("I have recieved the PC_filtration trigger");
	res.message = "The PC_filtration will start now";
	res.success = true;
	trigger_flag_ = true;
	return true;
}
// voxelization function

// cropping function
int main (int argc, char **argv)
{
	// create node
	ros::init(argc, argv, "PC_filter");
//	ros::NodeHandle n;
	ros::NodeHandle node("~"); // this is a provate node handel

	// Variables that may be defined from the launch file
	int rate;
	std::string listen_topic;
	std::string publish_topic;
	std::string service_name; // used for trigger


	// read parameters form the param server
	node.param("rate", rate, 10);
	node.param("number_of_PC_sampels", number_of_PC_sampels_, 5);
	node.param("cloud_frame", cloud_frame_, std::string("/camera_rgb_optical_frame"));
	node.param("listen_topic", listen_topic, std::string("/camera/depth_registered/points"));
	node.param("publish_topic", publish_topic, std::string("agile/filtered_3dPC"));
	node.param("service_name", service_name, std::string("trigger_PC_filter"));
	node.param("cell_size", cell_size_, 0.003);
	node.getParam("workspace", workspace_);
	// THIS MUST BE CHANGED LATER
	workspace_.resize(6);
	workspace_[0] = -0.322843, workspace_[1] = 0.294081, workspace_[2] = -0.336842, workspace_[3] = 0.0437264;
	workspace_[4] = -10,workspace_[5] = 10;
	//

	// creating the publisher and subscriber
	ros::Subscriber PC_subscriber = node.subscribe(listen_topic, int(ceil (35/rate)), message_callback);
	PC_publisher_ = node.advertise<sensor_msgs::PointCloud2>(publish_topic, 10, true);
	ros::ServiceServer trigger_service = node.advertiseService(service_name, service_callback);

	if(!PC_publisher_)
	{
		std::cout<< "There is a problem with the publisher";
	}

	ros::Rate r(rate); // determine the rate of calling
	int loop_iter;
	while (node.ok())
	{
//		PC_publisher.publish();
//		ROS_DEBUG("running %i", loop_iter);
		ros::spinOnce();
		r.sleep();
		++loop_iter;
	}
	return 0;
}
