#include <agile_grasp/grasp_localizer.h>

GraspLocalizer::GraspLocalizer(ros::NodeHandle& node, const std::string& cloud_topic,
	const std::string& cloud_frame, const std::string& end_effector_frame, int cloud_type, const std::string& svm_file_name,
	const ParametersSuction& params)
  : cloud_left_(new PointCloud()), cloud_right_(new PointCloud()),
  cloud_frame_(cloud_frame), end_effector_frame_(end_effector_frame), svm_file_name_(svm_file_name), num_clouds_(params.num_clouds_),
  num_clouds_received_(0), size_left_(0),node_handel_(node)
{
	//trigger service
  service_handel_ = node_handel_.advertiseService("trigger", &GraspLocalizer::trigger_capture, this);
  // subscribe to input point cloud ROS topic
  if (cloud_type == CLOUD_SIZED)
		cloud_sub_ = node.subscribe(cloud_topic, 1, &GraspLocalizer::cloud_sized_callback, this);
	else if (cloud_type == POINT_CLOUD_2)
		cloud_sub_ = node.subscribe(cloud_topic, 1, &GraspLocalizer::cloud_callback, this);
  // create ROS publisher for grasps
  grasps_pub_ = node.advertise<geometry_msgs::PoseArray>("grasps", 10);
  grasps_pub_pose_single_stamped = node.advertise<geometry_msgs::PoseStamped>("PoseStamped_grasps", 10);
  grasps_pub_bba_ = node.advertise<cob_perception_msgs::DetectionArray>("bounding_box_array", 10);
  // create localization object and initialize its parameters
  localization_ = new Localization(params.num_threads_, true, params.plotting_mode_);
  //  localization_->setCameraTransforms(params.cam_tf_left_, params.cam_tf_right_);
  localization_->setWorkspace(params.workspace_);
  /**  Segmentation(Region Growing) */
  localization_->setNormalRadiusSearch(params.normal_radius_search_);
  localization_->setNumOfKdTreeNeighbors(params.num_of_kdTree_neighbors_);
  localization_->setAngleThresholdBetweenNormals(params.angle_threshold_between_normals_);
  localization_->setCurvatureThreshold(params.curvature_threshold_);
  localization_->setMinimumSizeOfClusterAllowed(params.minimum_size_of_cluster_allowed_);
  /**  Parameters for segmentation (circle detection) or hand geometry parameters */
  localization_->setSuctionGripperRadius(params.suction_gripper_radius_);
  localization_->setMinDetectedRadius(params.min_detected_radius_);
  localization_->setMaxDetectedRadius(params.max_detected_radius_);
  localization_->setAngleTollerance(params.angle_tollerance_);
  localization_->setNormalDistanceWeight(params.normal_distance_weight_);
  localization_->setMaxNumberOfIterationsCircleDetection(params.max_number_of_iterations_circle_detection_);
  localization_->setSegmentationDistanceThreshold(params.segmentation_distance_threshold_);
  localization_->setAreaConsiderationRatio(params.area_consideration_ratio_);

  if (params.plotting_mode_ == 0)
  {
		plots_handles_ = false;
	}
	else
	{
		plots_handles_ = false;
		if (params.plotting_mode_ == 2)
			localization_->createVisualsPub(node, params.marker_lifetime_, cloud_frame_);
	}
}


GraspLocalizer::GraspLocalizer(ros::NodeHandle& node, const std::string& cloud_topic, 
	const std::string& cloud_frame, int cloud_type, const std::string& svm_file_name, 
	const ParametersFinger& params)
  : cloud_left_(new PointCloud()), cloud_right_(new PointCloud()), 
  cloud_frame_(cloud_frame), svm_file_name_(svm_file_name), num_clouds_(params.num_clouds_), 
  num_clouds_received_(0), size_left_(0)
{
  // subscribe to input point cloud ROS topic
  if (cloud_type == CLOUD_SIZED)
		cloud_sub_ = node.subscribe(cloud_topic, 1, &GraspLocalizer::cloud_sized_callback, this);
	else if (cloud_type == POINT_CLOUD_2)
		cloud_sub_ = node.subscribe(cloud_topic, 1, &GraspLocalizer::cloud_callback, this);
  
  // create ROS publisher for grasps
  grasps_pub_ = node.advertise<agile_grasp::Grasps>("grasps", 10);
  
  // create localization object and initialize its parameters
  localization_ = new Localization(params.num_threads_, true, params.plotting_mode_);
  localization_->setCameraTransforms(params.cam_tf_left_, params.cam_tf_right_);
  localization_->setWorkspace(params.workspace_);
  localization_->setNumSamples(params.num_samples_);
  localization_->setFingerWidth(params.finger_width_);
  localization_->setHandOuterDiameter(params.hand_outer_diameter_);
  localization_->setHandDepth(params.hand_depth_);
  localization_->setInitBite(params.init_bite_);
  localization_->setHandHeight(params.hand_height_);
		  
  min_inliers_ = params.min_inliers_;
  
  if (params.plotting_mode_ == 0)
  {
		plots_handles_ = false;
	}		
	else
	{
		plots_handles_ = false;		
		if (params.plotting_mode_ == 2)
			localization_->createVisualsPub(node, params.marker_lifetime_, cloud_frame_);
	}
}


void GraspLocalizer::cloud_callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
  if (num_clouds_received_ == num_clouds_)
    return;
  
  // get point cloud from topic
  if (cloud_frame_.compare(msg->header.frame_id) != 0 
			&& cloud_frame_.compare("/" + msg->header.frame_id) != 0)
  {
    std::cout << "Input cloud frame " << msg->header.frame_id << " is not equal to parameter " << cloud_frame_ 
			<< std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (num_clouds_received_ == 0){
    pcl::fromROSMsg(*msg, *cloud_left_);
    message_stamp_ = msg->header.stamp;
    message_frame_id_ = msg->header.frame_id;
  }
  else if (num_clouds_received_ == 1)
    pcl::fromROSMsg(*msg, *cloud_right_);
//   std::cout << "Received cloud # " << num_clouds_received_ << " with " << msg->height * msg->width << " points\n";
  num_clouds_received_++;
}


void GraspLocalizer::cloud_sized_callback(const agile_grasp::CloudSized& msg)
{
  // get point cloud from topic
  if (cloud_frame_.compare(msg.cloud.header.frame_id) != 0)
  {
    std::cout << "Input cloud frame " << msg.cloud.header.frame_id << " is not equal to parameter "
      << cloud_frame_ << std::endl;
    std::exit(EXIT_FAILURE);
  }
  
  pcl::fromROSMsg(msg.cloud, *cloud_left_);
  size_left_ = msg.size_left.data;
  std::cout << "Received cloud with size_left: " << size_left_ << std::endl;
  num_clouds_received_ = 1;
}


void GraspLocalizer::localizeGrasps()
{
  ros::Rate rate(1);
  std::vector<int> indices(0);
  
  while (ros::ok())
  {
    // wait for point clouds to arrive
    if (num_clouds_received_ == num_clouds_)
    {
      // localize grasps
      if (num_clouds_ > 1)
      {
        PointCloud::Ptr cloud(new PointCloud());
        *cloud = *cloud_left_ + *cloud_right_;
        hands_ = localization_->localizeHands(cloud, cloud_left_->size(), indices, false, false);
      }
      else
      {
        hands_ = localization_->localizeHands(cloud_left_, cloud_left_->size(), indices, false, false);
			}
      
      antipodal_hands_ = localization_->predictAntipodalHands(hands_, svm_file_name_);
      handles_ = localization_->findHandles(antipodal_hands_, min_inliers_, 0.005);
      
      // publish handles
      grasps_pub_.publish(createGraspsMsg(handles_));
      ros::Duration(1.0).sleep();
      
      // publish hands contained in handles
      grasps_pub_.publish(createGraspsMsgFromHands(handles_));
      ros::Duration(1.0).sleep();
      
      // reset
      num_clouds_received_ = 0;
    }
    
    ros::spinOnce();
    rate.sleep();
  }
}

bool GraspLocalizer::trigger_capture(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res)
{
//  ROS_INFO("trigger perception");
    trigger_ = true;
	res.success = true;
	res.message = "TRUE";
	ROS_DEBUG("I have received the trigger \n");
	std::cout<<"I have received the trigger \n";
  return true;
}


void GraspLocalizer::findSuctionGrasps()
{
  ros::Rate rate(1.0);
  std::vector<int> indices(0);
  trigger_ = true;

  int loop_counter = 0;
  while (ros::ok())
  {
    // wait for point clouds to arrive
    if (num_clouds_received_ == num_clouds_ && trigger_)
    {
    	std::cout << "The number of PC recived is/are: " <<num_clouds_ << "\n";
      // localize grasps
      if (num_clouds_ > 1 )
      {
//    	pcl::PointCloud<pcl::PointXYZRGB>::Ptr x (new pcl::PointCloud<pcl::PointXYZRGB>());
        PointCloud::Ptr cloud(new PointCloud());
        *cloud = *cloud_left_ + *cloud_right_;
        hands_ = localization_->localizeSuctionGrasps(cloud, cloud_left_->size(), indices, false, false);// last bool here is the planar removal
      }
      else
      {
        hands_ = localization_->localizeSuctionGrasps(cloud_left_, cloud_left_->size(), indices, false, false);
			}
// to be changed
//      antipodal_hands_ = localization_->predictAntipodalHands(hands_, svm_file_name_);
//      handles_ = localization_->findHandles(antipodal_hands_, min_inliers_, 0.005);

      // publish handles
      grasps_pub_.publish(createSuctionGraspsMsg(hands_));
      grasps_pub_bba_.publish(createDetectionArraySuctionMsgs(hands_));
/*      for(int i=0;i<hands_.size();i++)
      {
    	  geometry_msgs::PoseStamped PS;
    	  PS.header.stamp = ros::Time::now();
    	  PS.header.frame_id = message_frame_id_;
    	  Eigen::Matrix3d rot;
    	  rot<< hands_[i].getAxis(),hands_[i].getBinormal(),hands_[i].getApproach();
    	  std::cout<<"the ROT MATRIX IS \n"<<rot <<"\n";
    	  std::cout<<"the Vectors are\n"<<hands_[i].getAxis()<<hands_[i].getBinormal()<<hands_[i].getApproach() <<"\n";
    	  Eigen::Quaterniond q(rot);
    	  q.normalize();
    	  tf::quaternionEigenToMsg(q,PS.pose.orientation);
    	  tf::pointEigenToMsg(hands_[i].getGraspSurface(), PS.pose.position);
    	  grasps_pub_pose_single_stamped.publish(PS);
    	  std::cout<<"please press a key to continue to tf quat frame\n";
    	  cin.get();
    	  tf::Quaternion tfQuat;
    	  tf::Matrix3x3 tfRot(hands_[i].getAxis()[0],hands_[i].getAxis()[1],hands_[i].getAxis()[2],
    			  hands_[i].getBinormal()[0],hands_[i].getBinormal()[1],hands_[i].getBinormal()[2],
				  hands_[i].getApproach()[0],hands_[i].getApproach()[1],hands_[i].getApproach()[2]);
    	  tfRot.getRotation(tfQuat);
    	  tfQuat.normalize();
    	  tf::quaternionTFToMsg(tfQuat,PS.pose.orientation);
    	  tf::pointEigenToMsg(hands_[i].getGraspSurface(), PS.pose.position);
    	  std::cout<<"please press a key to continue to next frame\n";
    	  cin.get();
    	  grasps_pub_pose_single_stamped.publish(PS);
//    	  tf::Transform transform(tfRot);

      }*/
      
//       ros::Duration(1.0).sleep();

//      // publish hands contained in handles
//      grasps_pub_.publish(createGraspsMsgFromHands(handles_));
//      ros::Duration(1.0).sleep();

      // reset
      trigger_ = false;
      num_clouds_received_ = 0;
      loop_counter++;
    }
    else
    {
        num_clouds_received_ = 0;
        ros::spinOnce();
        ros::Duration(0.1).sleep();
    	if(!trigger_){
//     		std::cout << "Trigger has not been recived... \n";
//    		std::cout << "The number of PC recived is/are: " <<num_clouds_ << "\n";
    	}
    else
    	{
//     	std::cout << "Number of clouds recived is no sufficent" << "\n"
// 					<< "recieved: " << num_clouds_received_ << " requierd: "
// 					<< num_clouds_ << "\n";
    	}
    }

    ros::spinOnce();
//     rate.sleep();
  }
}
//
//void GraspLocalizer::PublishTF(const std::vector<GraspHypothesis>& grasps){
//	for(int i=0;i<grasps.size();i++)
//	{
//		tf::Transform transform;
//		Eigen::Vector3d vector = grasps[i].getApproach();
//		Eigen::Vector3d position = grasps[i].getGraspSurface();
//		transform.setOrigin(tf::Vector3(position[0],position[1],position[2]));
//		tf::Quaternion q;
//		transform.setRotation(q);
//		const std::string frame_name = "GraspHyp";
//		br_.sendTransform(tf::StampedTransform(transform, ros::Time::now(), cloud_frame_,frame_name));
//	}
//}

agile_grasp::Grasps GraspLocalizer::createGraspsMsg(const std::vector<GraspHypothesis>& hands)
{
  agile_grasp::Grasps msg;
  
  for (int i = 0; i < hands.size(); i++)
	{
  	msg.grasps.push_back(createGraspMsg(hands[i]));
  }
  
  msg.header.stamp = ros::Time::now();  
  return msg;
}


agile_grasp::Grasp GraspLocalizer::createGraspMsg(const GraspHypothesis& hand)
{
  agile_grasp::Grasp msg;
  tf::vectorEigenToMsg(hand.getGraspBottom(), msg.center);
  tf::vectorEigenToMsg(hand.getAxis(), msg.axis);
  tf::vectorEigenToMsg(hand.getApproach(), msg.approach);
  tf::vectorEigenToMsg(hand.getGraspSurface(), msg.surface_center);
  msg.width.data = hand.getGraspWidth();
  return msg;
}

geometry_msgs::PoseArray GraspLocalizer::createSuctionGraspsMsg(const std::vector<GraspHypothesis>& hands)
{
	geometry_msgs::PoseArray msg;

  for (int i = 0; i < hands.size(); i++)
	{
  	msg.poses.push_back(createSuctionGraspMsg(hands[i]));
  }
  msg.header.stamp = ros::Time::now();
//  msg.header.stamp = message_stamp_;
  msg.header.frame_id = message_frame_id_;
  return msg;
}


geometry_msgs::Pose GraspLocalizer::createSuctionGraspMsg(const GraspHypothesis& hand)
{
  geometry_msgs::Pose pose_msg;
  Eigen::Matrix3d rot;
  rot<< hand.getAxis(),hand.getBinormal(),hand.getApproach();
  Eigen::Quaterniond q(rot);
  q.normalize();
  tf::quaternionEigenToMsg(q,pose_msg.orientation);
  tf::pointEigenToMsg(hand.getGraspSurface(), pose_msg.position);
  return pose_msg;
}

cob_perception_msgs::DetectionArray GraspLocalizer::createDetectionArraySuctionMsgs(
		const std::vector<GraspHypothesis>& hands) {
	geometry_msgs::PoseArray msg_temp = createSuctionGraspsMsg(hands_);
	cob_perception_msgs::DetectionArray bba;
	for (int i = 0; i < hands_.size(); i++) {
		cob_perception_msgs::Detection bb;
		bb.pose.pose = msg_temp.poses[i];
//		bb.pose.header.stamp = message_stamp_;
		bb.pose.header.stamp = ros::Time::now();
		bb.pose.header.frame_id = message_frame_id_;
		bb.header = bb.pose.header;
		bba.detections.push_back(bb);
	}
	bba.header.stamp = message_stamp_;
	bba.header.frame_id = message_frame_id_;

	return bba;
}

agile_grasp::Grasps GraspLocalizer::createGraspsMsgFromHands(const std::vector<Handle>& handles)
{
  agile_grasp::Grasps msg;  
  for (int i = 0; i < handles.size(); i++)
  {
    const std::vector<GraspHypothesis>& hands = handles[i].getHandList();
    const std::vector<int>& inliers = handles[i].getInliers();
    
    for (int j = 0; j < inliers.size(); j++)
    {
      msg.grasps.push_back(createGraspMsg(hands[inliers[j]]));
    }
  }
  msg.header.stamp = ros::Time::now();
  std::cout << "Created grasps msg containing " << msg.grasps.size() << " hands\n";
  return msg;
}


agile_grasp::Grasps GraspLocalizer::createGraspsMsg(const std::vector<Handle>& handles)
{
  agile_grasp::Grasps msg;
  for (int i = 0; i < handles.size(); i++)
    msg.grasps.push_back(createGraspMsg(handles[i]));
  msg.header.stamp = ros::Time::now();
  std::cout << "Created grasps msg containing " << msg.grasps.size() << " handles\n";
  return msg;
}


agile_grasp::Grasp GraspLocalizer::createGraspMsg(const Handle& handle)
{
  agile_grasp::Grasp msg;
  tf::vectorEigenToMsg(handle.getCenter(), msg.center);
  tf::vectorEigenToMsg(handle.getAxis(), msg.axis);
  tf::vectorEigenToMsg(handle.getApproach(), msg.approach);
  tf::vectorEigenToMsg(handle.getHandsCenter(), msg.surface_center);
  msg.width.data = handle.getWidth();
  return msg;
}
