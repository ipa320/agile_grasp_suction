#include <ros/ros.h>

#include <agile_grasp/grasp_localizer.h>

const std::string CLOUD_TOPIC = "input_cloud";
const std::string CLOUD_FRAME = "camera_rgb_optical_frame";
const std::string END_EFFECTOR_FRAME = "camera_rgb_optical_frame";
const int CLOUD_TYPE = 0;
const int NUM_CLOUDS = 1;
const int NUM_THREADS = 1;
const double WORKSPACE[6] = {-10, 10, -10, 10, -10, 10};
const std::string CLOUD_TYPES[2] = {"sensor_msgs/PointCloud2", "grasp_affordances/CloudSized"};
const std::string PLOT_MODES[3] = {"none", "pcl", "rviz"};


// parameters for segmentation
const double NORMAL_RADIUS_SEARCH = 0.01;
const int NUM_OF_KDTREE_NEIGHBOURS = 30;
const double ANGLE_THRESHHOLD_BETWEEN_NORMALS = 3.5;// in degrees
const double CURVATURE_THRESHOLD = 1.0;
const int MINUMUM_SIZE_OF_CLUSTER_ALLOWED = 300;

// parameters for segmentation (circle detection)
const double MIN_DETECTED_RADIUS = 0.02;
const double MAX_DETECTED_RADIUS = 0.03;
const double SUCTION_GRIPPER_RADIUS = 0.0125;
const double ANGLE_TOLERANCE = 4.0; // [degrees] the tolerance for the circle detection from the given axis
const double NORMAL_DISTANCE_WEIGHT = 0.1;
const int MAX_NR_ITER_CIRCLE_DETECTION = 1000;
const double SEGMENTATION_DIST_THREHOLD = 0.004;
const double AREA_CONSIDERATION_RATIO= 1.0;



int main(int argc, char** argv)
{
  // initialize ROS
  ros::init(argc, argv, "find_grasps");
  ros::NodeHandle node("~");

  GraspLocalizer::ParametersSuction params;

  // read ROS parameters
  std::string cloud_topic;
  std::string cloud_frame;
  std::string end_effector_frame;
  std::vector<double> workspace;
  int cloud_type;

  node.param("cloud_topic", cloud_topic, CLOUD_TOPIC);// name to be searched for, value where info is stored, default
  node.param("cloud_frame", cloud_frame, CLOUD_FRAME);
  node.param("end_effector_frame",end_effector_frame,END_EFFECTOR_FRAME);
  node.param("cloud_type", cloud_type, CLOUD_TYPE);

  // Program parameters
  node.param("num_threads", params.num_threads_, NUM_THREADS);
  node.param("num_clouds", params.num_clouds_, NUM_CLOUDS);

  // parameters for plotting
  node.param("plotting", params.plotting_mode_, 0);
  node.param("marker_lifetime", params.marker_lifetime_, 0.0);

  // parameters for segmentation
  node.param("normal_radius_search", params.normal_radius_search_, NORMAL_RADIUS_SEARCH);
  node.param("num_of_kdTree_neighbours", params.num_of_kdTree_neighbors_, NUM_OF_KDTREE_NEIGHBOURS);
  node.param("angle_threshold_between_normals", params.angle_threshold_between_normals_, ANGLE_THRESHHOLD_BETWEEN_NORMALS);
  node.param("curvature_threshold", params.curvature_threshold_, CURVATURE_THRESHOLD);
  node.param("minimum_size_of_cluster_allowed", params.minimum_size_of_cluster_allowed_, MINUMUM_SIZE_OF_CLUSTER_ALLOWED);

  // parameters for segmentation (circle detection)
  node.param("suction_gripper_radius", params.suction_gripper_radius_, SUCTION_GRIPPER_RADIUS);
  node.param("min_detected_radius", params.min_detected_radius_, MIN_DETECTED_RADIUS);
  node.param("max_detected_radius", params.max_detected_radius_, MAX_DETECTED_RADIUS);
  node.param("angle_tollerance", params.angle_tollerance_, ANGLE_TOLERANCE);
  node.param("normal_distance_weight", params.normal_distance_weight_, NORMAL_DISTANCE_WEIGHT);
  node.param("max_number_of_iterations_circle_detection", params.max_number_of_iterations_circle_detection_, MAX_NR_ITER_CIRCLE_DETECTION);
  node.param("segmentation_distance_threshold", params.segmentation_distance_threshold_, SEGMENTATION_DIST_THREHOLD);
  node.param("area_ratio_to_consider_circle",params.area_consideration_ratio_,AREA_CONSIDERATION_RATIO);
  node.getParam("workspace", workspace);

  Eigen::VectorXd ws(6);
  ws << workspace[0], workspace[1], workspace[2], workspace[3], workspace[4], workspace[5];
  params.workspace_ = ws;



  std::cout << "-- Parameters --\n";
  std::cout << " Input\n";
  std::cout << "  cloud_topic: " << cloud_topic << "\n";
  std::cout << "  cloud_frame: " << cloud_frame << "\n";
  std::cout << "  end_effector_frame: " << cloud_frame << "\n";
  std::cout << "  cloud_type: " << CLOUD_TYPES[cloud_type] << "\n";

  std::cout << " Segmentation(Region Growing)\n";
  std::cout << "  sphere radius used for normal estimation: " << params.normal_radius_search_ << "\n";
  std::cout << "  num of KdTree neighbors: " << params.num_of_kdTree_neighbors_ << "\n";
  std::cout << "  angle threshold between normals: " << params.angle_threshold_between_normals_ << "\n";
  std::cout << "  region curvature threshold: " << params.curvature_threshold_ << "\n";
  std::cout << "  minimum size of cluster allowed: " << params.minimum_size_of_cluster_allowed_ << "\n";
  std::cout << "  area consideration ratio: " << params.area_consideration_ratio_ << "\n";

  std::cout << " Grasp Parameters\n";
  std::cout << "  min detected radius: " << params.min_detected_radius_ << "\n";
  std::cout << "  max detected radius: " << params.max_detected_radius_ << "\n";
  std::cout << "  suction gripper radius_: " << params.suction_gripper_radius_ << "\n";
  std::cout << "  angle tollerance: " << params.angle_tollerance_ << "\n";
  std::cout << "  normal distance weight: " << params.normal_distance_weight_ << "\n";
  std::cout << "  max number of iterations circle detection: " << params.max_number_of_iterations_circle_detection_ << "\n";
  std::cout << "  segmentation distance threshold: " << params.segmentation_distance_threshold_ << "\n";

  std::cout << " Environment parameters\n";
  std::cout << "  workspace: " << ws.transpose() << "\n";

  std::cout << " Program parameters\n";
  std::cout << "  num_threads: " << params.num_threads_ << "\n";
  std::cout << "  num_clouds: " << params.num_clouds_ << "\n";

  std::cout << " Visualization\n";
  std::cout << "  plot_mode: " << PLOT_MODES[params.plotting_mode_] << "\n";
  std::cout << "  marker_lifetime: " << params.marker_lifetime_ << "\n";

  std::string svm_file_name = "";
  GraspLocalizer loc(node, cloud_topic, cloud_frame, end_effector_frame, cloud_type, svm_file_name, params);// there are 2 constructors depending of whcih type of struct params has
  std::cout << "GraspLocalizerObj created...\n";
//  ros::spin();
  loc.findSuctionGrasps();

	return 0;
}
