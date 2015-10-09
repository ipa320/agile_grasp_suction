/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2015, Andreas ten Pas
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef LOCALIZATION_H_
#define LOCALIZATION_H_

// system dependencies
#include <iostream>
#include <math.h>
#include <set>
#include <string>
#include <time.h>

// PCL dependencies
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
// PCL used for region growing
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>
// PCL used for circle detection
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_circle3d.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/ModelCoefficients.h>

// project dependencies
#include <agile_grasp/grasp_hypothesis.h>
#include <agile_grasp/hand_search.h>
#include <agile_grasp/handle.h>
#include <agile_grasp/handle_search.h>
#include <agile_grasp/learning.h>
#include <agile_grasp/plot.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudRGB;
#include <pcl/surface/convex_hull.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/filters/project_inliers.h>

// these includes have been added to try out the normal estimation using polonomial fitting
#include <pcl/common/common.h>
#include <pcl/io/obj_io.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/grid_projection.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/organized_fast_mesh.h>
#include <pcl/surface/ear_clipping.h>
#include <pcl/surface/poisson.h>
#include <pcl/common/common.h>
#include <boost/random.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/make_shared.hpp>
// ****
#include <pcl/io/obj_io.h>
#include <pcl/TextureMesh.h>
#include <pcl/surface/texture_mapping.h>


/** Localization class
 *
 * \brief High level interface for the localization of grasp hypotheses and handles
 * 
 * This class provides a high level interface to search for grasp hypotheses.
 * The class does the preprocessing of the pointcloud, the localization of grasps, postprocessing of the grasps
 * ploting of the results (seperate thread inside localization) and interfaces to the Plot class to plot to Rviz.
 * 
*/
class Localization
{
public:
	struct callback_args{
	  // structure used to pass arguments to the callback function
	  PointCloud::Ptr clicked_points_cloud;
	  pcl::visualization::PCLVisualizer::Ptr viewerPtr;
	};

	/**
	 * \brief Default Constructor.
	 * currently not being used
	*/
	Localization() : num_threads_(1), plotting_mode_(1), plots_camera_sources_(false), cloud_(new PointCloud), cloud_rgb_(new PointCloudRGB)
	{ }
	
	/**
	 * \brief Constructor.
	 * Note that the parameters for the localization obj must be set using the setters and getters. For an example check the GraspLocalizer constructor.
	 * \param num_threads the number of threads to be used in the search
	 * \param filter_boundaries whether grasp hypotheses that are close to the point cloud boundaries are filtered out
	 * \param plots_hands whether grasp hypotheses are plotted
	*/
	Localization(int num_threads, bool filters_boundaries, int plotting_mode) :
			num_threads_(num_threads), filters_boundaries_(filters_boundaries), 
      plotting_mode_(plotting_mode), plots_camera_sources_(false), cloud_rgb_(new PointCloudRGB),
      cloud_(new PointCloud), viewer_comb_(new pcl::visualization::PCLVisualizer ("Algorithim output"))
	{
		first_plot_ = true;
		viewer_point_indicies_.resize(4);
		for(int i=0; i<viewer_point_indicies_.size();i++)
		{
			viewer_point_indicies_[i] = i;
		}

		 viewer_comb_->createViewPort (0.0, 0.5, 0.5, 1.0, viewer_point_indicies_[0]);
		 viewer_comb_->createViewPort (0.5, 0.5, 1.0, 1.0, viewer_point_indicies_[1]);
		 viewer_comb_->createViewPort (0.0, 0.0, 0.5, 0.5, viewer_point_indicies_[2]);
		 viewer_comb_->createViewPort (0.5, 0.0, 1.0, 0.5, viewer_point_indicies_[3]);
		 viewer_comb_->close();
	}
	
	/**
	 * \brief Find handles given a list of grasp hypotheses.
	 * \param hand_list the list of grasp hypotheses
	 * \param min_inliers the minimum number of handle inliers
	 * \param min_length the minimum length of the handle
	 * \param is_plotting whether the handles are plotted
	*/
	std::vector<Handle> findHandles(const std::vector<GraspHypothesis>& hand_list, int min_inliers,	double min_length);
	
	/**
	 * \brief Predict antipodal grasps given a list of grasp hypotheses.
	 * \param hand_list the list of grasp hypotheses
	 * \param svm_filename the location and filename of the SVM
	*/
	std::vector<GraspHypothesis> predictAntipodalHands(const std::vector<GraspHypothesis>& hand_list,
			const std::string& svm_filename);
	
	/**
	 * \brief Localize hands in a given point cloud. this is the main localize suction grasp function
	 * \param cloud_in the input point cloud
	 * \param indices the set of point cloud indices for which point neighborhoods are found
	 * \param calculates_antipodal whether the grasp hypotheses are checked for being antipodal
	 * \param uses_clustering whether clustering is used for processing the point cloud
	 * \return the list of grasp hypotheses found
	*/
	std::vector<GraspHypothesis> localizeSuctionGrasps(const PointCloudRGB::Ptr& cloud_in, int size_left,
			const std::vector<int>& indices, bool calculates_antipodal, bool uses_clustering,bool plot_on_flag = true);

	/**
	 * \brief Localize hands in a given point cloud. this is the main localizeHands function
	 * \param cloud_in the input point cloud
	 * \param indices the set of point cloud indices for which point neighborhoods are found
	 * \param calculates_antipodal whether the grasp hypotheses are checked for being antipodal
	 * \param uses_clustering whether clustering is used for processing the point cloud
	 * \return the list of grasp hypotheses found
	*/
	std::vector<GraspHypothesis> localizeHands(const PointCloud::Ptr& cloud_in, int size_left,
			const std::vector<int>& indices, bool calculates_antipodal, bool uses_clustering);
	
	/**
	 * \brief Localize hands given two point cloud files.
	 * \param pcd_filename_left the first point cloud file location and name
	 * \param pcd_filename_right the second point cloud file location and name
	 * \param calculates_antipodal whether the grasp hypotheses are checked for being antipodal
	 * \param uses_clustering whether clustering is used for processing the point cloud
	 * \return the list of grasp hypotheses found
	*/
	std::vector<GraspHypothesis> localizeHands(const std::string& pcd_filename_left,
			const std::string& pcd_filename_right, bool calculates_antipodal = false, 
      bool uses_clustering = false, bool use_suction=false);
	
	/**
	 * \brief Localize hands given two point cloud files and a set of point cloud indices.
	 * \param pcd_filename_left the first point cloud file location and name
	 * \param pcd_filename_right the second point cloud file location and name
	 * \param indices the set of point cloud indices for which point neighborhoods are found
	 * \param calculates_antipodal whether the grasp hypotheses are checked for being antipodal
	 * \param uses_clustering whether clustering is used for processing the point cloud
	 * \return the list of grasp hypotheses found
	*/
	std::vector<GraspHypothesis> localizeHands(const std::string& pcd_filename_left,
			const std::string& pcd_filename_right, const std::vector<int>& indices, 
      bool calculates_antipodal =	false, bool uses_clustering = false, bool use_suction=false);
	
	/**
	 * \brief Set the camera poses.
	 * \param cam_tf_left the pose of the left camera
	 * \param cam_tf_right the pose of the right camera
	*/
	void setCameraTransforms(const Eigen::Matrix4d& cam_tf_left, const Eigen::Matrix4d& cam_tf_right)
	{
		cam_tf_left_ = cam_tf_left;
		cam_tf_right_ = cam_tf_right;
	}
	
	/**
	 * \brief Return the camera pose of one given camera.
	 * \param is_left true if the pose of the left camera is wanted, false if the pose of the right camera is wanted
	 * \return the pose of the camera specified by @p is_left
	*/
	const Eigen::Matrix4d& getCameraTransform(bool is_left)
	{
		if (is_left)
			return cam_tf_left_;
		else
			return cam_tf_right_;
	}
	
	/**
	* \brief calculate the normals of the entier pointcloud or the subcloud defined by the indiceis
	* \param cloud_in the input cloud
	* \param the KD tree
	* \param radius used to calculate the normals
	* \param the Pointer to the memory address where the output will go
	*/
	void CalculateNormalsForPointCloud(
			PointCloudRGB::Ptr& cloud_in,
			pcl::search::KdTree<pcl::PointXYZRGB>::Ptr& tree,
			double normal_radius_search,
			pcl::PointCloud<pcl::Normal>::Ptr& cloud_normals);

	/**
	* \brief calculate the normals of the subcloud defined by the indiceis
	*
	* \param cloud_in the input cloud
	* \param the KD tree
	* \param radius used to calculate the normals
	* \param the Pointer to the memory address where the output will go
	*/
	void CalculateNormalsForPointCloud(
			PointCloud::Ptr& cloud_in,
			pcl::search::KdTree<pcl::PointXYZ>::Ptr& tree,
			double normal_radius_search,
			pcl::PointCloud<pcl::Normal>::Ptr& cloud_normals,
			pcl::PointIndices::Ptr& indiceis);

	/**
	* \brief calculate the normals of the subcloud defined by the indiceis
	*
	* \param cloud_in the input cloud
	* \param the KD tree
	* \param radius used to calculate the normals
	* \param the Pointer to the memory address where the output will go
	* \return a vector containing the point indicies of each cluster
	* \return the segmented point cloud in segmented_colored_pc
	*/
	std::vector <pcl::PointIndices> ClusterUsingRegionGrowing(
			const PointCloudRGB::Ptr& cloud_in,
			const pcl::search::KdTree<pcl::PointXYZRGB>::Ptr& tree,
			const double normal_radius_search,
			const pcl::PointCloud<pcl::Normal>::Ptr& cloud_normals,
			const double angle_threshold_between_normals,
			const double curvature_threshold,
			const int num_of_kdTree_neighbours,
			const int min_size_of_cluster_allowed,
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr& segmented_colored_pc,
			const int max_size_of_cluster_allowed = 1000000);
	/**
	 * \brief extract a circle from each cluster
	 * \param clusters the clusters for which a circle will be fitted
	 * \param cloud the cloud from which the subcloud (cluster) will be extracted
	 * \param cloud_normals the normals of the cloud
	 * \return a vector of circle coefficients
	 * \return a vector of point indices corresponding to the points that are considered as circle inliers
	 */
	void CircleExtraction(
			std::vector <pcl::PointIndices>& clusters,
			PointCloudRGB::Ptr& cloud,
			pcl::PointCloud<pcl::Normal>::Ptr& cloud_normals,
			std::vector<pcl::PointIndices>& circle_inliners_of_all_clusters,
			std::vector<pcl::ModelCoefficients>& circle_coefficients_of_all_clusters);
	/**
	 * \brief this function is a container function containing all the grasp filtration functions that are to be used.
	 * changes the input parameters
	 * \param min_detected_radius will be read from the parameter server via find_suction_grasp.cpp
	 * \param area_consideration_ratio will be read from the parameter server via find_suction_grasp.cpp
	 * \sa find_suction_grasp.cpp
	 */
	void GraspFiltration(const PointCloudRGB::Ptr& cloud,
			std::vector<pcl::PointIndices>& circle_inliners_of_all_clusters,
			std::vector<pcl::ModelCoefficients>& circle_coefficients_of_all_clusters,
			double min_detected_radius = 0,
			double area_consideration_ratio = 0);
	/**
	 * \brief filters suction grasps according to area.
	 * If area is smaller than area min the circle coefficients and inliners are resized to 0
	 * the current implementation depends on finding the minimum area of the current circle that
	 * would hold the full suction_gripper (circle) inside. This minimum area is calculated as the minimum area
	 *  of a circle that will hold the complete sunction_gripper inside it (to visualize draw 2 concentric circles 1 smaller than the other)
	 * \param segmentation_distance_threshold will be read from the parameter server via find_suction_grasp.cpp
	 * \param area_consideration_ratio will be read from the parameter server via find_suction_grasp.cpp
	 * \param min_detected_radius will be read from the parameter server via find_suction_grasp.cpp
	 * \param suction_gripper_radius will be read from the parameter server via find_suction_grasp.cpp
	 */
	void FiltrationAccToArea(const PointCloudRGB::Ptr& cloud,
			std::vector<pcl::PointIndices>& circle_inliners_of_all_clusters,
			std::vector<pcl::ModelCoefficients>& circle_coefficients_of_all_clusters,
			double min_detected_radius = 0,
			double area_consideration_ratio = 0,
			double segmentation_distance_threshold = 0,
			double suction_gripper_radius = 0);
	/**
	 * \brief this function is a container function containing all the Post Processing functions that are to be used.
	 * changes the input parameters
	 * \sa GraspingVectorDirectionCorrection
	 */
	void PostProcessing(std::vector<pcl::PointIndices>& circle_inliners_of_all_clusters,
			std::vector<pcl::ModelCoefficients>& circle_coefficients_of_all_clusters,
			std::vector <pcl::PointIndices>& clusters, PointCloudRGB::Ptr& cloud);
	/**
	 * \brief aligns all the suction grasp vectors towards the camera
	 * changes input parameters
	 */
	void GraspingVectorDirectionCorrection(const std::vector<pcl::PointIndices>& circle_inliners_of_all_clusters,
			std::vector<pcl::ModelCoefficients>& circle_coefficients_of_all_clusters);
	/**
	 * \brief calculates a right hand coordinate system for each grasp from the grasp direction vector.
	 * the grasp direction vector is the vector normal to the circle and is found inside the circle_coefficients
	 */
	void CoodinateSystemCalculation(const std::vector<pcl::PointIndices>& circle_inliners_of_all_clusters,
			const std::vector<pcl::ModelCoefficients>& circle_coefficients_of_all_clusters,
			std::vector <pcl::PointIndices>& clusters,
			PointCloudRGB::Ptr& cloud,
			std::vector<GraspHypothesis>& suction_grasp_hyp_list);
	/**
	 * \brief visualizes the pointcloud at different stages of processing
	 * it is seperated into 4 parts as follows starting from the upper left and clockwise
	 * - Upper left (1) is the raw cloud
	 * - upper right (2)  is the preprocessed cloud
	 * - lower left (3)is the region segmented cloud
	 * - lower right (4) cloud is where the grasps are plotted as cylinders
	 */
	void visualize();
	/**
	 * \brief starts the visualization thread.
	 * The thread is never joined since it should always run
	 * \sa visualize
	 */
	void plot_thread_start();

	void plot_thread_join();

	/**
	 * \brief call back method called when a point in the point cloud is clicked, responsible for cropping the point cloud (by manually changing the workspace)
	 * clicks are registered by pressing shift and left click. When a point is clicked the coordinates of the point are displayed in the terminal and the points are colored red in the PCL visualizer.
	 * If more than 2 points have been clicked the work space is set to be the largest box containing all the clicked points.
	 * If more than 6 points have been clicked the workspace is reset and awaits new clicks
	 */
	void point_pick_callback (const pcl::visualization::PointPickingEvent& event, void* args);
	/**
	 * \brief Set the dimensions of the robot's workspace.
	 * \param workspace 1x6 vector containing the robot's workspace dimensions
	*/
	void setWorkspace(const Eigen::VectorXd& workspace)
	{
		workspace_ = workspace;
	}
	
	void setSuctionGripperRadius(const double suction_gripper_radius)
	{
		suction_gripper_radius_ = suction_gripper_radius;
	}

	/**
	 * \brief Set the number of samples to be used for the search.
	 * \param num_samples the number of samples to be used for the search
	*/
	void setNumSamples(int num_samples)
	{
		num_samples_ = num_samples;
	}
	
	/**
	 * \brief Set the radius to be used for the point neighborhood search in the hand search.
	 * \param nn_radius_hands the radius to be used for the point neighborhood search
	*/
	void setNeighborhoodRadiusHands(double nn_radius_hands)
	{
		nn_radius_hands_ = nn_radius_hands;
	}
	
	/**
	 * \brief Set the radius to be used for the point neighborhood search in the quadric fit.
	 * \param nn_radius_hands the radius to be used for the point neighborhood search
	*/
	void setNeighborhoodRadiusTaubin(double nn_radius_taubin)
	{
		nn_radius_taubin_ = nn_radius_taubin;
	}
	
	/**
	 * \brief Set the finger width of the robot hand.
	 * \param finger_width the finger width
	*/
	void setFingerWidth(double finger_width)
	{
		finger_width_ = finger_width;
	}
	
	/**
	 * \brief Set the hand depth of the robot hand.
	 * \param hand_depth the hand depth of the robot hand (usually the finger length)
	*/
	void setHandDepth(double hand_depth)
	{
		hand_depth_ = hand_depth;
	}
	
	/**
	 * \brief Set the maximum aperture of the robot hand.
	 * \param hand_outer_diameter the maximum aperture of the robot hand
	*/
	void setHandOuterDiameter(double hand_outer_diameter)
	{
		hand_outer_diameter_ = hand_outer_diameter;
	}
	
	/**
	 * \brief Set the initial "bite" of the robot hand (usually the minimum object "height").
	 * \param init_bite the initial "bite" of the robot hand (@see FingerHand)
	*/
	void setInitBite(double init_bite)
	{
		init_bite_ = init_bite;
	}
	
	/**
	 * \brief Set the height of the robot hand (antopodial grasps).
	 * \param hand_height the height of the robot hand, the hand extends plus/minus this value along the hand axis
	*/
	void setHandHeight(double hand_height)
	{
		hand_height_ = hand_height;
	}
		
	/**
	 * \brief Set the publisher for Rviz visualization, the lifetime of visual markers, and the frame associated with 
	 * the grasps.
	 * \param node the ROS node
	 * \param marker_lifetime the lifetime of each visual marker
	 * \param frame the frame to which the grasps belong
	*/ 
	void createVisualsPub(ros::NodeHandle& node, double marker_lifetime, const std::string& frame)
	{
		plot_.createVisualPublishers(node, marker_lifetime);
		visuals_frame_ = frame;
	}

	void setAngleThresholdBetweenNormals(double angleThresholdBetweenNormals) {
		angle_threshold_between_normals_ = angleThresholdBetweenNormals;
	}

	void setAngleTollerance(double angleTollerance) {
		angle_tollerance_ = angleTollerance;
	}

	void setCurvatureThreshold(double curvatureThreshold) {
		curvature_threshold_ = curvatureThreshold;
	}

	void setMaxDetectedRadius(double maxDetectedRadius) {
		max_detected_radius_ = maxDetectedRadius;
	}

	void setMaxNumberOfIterationsCircleDetection(
			int maxNumberOfIterationsCircleDetection) {
		max_number_of_iterations_circle_detection_ =
				maxNumberOfIterationsCircleDetection;
	}

	void setMinDetectedRadius(double minDetectedRadius) {
		min_detected_radius_ = minDetectedRadius;
	}

	void setMinimumSizeOfClusterAllowed(int minimumSizeOfClusterAllowed) {
		minimum_size_of_cluster_allowed_ = minimumSizeOfClusterAllowed;
	}

	void setNormalDistanceWeight(double normalDistanceWeight) {
		normal_distance_weight_ = normalDistanceWeight;
	}

	void setNormalRadiusSearch(double normalRadiusSearch) {
		normal_radius_search_ = normalRadiusSearch;
	}

	void setNumOfKdTreeNeighbors(int numOfKdTreeNeighbors) {
		num_of_kdTree_neighbors_ = numOfKdTreeNeighbors;
	}

	void setSegmentationDistanceThreshold(
			double segmentationDistanceThreshold) {
		segmentation_distance_threshold_ = segmentationDistanceThreshold;
	}

	void setAreaConsiderationRatio(double areaConsiderationRatio) {
		area_consideration_ratio_ = areaConsiderationRatio;
	}

private:

	/**
	 * \brief Comparator for checking uniqueness of two 3D-vectors. 
	*/
	struct UniqueVectorComparator
	{
		/**
		 * \brief Compares two 3D-vectors for uniqueness.
		 * \param a the first 3D-vector
		 * \param b the second 3D-vector
		 * \return true if they have no equal elements, false otherwise
		*/
		bool operator ()(const Eigen::Vector3i& a, const Eigen::Vector3i& b)
		{
			for (int i = 0; i < a.size(); i++)
			{
				if (a(i) != b(i))
				{
					return a(i) < b(i);
				}
			}

			return false;
		}
	};
	
	/**
	 * \brief Voxelize the point cloud and keep track of the camera source for each voxel.
	 * \param[in] cloud_in the point cloud to be voxelized
	 * \param[in] pts_cam_source_in the camera source for each point in the point cloud
	 * \param[out] cloud_out the voxelized point cloud
	 * \param[out] pts_cam_source_out the camera source for each point in the voxelized cloud
	 * \param[in] cell_size the size of each voxel
	*/
	void voxelizeCloud(const PointCloud::Ptr& cloud_in, const Eigen::VectorXi& pts_cam_source_in,
			PointCloud::Ptr& cloud_out, Eigen::VectorXi& pts_cam_source_out, double cell_size);

	/**
	 * \brief Voxelize the point cloud and keep track of the camera source for each voxel.
	 * \param[in] cloud_in the point cloud to be voxelized
	 * \param[in] pts_cam_source_in the camera source for each point in the point cloud
	 * \param[out] cloud_out the voxelized point cloud
	 * \param[out] pts_cam_source_out the camera source for each point in the voxelized cloud
	 * \param[in] cell_size the size of each voxel
	*/
	void voxelizeCloud(const PointCloudRGB::Ptr& cloud_in, const Eigen::VectorXi& pts_cam_source_in,
			PointCloudRGB::Ptr& cloud_out, Eigen::VectorXi& pts_cam_source_out, double cell_size);

     /**
	 * \brief Preproces the point cloud and keep track of the camera sources. The function removes NAN points,
	 * filters the points out side the work space and if the uses_clustering flag is true removes the largest plane in the Point cloud
	 * \param[in] cloud_in the point cloud to preprocessed
	 * \param[out] a copy of the pointcloud after processing used for plotting
	 * \param[in] the size of the point cloud on the left side
	 * \param[out] uses_clustering the flag determining if the largest plane should be removed
	 * \retrun the Preprocessed point cloud
	 */

	PointCloudRGB::Ptr PointCloudPreProcessing(
			const PointCloudRGB::Ptr& cloud_in, PointCloudRGB::Ptr& cloud_plot,
			int size_left, bool uses_clustering);

	/**
	 * \brief Filter out points in the point cloud that lie outside the workspace.
	 *  and keep track of the camera source for each point that is not filtered out.
	 * \param[in] cloud_in the point cloud to be filtered
	 * \param[in] pts_cam_source_in the camera source for each point in the point cloud
	 * \param[out] cloud_out the filtered point cloud
	 * \param[out] pts_cam_source_out the camera source for each point in the filtered cloud
	*/
	void filterWorkspace(const PointCloud::Ptr& cloud_in, const Eigen::VectorXi& pts_cam_source_in,
			PointCloud::Ptr& cloud_out, Eigen::VectorXi& pts_cam_source_out);
	
	/**
	 * \brief Filter out points in the point cloud that lie outside the workspace.
	 *  and keep track of the camera source for each point that is not filtered out.
	 * \param[in] cloud_in the point cloud to be filtered
	 * \param[in] pts_cam_source_in the camera source for each point in the point cloud
	 * \param[out] cloud_out the filtered point cloud
	 * \param[out] pts_cam_source_out the camera source for each point in the filtered cloud
	*/
	void filterWorkspace(const PointCloudRGB::Ptr& cloud_in, const Eigen::VectorXi& pts_cam_source_in,
				PointCloudRGB::Ptr& cloud_out, Eigen::VectorXi& pts_cam_source_out);

	/**
	 * \brief Filter out grasp hypotheses that are close to the workspace boundaries.
	 * \param hand_list the list of grasp hypotheses to be filtered
	 * \return the list of grasp hypotheses that are not filtered out
	*/
	std::vector<GraspHypothesis> filterHands(const std::vector<GraspHypothesis>& hand_list);

	/**
	 * \brief Round a 3D-vector down to the closest, smaller integers.
	 * \param a the 3D-vector to be rounded down
	 * \return the rounded down 3D-vector
	*/ 
	Eigen::Vector3i floorVector(const Eigen::Vector3d& a);

//	void addCylindersToPlot(
//			const boost::shared_ptr<pcl::visualization::PCLVisualizer>& plot_obj,
//			std::vector<pcl::PointIndices>& circle_inliners,
//			std::vector<pcl::ModelCoefficients>& circle_coefficients_of_all_clusters);

	//common used variables for both suction and finger grippers
	Plot plot_; ///< the plot object
	Eigen::VectorXd workspace_; ///< the robot's workspace dimensions
	Eigen::Matrix3Xd cloud_normals_; ///< the normals for each point in the point cloud
	PointCloud::Ptr cloud_; ///< the input point cloud
	PointCloudRGB::Ptr cloud_rgb_; ///< the input point cloud
	int num_threads_; ///< the number of CPU threads used in the search

	//parameters used only for finger grippers
	Eigen::Matrix4d cam_tf_left_, cam_tf_right_; ///< the camera poses
	double nn_radius_taubin_; ///< the radius of the neighborhood search used in the grasp hypothesis search
	double nn_radius_hands_; ///< the radius of the neighborhood search used in the quadric fit
	double finger_width_; ///< width of the fingers
	double hand_outer_diameter_; ///< maximum hand aperture
	double hand_depth_; ///< hand depth (finger length)
	double hand_height_; ///< hand height
	double init_bite_; ///< initial bite
	int num_samples_; ///< the number of samples used in the search

	//parameters used only for suction grippers
	/**  Segmentation(Region Growing) */
	double normal_radius_search_;///< Sphere radius that is to be used for determining the nearest neighbors used for the normal detection. Segmentation(Region Growing)
	int num_of_kdTree_neighbors_;///< number of neighbors to be sampled from the KdTree. Segmentation(Region Growing)
	double angle_threshold_between_normals_;///< [degrees] if the angles between normals is greater than this value the area will not be added to the cluster. Segmentation(Region Growing)
	double curvature_threshold_;///< the curvature diff threshold of the region such that the surfaces are considered to be a region. Segmentation(Region Growing)
	int minimum_size_of_cluster_allowed_;///< the minimum number of points to be detected to consider a region a group. Segmentation(Region Growing)

	/**  Parameters for segmentation (circle detection) or hand geometry parameters */
	double min_detected_radius_;///<[meters]. Segmentation (circle detection)
	double max_detected_radius_;///<[meters]. Segmentation (circle detection)
	double suction_gripper_radius_; ///<[meters] // used for area filtration using sectors. Segmentation (circle detection)
	double angle_tollerance_; ///< [degrees] the tolerance for the circle detection from the given axis. Segmentation (circle detection)
	double normal_distance_weight_;///< range [0-1]max_number_of_iterations_circle_detection_
	int max_number_of_iterations_circle_detection_;///< The maximum number of iterations run by the RANSAC algorithm to find a cricle. Segmentation (circle detection)
	double segmentation_distance_threshold_;///<[meters] distance threshold from circle model, points further than the threshold are not considered the smaller the value the more exact it is to a circle and less of a hollow cylinder. Segmentation (circle detection)
    double area_consideration_ratio_;

	//plotting parameters
	bool plots_camera_sources_; ///< whether the camera source is plotted for each point in the point cloud
	bool filters_boundaries_; ///< whether grasp hypotheses close to the workspace boundaries are filtered out
	int plotting_mode_; ///< what plotting mode is used
	std::string visuals_frame_; ///< visualization frame for Rviz
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_comb_;///< plotting object used for multiple cloud visualization. Upper left is the raw cloud, upper right is the preprocessed cloud, lower left is the region segmented cloud, lower right cloud is where the grasps are plotted as cylinders
	std::vector<int> viewer_point_indicies_;///< the indices used to define the view ports for the viewer
	bool first_plot_;///< a flag to check if this is the first plot execution
	boost::thread ploter_thread_;///< a thread used for the plotting object
//	boost::mutex updateModelMutex;///< mutex used to protect the plotting thread from racing
	boost::recursive_mutex updateModelMutex;///< mutex used to protect the plotting thread from racing with processing thread

	/** constants for plotting modes */
	static const int NO_PLOTTING = 0; ///< no plotting
	static const int PCL_PLOTTING = 1; ///< plotting in PCL
	static const int RVIZ_PLOTTING = 2; ///< plotting in Rviz
};






#endif
