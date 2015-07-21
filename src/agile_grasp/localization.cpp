#include <agile_grasp/localization.h>
#include <omp.h>

std::vector<GraspHypothesis> Localization::localizeSuctionGrasps(const PointCloud::Ptr& cloud_in, int size_left,
	const std::vector<int>& indices, bool calculates_antipodal, bool uses_clustering, bool plot_on_flag)
{
//	PointCloud::Ptr cloud_plot2(new PointCloud);
//	PointCloud::Ptr cloud2;
//	cloud2 = PointCloudPreProcessing(cloud_in,cloud_plot2,size_left,uses_clustering);
	std::cout << "Generating camera sources for ";
	std::vector<GraspHypothesis> suction_grasp_hyp_list;
	double t0 = omp_get_wtime();
	if (size_left == 0 || cloud_in->size() == 0)
	{
		std::cout << "Input cloud is empty!\n";
		std::cout << size_left << std::endl;
		suction_grasp_hyp_list.resize(0);
		return suction_grasp_hyp_list;
	}

	// set camera source for all points (0 = left, 1 = right)
	std::cout << "Generating camera sources for " << cloud_in->size() << " points ...\n";
	Eigen::VectorXi pts_cam_source(cloud_in->size());
	if (size_left == cloud_in->size())
		pts_cam_source << Eigen::VectorXi::Zero(size_left);
	else
		pts_cam_source << Eigen::VectorXi::Zero(size_left), Eigen::VectorXi::Ones(cloud_in->size() - size_left);

	// remove NAN points from the cloud
	std::vector<int> nan_indices;
	pcl::removeNaNFromPointCloud(*cloud_in, *cloud_in, nan_indices);

	// reduce point cloud to workspace
	std::cout << "Filtering workspace ...\n cloud size before filtering: "<< cloud_in->size()<<"\n";
	PointCloud::Ptr cloud(new PointCloud);
	filterWorkspace(cloud_in, pts_cam_source, cloud, pts_cam_source);
	std::cout << "cloud size after filtering:" << cloud->size() << " points left\n";

	// store complete cloud for later plotting
	PointCloud::Ptr cloud_plot(new PointCloud);
	*cloud_plot = *cloud;
	*cloud_ = *cloud;

	// voxelize point cloud
	std::cout << "Voxelizing point cloud\n";
	double t1_voxels = omp_get_wtime();
	voxelizeCloud(cloud, pts_cam_source, cloud, pts_cam_source, 0.003);
	double t2_voxels = omp_get_wtime() - t1_voxels;
	std::cout << " Created " << cloud->points.size() << " voxels in " << t2_voxels << " sec\n";

	// plot camera source for each point in the cloud
	if (plots_camera_sources_)
		plot_.plotCameraSource(pts_cam_source, cloud);
	// visualization
	bool print_pcl_before_and_after_filtering = false;
	if(print_pcl_before_and_after_filtering)
	{
		std::cout << "cloud size before filtering:";
		plot_.plotCloud(cloud_in);
		std::cout << "cloud size after filtering:";
		plot_.plotCloud(cloud);
	}


	if (uses_clustering)
	{
    std::cout << "Finding point cloud clusters ... \n";

		// Create the segmentation object for the planar model and set all the parameters
		pcl::SACSegmentation<pcl::PointXYZ> seg;
		pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
		pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>());
		seg.setOptimizeCoefficients(true);
		seg.setModelType(pcl::SACMODEL_PLANE);
		seg.setMethodType(pcl::SAC_RANSAC);
		seg.setMaxIterations(100);
		seg.setDistanceThreshold(0.01);

		// Segment the largest planar component from the remaining cloud
		seg.setInputCloud(cloud);
		seg.segment(*inliers, *coefficients);
		if (inliers->indices.size() == 0)
		{
			std::cout << " Could not estimate a planar model for the given dataset.\n check localization.cpp" << std::endl;
			suction_grasp_hyp_list.resize(0);
			return suction_grasp_hyp_list;
		}

    std::cout << " PointCloud representing the planar component: " << inliers->indices.size()
      << " data points." << std::endl;

		// Extract the nonplanar inliers
		pcl::ExtractIndices<pcl::PointXYZ> extract;
		extract.setInputCloud(cloud);
		extract.setIndices(inliers);
		extract.setNegative(true);// this extracts all the points that are not in the inliners set
		std::vector<int> indices_cluster;
		extract.filter(indices_cluster);
		PointCloud::Ptr cloud_cluster(new PointCloud);
		cloud_cluster->points.resize(indices_cluster.size());
		Eigen::VectorXi cluster_cam_source(indices_cluster.size());
		for (int i = 0; i < indices_cluster.size(); i++)
		{
			cloud_cluster->points[i] = cloud->points[indices_cluster[i]];
			cluster_cam_source[i] = pts_cam_source[indices_cluster[i]];
		}
		// visualization
		bool print_pcl_before_and_after_plane_extraction = false;
		if(print_pcl_before_and_after_plane_extraction)
			{
				std::cout << "cloud size before plane removal:\n";
				plot_.plotCloud(cloud);
				std::cout << "cloud size after plane removal:\n";
				plot_.plotCloud(cloud_cluster);
				std::cout << "plane removed:\n";
				extract.setNegative(false);
				extract.filter(*cloud_plane);
				plot_.plotCloud(cloud_plane);
			}

		// overwrite old cloud with the cloud without the plane
//		cloud->~PointCloud();
//		PointCloud::Ptr cloud = cloud_cluster;
		cloud = cloud_cluster;
		*cloud_plot = *cloud;
		std::cout << " PointCloud representing the non-planar component: " << cloud->points.size()
      << " data points." << std::endl;
	}
	double t_end_preprocessing = omp_get_wtime();
	std::cout << "preprocessing done in " << t_end_preprocessing - t0 << " sec\n";

			double t_seg_start = omp_get_wtime();
			pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;// create normal estimator
			pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ()); // create KD tree
			pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>); // output data set for normals
			std::vector <pcl::PointIndices> clusters; // create the output data set for clusters

			// parameters for segmentation
			double normal_radius_search = normal_radius_search_;
			int num_of_kdTree_neighbours = num_of_kdTree_neighbors_;
			double angle_threshold_between_normals = angle_threshold_between_normals_;// in degrees
			double curvature_threshold = curvature_threshold_;
			int minimum_size_of_cluster_allowed = minimum_size_of_cluster_allowed_;
			// extraction of normals for the entier cloud
			normal_estimator.setInputCloud (cloud);
			normal_estimator.setSearchMethod (tree);
			normal_estimator.setRadiusSearch (normal_radius_search);
			normal_estimator.compute (*cloud_normals);

			pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg; // create the RegionGrowing object
			reg.setMinClusterSize (minimum_size_of_cluster_allowed);
			double t_end_normal_calc = omp_get_wtime();
			std::cout << "Normal calculation done in " <<  t_end_normal_calc -t_seg_start << " sec\n";
			//reg.setMaxClusterSize (1000000);
			reg.setSearchMethod (tree);
			reg.setNumberOfNeighbours (num_of_kdTree_neighbours); // number of neighbors to sample from the KD tree
			reg.setInputCloud (cloud);
			reg.setInputNormals (cloud_normals);
			reg.setSmoothnessThreshold (angle_threshold_between_normals / 180.0 * M_PI); // the angle between normals threshold
			reg.setCurvatureThreshold (curvature_threshold);// the curvature threshold
			reg.extract (clusters);
			double t_end_cirlce_clustering = omp_get_wtime();
		   std::cout << "Clustering done in " <<  t_end_cirlce_clustering -t_end_normal_calc << " sec\n";


			//circle extraction
			PointCloud::Ptr cluster_cloud_complete(new PointCloud);// the segmented cloud after removing the points determined to not fall in a region
			PointCloud::Ptr cluster_cloud_circle(new PointCloud);// contains the cloud where all detected circles are projected
			pcl::PointCloud<pcl::Normal>::Ptr cluster_normals (new pcl::PointCloud<pcl::Normal>); // output data set for normals
			pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg_cluster; // segmentor
			pcl::ExtractIndices<pcl::PointXYZ> extract_sub_cloud; // used to extract the sub cloud for each cluster
			pcl::ExtractIndices<pcl::Normal> extract_sub_normals;// used to extract the normals for each cluster
			pcl::ModelCoefficients::Ptr coefficients_circle (new pcl::ModelCoefficients); // the coefficients of the circle detected
			pcl::PointIndices::Ptr inlineers_circle (new pcl::PointIndices);// the inliners of the point indices
			std::vector<pcl::PointIndices> circle_inliners_of_all_clusters;// a vector of PointIndices where each is entry corresponds to a the indices of a detected circle in a cluster
			circle_inliners_of_all_clusters.resize(clusters.size());
			std::vector<pcl::ModelCoefficients> circle_coefficients_of_all_clusters;// a vector of PointIndices where each is entry corresponds to a the indices of a detected circle in a cluster
			circle_coefficients_of_all_clusters.resize(clusters.size());
			double min_detected_radius = min_detected_radius_;
			double max_detected_radius = max_detected_radius_;
			double angle_tollerance = angle_tollerance_; // [degrees] the tollerance for the circle detection from the given axis
			double normal_distance_weight = normal_distance_weight_;
			int max_number_of_iterations_circle_detection = max_number_of_iterations_circle_detection_;
			double segmentation_distance_threshold = segmentation_distance_threshold_;
			double area_consideration_ratio = area_consideration_ratio_;
			//const boost::shared_ptr aPtr(clusters);


			int number_of_clusters_without_circle =0;
			for (int i =0; i < clusters.size(); i++) {
				pcl::PointIndices::Ptr aPtr(new pcl::PointIndices(clusters[i]));
				// extraction of a sub-cloud i.e. one region from the regions
//				extract_sub_cloud.setIndices(aPtr);
//				extract_sub_cloud.setInputCloud(cloud);
//				extract_sub_cloud.setNegative(false);
//				extract_sub_cloud.filter(*cluster_cloud);
//				std::cout<< "\n the number of points in the current cluster is : "<< aPtr->indices.size();
//				plot_.plotCloud(cluster_cloud);

				// extraction of the normals of a sub-cloud i.e. one region from the regions
				extract_sub_normals.setInputCloud(cloud_normals);
				extract_sub_normals.setIndices(aPtr);
				extract_sub_normals.setNegative(false);
				extract_sub_normals.filter (*cluster_normals);

				// segmentation object for circle segmentation and set all the parameters
				seg_cluster.setOptimizeCoefficients (true);
				seg_cluster.setModelType (pcl::SACMODEL_CIRCLE3D);
				seg_cluster.setMethodType (pcl::SAC_RANSAC);
				seg_cluster.setNormalDistanceWeight (normal_distance_weight);
				seg_cluster.setMaxIterations (max_number_of_iterations_circle_detection);
				seg_cluster.setDistanceThreshold (segmentation_distance_threshold);// distance from model to be considerd an inliner
				seg_cluster.setRadiusLimits (min_detected_radius, max_detected_radius);
				Eigen::Vector3f Axis = cluster_normals->points[0].getNormalVector3fMap();
				seg_cluster.setAxis(Axis);
				seg_cluster.setEpsAngle(angle_tollerance/180*M_PI);
				seg_cluster.setInputCloud (cloud);
				seg_cluster.setInputNormals (cloud_normals);
				seg_cluster.setIndices(aPtr);
				seg_cluster.segment (*inlineers_circle, *coefficients_circle);

				// fitting a hull to the circle inliners by projecting the circle inliners onto a plane that is defined by the circle coefficients
				PointCloud::Ptr temp_cloud(new PointCloud);// the segmented cloud after removing the points determined to not fall in a region
				// defining the plane coefficients
				pcl::ModelCoefficients::Ptr circle_to_cylinder_test (new pcl::ModelCoefficients);
					circle_to_cylinder_test->values.push_back(coefficients_circle->values[4]);//a
					circle_to_cylinder_test->values.push_back(coefficients_circle->values[5]);//b
					circle_to_cylinder_test->values.push_back(coefficients_circle->values[6]);//c
					circle_to_cylinder_test->values.push_back(sqrt((pow(coefficients_circle->values[0],2)+pow(coefficients_circle->values[1],2)+pow(coefficients_circle->values[2],2))));//d
				// project the inliers to the plane
				pcl::ProjectInliers<pcl::PointXYZ> proj;
				proj.setModelType (pcl::SACMODEL_PLANE);// points are projected on a plane because circle 3d Does not have a model in project inliers
				proj.setInputCloud (cloud);
				proj.setModelCoefficients (circle_to_cylinder_test);
				proj.setIndices (inlineers_circle);
				proj.filter (*temp_cloud);
//				plot_.plotCloud(temp_cloud, "inliers projected onto plane");
				// fit the hull
				std::vector<pcl::Vertices> hull_vertices;
				pcl::ConvexHull<pcl::PointXYZ> hull3;
				hull3.setInputCloud (temp_cloud);
//				hull3.setIndices (inlineers_circle);
				hull3.setComputeAreaVolume(true);
				hull3.reconstruct(*temp_cloud,hull_vertices);//hull.getTotalArea(), hull.getDimension()
				// check area condition and accordingly add or ignore the detected circle.
				if(hull3.getTotalArea()>= M_PI*pow(min_detected_radius,2)*area_consideration_ratio)
				{// area large enough consider
					circle_inliners_of_all_clusters[i] = *inlineers_circle; // append the inliner indecies of the detected circle
					circle_coefficients_of_all_clusters[i] = *coefficients_circle;
				}

				// post processing for proper result production
				if (circle_inliners_of_all_clusters[i].indices.size() != 0) {
					// checking the angle between the zaxis and the vector computed for the approach
					Eigen::Vector3d Zaxis(0, 0, 1); // assuming a positive z axis from the camera to the scene
					Eigen::Vector3d cylinder_vector(
							// represents the approach vector
							circle_coefficients_of_all_clusters[i].values[4],
							circle_coefficients_of_all_clusters[i].values[5],
							circle_coefficients_of_all_clusters[i].values[6]);
					double dotproduct = Zaxis.dot(cylinder_vector); // it is the cos of the angel since both vectors are normalized there is no need to divide
					if (dotproduct > 0) { // the vectors are between |-90 and 90 degrees| from each other then we reverse the direction
						circle_coefficients_of_all_clusters[i].values[4] =
								-circle_coefficients_of_all_clusters[i].values[4];
						circle_coefficients_of_all_clusters[i].values[5] =
								-circle_coefficients_of_all_clusters[i].values[5];
						circle_coefficients_of_all_clusters[i].values[6] =
								-circle_coefficients_of_all_clusters[i].values[6];
					}

					// creating a coordinate system using the vector obtained
					Eigen::Vector3d direction_vectror_z(
												circle_coefficients_of_all_clusters[i].values[4],
												circle_coefficients_of_all_clusters[i].values[5],
												circle_coefficients_of_all_clusters[i].values[6]);
					Eigen::Vector3d direction_vectror_y(// this arrangement offers one of the perpendicular vectors on z (the approach vector)
												-circle_coefficients_of_all_clusters[i].values[5],
												circle_coefficients_of_all_clusters[i].values[4],
												0);
					Eigen::Vector3d direction_vectror_x = direction_vectror_y.cross(direction_vectror_z);

					Eigen::Vector3d surface_point(
							circle_coefficients_of_all_clusters[i].values[0],
							circle_coefficients_of_all_clusters[i].values[1],
							circle_coefficients_of_all_clusters[i].values[2]);
					GraspHypothesis grasp(surface_point, direction_vectror_z,direction_vectror_y,direction_vectror_x);
					suction_grasp_hyp_list.push_back(grasp);
				} else {
					number_of_clusters_without_circle++;
				}
			//std::cout<< "\n The circle has the following coefficients: \n" <<coefficients_circle->values;

			}
			double t_end_cirlce_extraction = omp_get_wtime();
			std::cout << "circle extraction done in " <<  t_end_cirlce_extraction -t_end_cirlce_clustering << " sec\n";

			 // Stats
			 double t_seg_end = omp_get_wtime();
			 std::cout<<"************Stats************\n";
			 std::cout<< "The number of clusters in the current cluster is : "<< clusters.size()<<"\n";
			 std::cout<< "The number of clusters which have no circle is: "<< number_of_clusters_without_circle<<"\n";
			 std::cout << "complete algorithm without  done in " << t_seg_end - t0 << " sec\n";

			 /**
			  * The coming section is used for plotting the results
			  */

			 bool print_string_data = false;
			 if(plot_on_flag){
				 double t_start_plot= omp_get_wtime();
			 // Preprocessing for plotting
			 pcl::PointIndices concatinated_circle_inliners;
			 pcl::PointIndices concatinated_clusters;
		for (int i; i < circle_inliners_of_all_clusters.size(); i++) {
			// Concatenate all the indices of circles
			if (circle_inliners_of_all_clusters[i].indices.size()!= 0){
			concatinated_circle_inliners.indices.insert(
					concatinated_circle_inliners.indices.end(),
					circle_inliners_of_all_clusters[i].indices.begin(),
					circle_inliners_of_all_clusters[i].indices.end());
			}
			// Concatenate all the viewer_comb of the clusters removing the non clusterd points
			concatinated_clusters.indices.insert(
					concatinated_clusters.indices.end(),
					clusters[i].indices.begin(), clusters[i].indices.end());
		}
		// extract the points corresponding to the indices
		pcl::PointIndices::Ptr concatinated_circle_inliners_pointer (new pcl::PointIndices(concatinated_circle_inliners));
			 extract_sub_cloud.setInputCloud(cloud);
			 extract_sub_cloud.setIndices(concatinated_circle_inliners_pointer);
			 extract_sub_cloud.setNegative(false);
			 extract_sub_cloud.filter(*cluster_cloud_circle);
		// extract the points corresponding to the indices
		pcl::PointIndices::Ptr concatinated_clusters_pointer (new pcl::PointIndices(concatinated_clusters));
			 extract_sub_cloud.setIndices(concatinated_clusters_pointer);
			 extract_sub_cloud.filter(*cluster_cloud_complete);


			 // plots
			 if(first_plot_){
			 viewer_comb_->removeAllShapes();
//			 boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_comb (new pcl::visualization::PCLVisualizer ("Algorithim output"));
			 // common comands
			 viewer_comb_->initCameraParameters ();
			 viewer_comb_->addCoordinateSystem (0.1);
			 // part 1(pre processing)
//			 int viewPortID_1(0);
//			 viewer_comb_->createViewPort (0.0, 0.5, 0.5, 1.0, viewer_point_indicies_[0]);
			 pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> RawCloudColor (cloud, 150, 150, 150);
			 viewer_comb_->addPointCloud<pcl::PointXYZ> (cloud_in,RawCloudColor,"RawCloud",viewer_point_indicies_[0]);// cloud_
			 viewer_comb_->addText ("Raw input point cloud", 10, 10, "v1 text", viewer_point_indicies_[0]);

			 // part 2(Post processing)
//			 int viewPortID_2(0);
//			 viewer_comb_->createViewPort (0.5, 0.5, 1.0, 1.0, viewer_point_indicies_[1]);
			 pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> PreprocessedCloudColor (cloud, 150, 150, 150);
			 viewer_comb_->addPointCloud<pcl::PointXYZ> (cloud,PreprocessedCloudColor,"PreprocessedCloud",viewer_point_indicies_[1]);
			 viewer_comb_->addText ("Post pre-processing", 10, 10, "v2 text", viewer_point_indicies_[1]);

			 // part 3 (segmentation)
//			 int viewPortID_3(0);
//			 viewer_comb_->createViewPort (0.0, 0.0, 0.5, 0.5, viewer_point_indicies_[2]);
			 viewer_comb_->addText ("Segmentation", 10, 10, "v3 text", viewer_point_indicies_[2]);
			 pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> SegmentationColor(reg.getColoredCloud ());
			 viewer_comb_->addPointCloud<pcl::PointXYZRGB> (reg.getColoredCloud (), SegmentationColor, "segmented_cloud",viewer_point_indicies_[2]);
			 viewer_comb_->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (reg.getColoredCloud (), cloud_normals, 50, 0.05, "normals_v2",viewer_point_indicies_[2]);
			 viewer_comb_->addPointCloud<pcl::PointXYZ> (cluster_cloud_circle,"CircleCloud_v1",viewer_point_indicies_[2]);

			 // part 4 (Grasps)
			 int viewPortID_4(0);
//			 viewer_comb_->createViewPort (0.5, 0.0, 1.0, 0.5, viewer_point_indicies_[3]);
			 viewer_comb_->addText ("GraspDetection", 10, 10, "v4 text", viewer_point_indicies_[3]);
			 pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> GraspsColor (cloud, 0, 150, 200);
			 viewer_comb_->addPointCloud<pcl::PointXYZ> (cluster_cloud_complete, GraspsColor, "grasps_cloud",viewer_point_indicies_[3]);
			 viewer_comb_->addPointCloud<pcl::PointXYZ> (cluster_cloud_circle,"CircleCloud_v2",viewer_point_indicies_[3]);
//			 addCylindersToPlot(viewer_comb_,circle_inliners_of_all_clusters,circle_coefficients_of_all_clusters);
			 first_plot_ = false;
			 }
			 else
			 {
				 //Updating the PC plot here
				 viewer_comb_->updatePointCloud(cloud_in,"RawCloud");// update raw input

				 viewer_comb_->updatePointCloud(cloud,"PreprocessedCloud");// update preprocessed cloud

				 viewer_comb_->updatePointCloud(reg.getColoredCloud (), "segmented_cloud");// update segmented Poincloud
				 viewer_comb_->updatePointCloud(cluster_cloud_circle,"CircleCloud_v1");// update segmented Poincloud
				 viewer_comb_->removePointCloud("normals_v2",viewer_point_indicies_[2]);
				 viewer_comb_->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (reg.getColoredCloud (), cloud_normals, 50, 0.05, "normals_v2",viewer_point_indicies_[2]);

				 pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> GraspsColor (cloud, 0, 150, 200);
				 viewer_comb_->updatePointCloud(cluster_cloud_complete,GraspsColor,"grasps_cloud");
				 viewer_comb_->updatePointCloud(cluster_cloud_circle,"CircleCloud_v2");
			 }
			 viewer_comb_->removeAllShapes();
			 for (int i; i < circle_inliners_of_all_clusters.size(); i++) {
			 				 if (circle_inliners_of_all_clusters[i].indices.size()!= 0){
			 					 if(print_string_data){
			 					 std::cout<< "The parameters of cylinder "<<i<<" are \n";
			 					 std::cout<< "x: "<<circle_coefficients_of_all_clusters[i].values[0]<<"\n";
			 					 std::cout<< "y: "<<circle_coefficients_of_all_clusters[i].values[1]<<"\n";
			 					 std::cout<< "z: "<<circle_coefficients_of_all_clusters[i].values[2]<<"\n";
			 					 std::cout<< "radius: "<<circle_coefficients_of_all_clusters[i].values[3]<<"\n";
			 					 std::cout<< "vector x: "<<circle_coefficients_of_all_clusters[i].values[4]<<"\n";
			 					 std::cout<< "vector y: "<<circle_coefficients_of_all_clusters[i].values[5]<<"\n";
			 					 std::cout<< "vector z: "<<circle_coefficients_of_all_clusters[i].values[6]<<"\n";
			 					 std::cout<<"************************\n";
			 					 }
//			 					Eigen::Vector3d surface_grasp_point = suction_grasp_hyp_list[i].getGraspSurface();
//			 					Eigen::Vector3d end_point_xaixs = surface_grasp_point + 0.02*suction_grasp_hyp_list[i].getAxis();
//			 					Eigen::Vector3d end_point_yaixs = surface_grasp_point + 0.02*suction_grasp_hyp_list[i].getBinormal();
//			 					Eigen::Vector3d end_point_zaixs = surface_grasp_point + 0.02*suction_grasp_hyp_list[i].getApproach();
//
//			 					pcl::PointXYZ start_point;
//			 					start_point.x = surface_grasp_point[0];start_point.y = surface_grasp_point[1];start_point.z = surface_grasp_point[2];
//			 					pcl::PointXYZ end_point_x;
//			 					end_point_x.x = end_point_xaixs[0];end_point_x.y = end_point_xaixs[1];end_point_x.z = end_point_xaixs[2];
//			 					viewer_comb_->addArrow(start_point,end_point_x,1.0,1.0,1.0,true,"arrow_x"+i,viewer_point_indicies_[3]);
//			 					pcl::PointXYZ end_point_y;
//			 					end_point_y.x = end_point_yaixs[0];end_point_y.y = end_point_yaixs[1];end_point_y.z = end_point_yaixs[2];
//			 					viewer_comb_->addArrow(start_point,end_point_y,1.0,1.0,1.0,true,"arrow_y"+i,viewer_point_indicies_[3]);
//			 					pcl::PointXYZ end_point_z;
//			 					end_point_z.x = end_point_zaixs[0];end_point_z.y = end_point_zaixs[1];end_point_z.z = end_point_zaixs[2];
//			 					viewer_comb_->addArrow(start_point,end_point_z,1.0,1.0,1.0,true,"arrow_z"+i,viewer_point_indicies_[3]);
			 					 // the order of the Coefficients are different regarding the radis and direction
			 					 pcl::ModelCoefficients circle_to_cylinder;
			 					 circle_to_cylinder.values.push_back(circle_coefficients_of_all_clusters[i].values[0]);
			 					 circle_to_cylinder.values.push_back(circle_coefficients_of_all_clusters[i].values[1]);
			 					 circle_to_cylinder.values.push_back(circle_coefficients_of_all_clusters[i].values[2]);
			 					 circle_to_cylinder.values.push_back(circle_coefficients_of_all_clusters[i].values[4]*2/100);
			 					 circle_to_cylinder.values.push_back(circle_coefficients_of_all_clusters[i].values[5]*2/100);
			 					 circle_to_cylinder.values.push_back(circle_coefficients_of_all_clusters[i].values[6]*2/100);// scaling down for plotting
			 					 circle_to_cylinder.values.push_back(circle_coefficients_of_all_clusters[i].values[3]);
			 					viewer_comb_->addCylinder(circle_to_cylinder,"Circle"+i,viewer_point_indicies_[3]);
			 			 }
			 			 }
//			 viewer_comb_->spin();
			 viewer_comb_->spinOnce(100);
//			 while (!viewer_comb_->wasStopped()){
//				 viewer_comb_->spinOnce(100);
//			 			 			   }
			 double t_end_plot = omp_get_wtime();
			 std::cout << "Plotting requierd " <<  t_end_plot- t_start_plot<< " sec\n";
		}

	// draw down-sampled and workspace reduced cloud
	cloud_plot = cloud;
//
//
//  // set plotting within handle search on/off
//  bool plots_hands;
//  if (plotting_mode_ == PCL_PLOTTING)
//		plots_hands = true;
//  else
//		plots_hands = false;
//
//	// find hand configurations
//  HandSearch hand_search(finger_width_, hand_outer_diameter_, hand_depth_, hand_height_, init_bite_, num_threads_,
//		num_samples_, plots_hands);
//	suction_grasp_hyp_list = hand_search.findHands(cloud, pts_cam_source, indices, cloud_plot, calculates_antipodal, uses_clustering);
//
//	// remove hands at boundaries of workspace
//	if (filters_boundaries_)
//  {
//    std::cout << "Filtering out hands close to workspace boundaries ...\n";
//    suction_grasp_hyp_list = filterHands(suction_grasp_hyp_list);
//    std::cout << " # hands left: " << suction_grasp_hyp_list.size() << "\n";
//  }
//
//	double t2 = omp_get_wtime();
//	std::cout << "Hand localization done in " << t2 - t0 << " sec\n";
//
//	if (plotting_mode_ == PCL_PLOTTING || true)
//	{
//		plot_.plotHands(suction_grasp_hyp_list, cloud_plot, "");
//	}
//	else if (plotting_mode_ == RVIZ_PLOTTING)
//	{
//		plot_.plotGraspsRviz(suction_grasp_hyp_list, visuals_frame_);
//	}


//	plot_.plotHands(suction_grasp_hyp_list, cloud_plot, "");// the function createNormalsCloud has been modified to reverse the directions of the normlals being ploted
	return suction_grasp_hyp_list;
}


std::vector<GraspHypothesis> Localization::localizeHands(const PointCloud::Ptr& cloud_in, int size_left,
	const std::vector<int>& indices, bool calculates_antipodal, bool uses_clustering)// the original function
{		
	double t0 = omp_get_wtime();
	std::vector<GraspHypothesis> hand_list;
	
	if (size_left == 0 || cloud_in->size() == 0)
	{
		std::cout << "Input cloud is empty!\n";
		std::cout << size_left << std::endl;
		hand_list.resize(0);
		return hand_list;
	}
	
	// set camera source for all points (0 = left, 1 = right)
	std::cout << "Generating camera sources for " << cloud_in->size() << " points ...\n";
	Eigen::VectorXi pts_cam_source(cloud_in->size());
	if (size_left == cloud_in->size())
		pts_cam_source << Eigen::VectorXi::Zero(size_left);
	else
		pts_cam_source << Eigen::VectorXi::Zero(size_left), Eigen::VectorXi::Ones(cloud_in->size() - size_left);
		
	// remove NAN points from the cloud
	std::vector<int> nan_indices;
	pcl::removeNaNFromPointCloud(*cloud_in, *cloud_in, nan_indices);

	// reduce point cloud to workspace
	std::cout << "Filtering workspace ...\n cloud size before filtering: "<< cloud_in->size()<<"\n";
	PointCloud::Ptr cloud(new PointCloud);
	filterWorkspace(cloud_in, pts_cam_source, cloud, pts_cam_source);
	std::cout << "cloud size after filtering:" << cloud->size() << " points left\n";

	// store complete cloud for later plotting
	PointCloud::Ptr cloud_plot(new PointCloud);
	*cloud_plot = *cloud;
	*cloud_ = *cloud;

	// voxelize point cloud
	std::cout << "Voxelizing point cloud\n";
	double t1_voxels = omp_get_wtime();
	voxelizeCloud(cloud, pts_cam_source, cloud, pts_cam_source, 0.003);
	double t2_voxels = omp_get_wtime() - t1_voxels;
	std::cout << " Created " << cloud->points.size() << " voxels in " << t2_voxels << " sec\n";

	// plot camera source for each point in the cloud
	if (plots_camera_sources_)
		plot_.plotCameraSource(pts_cam_source, cloud);
	// visualization
	bool print_pcl_before_and_after_filtering = false;
	if(print_pcl_before_and_after_filtering)
	{
		std::cout << "cloud size before filtering:";
		plot_.plotCloud(cloud_in);
		std::cout << "cloud size after filtering:";
		plot_.plotCloud(cloud);
	}


	if (uses_clustering)
	{
    std::cout << "Finding point cloud clusters ... \n";
        
		// Create the segmentation object for the planar model and set all the parameters
		pcl::SACSegmentation<pcl::PointXYZ> seg;
		pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
		pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>());
		seg.setOptimizeCoefficients(true);
		seg.setModelType(pcl::SACMODEL_PLANE);
		seg.setMethodType(pcl::SAC_RANSAC);
		seg.setMaxIterations(100);
		seg.setDistanceThreshold(0.01);

		// Segment the largest planar component from the remaining cloud
		seg.setInputCloud(cloud);
		seg.segment(*inliers, *coefficients);
		if (inliers->indices.size() == 0)
		{
			std::cout << " Could not estimate a planar model for the given dataset.\n check localization.cpp" << std::endl;
			hand_list.resize(0);
			return hand_list;
		}
    
    std::cout << " PointCloud representing the planar component: " << inliers->indices.size()
      << " data points." << std::endl;

		// Extract the nonplanar inliers
		pcl::ExtractIndices<pcl::PointXYZ> extract;
		extract.setInputCloud(cloud);
		extract.setIndices(inliers);
		extract.setNegative(true);// this extracts all the points that are not in the inliners set
		std::vector<int> indices_cluster;
		extract.filter(indices_cluster);
		PointCloud::Ptr cloud_cluster(new PointCloud);
		cloud_cluster->points.resize(indices_cluster.size());
		Eigen::VectorXi cluster_cam_source(indices_cluster.size());
		for (int i = 0; i < indices_cluster.size(); i++)
		{
			cloud_cluster->points[i] = cloud->points[indices_cluster[i]];
			cluster_cam_source[i] = pts_cam_source[indices_cluster[i]];
		}
		// visualization
		bool print_pcl_before_and_after_plane_extraction = false;
		if(print_pcl_before_and_after_plane_extraction)
			{
				std::cout << "cloud size before plane removal:";
				plot_.plotCloud(cloud);
				std::cout << "cloud size after plane removal:";
				plot_.plotCloud(cloud_cluster);
			}

		// overwrite old cloud with the cloud without the plane
		cloud = cloud_cluster;
		*cloud_plot = *cloud;
		std::cout << " PointCloud representing the non-planar component: " << cloud->points.size()
      << " data points." << std::endl;
	}
	bool use_region_growing = false;
		if (use_region_growing)
		{
			double t_seg_start = omp_get_wtime();

			pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;// create normal estimator
			pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ()); // create KD tree
			pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>); // output data set for normals
			std::vector <pcl::PointIndices> clusters; // create the output data set for clusters

			// parameters for segmentation
			double normal_radius_search = 0.01;
			int num_of_kdTree_neighbours = 30;
			double angle_threshold_between_normals = 3.5;// in degrees
			double curvature_threshold = 1.0;
			int minimum_size_of_cluster_allowed = 300;
			// extraction of normals for the entier cloud
			normal_estimator.setInputCloud (cloud);
			normal_estimator.setSearchMethod (tree);
			normal_estimator.setRadiusSearch (normal_radius_search);
			normal_estimator.compute (*cloud_normals);

			pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg; // create the RegionGrowing object
			reg.setMinClusterSize (minimum_size_of_cluster_allowed);
			//reg.setMaxClusterSize (1000000);
			reg.setSearchMethod (tree);
			reg.setNumberOfNeighbours (num_of_kdTree_neighbours); // number of neighbors to sample from the KD tree
			reg.setInputCloud (cloud);
			reg.setInputNormals (cloud_normals);
			reg.setSmoothnessThreshold (angle_threshold_between_normals / 180.0 * M_PI); // the angle between normals threshold
			reg.setCurvatureThreshold (curvature_threshold);// the curvature threshold
			reg.extract (clusters);



			//circle extraction
//			PointCloud::Ptr cluster_cloud(new PointCloud);
			PointCloud::Ptr cluster_cloud_complete(new PointCloud);// the segmented cloud after removing the points determined to not fall in a region
			PointCloud::Ptr cluster_cloud_circle(new PointCloud);// contains the cloud where all detected circles are projected
			pcl::PointCloud<pcl::Normal>::Ptr cluster_normals (new pcl::PointCloud<pcl::Normal>); // output data set for normals
			pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg_cluster; // segmentor
			pcl::ExtractIndices<pcl::PointXYZ> extract_sub_cloud; // used to extract the sub cloud for each cluster
			pcl::ExtractIndices<pcl::Normal> extract_sub_normals;// used to extract the normals for each cluster
			pcl::ModelCoefficients::Ptr coefficients_circle (new pcl::ModelCoefficients); // the coefficients of the circle detected
			pcl::PointIndices::Ptr inlineers_circle (new pcl::PointIndices);// the inliners of the point indices
			std::vector<pcl::PointIndices> circle_inliners_of_all_clusters;// a vector of PointIndices where each is entry corresponds to a the indices of a detected circle in a cluster
			circle_inliners_of_all_clusters.resize(clusters.size());
			std::vector<pcl::ModelCoefficients> circle_coefficients_of_all_clusters;// a vector of PointIndices where each is entry corresponds to a the indices of a detected circle in a cluster
			circle_coefficients_of_all_clusters.resize(clusters.size());
			double min_detected_radius = 0.02;
			double max_detected_radius = 0.03;
			double angle_tollerance = 4.0; // [degrees] the tollerance for the circle detection from the given axis
			double normal_distance_weight = 0.1;
			int max_number_of_iterations_circle_detection = 1000;
			double segmentation_distance_threshold = 0.004;
			//const boost::shared_ptr aPtr(clusters);

			for (int i; i < clusters.size(); i++) {
				pcl::PointIndices::Ptr aPtr(new pcl::PointIndices(clusters[i]));
				// extraction of a sub-cloud i.e. one region from the regions
//				extract_sub_cloud.setIndices(aPtr);
//				extract_sub_cloud.setInputCloud(cloud);
//				extract_sub_cloud.setNegative(false);
//				extract_sub_cloud.filter(*cluster_cloud);
				//std::cout<< "\n the number of points in the current cluster is : "<< aPtr->indices.size();
				//plot_.plotCloud(cluster_cloud);

				// extraction of the normals of a sub-cloud i.e. one region from the regions
				extract_sub_normals.setInputCloud(cloud_normals);
				extract_sub_normals.setIndices(aPtr);
				extract_sub_normals.setNegative(false);
				extract_sub_normals.filter (*cluster_normals);

				// segmentation object for circle segmentation and set all the parameters
				seg_cluster.setOptimizeCoefficients (true);
				//seg_cluster.setModelType (pcl::SACMODEL_CIRCLE2D);
				seg_cluster.setModelType (pcl::SACMODEL_CIRCLE3D);
				seg_cluster.setMethodType (pcl::SAC_RANSAC);
				seg_cluster.setNormalDistanceWeight (normal_distance_weight);
				seg_cluster.setMaxIterations (max_number_of_iterations_circle_detection);
				seg_cluster.setDistanceThreshold (segmentation_distance_threshold);// distance from model to be considerd an inliner
				seg_cluster.setRadiusLimits (min_detected_radius, max_detected_radius);
				Eigen::Vector3f Axis = cluster_normals->points[0].getNormalVector3fMap();
				seg_cluster.setAxis(Axis);
				seg_cluster.setEpsAngle(angle_tollerance/180*M_PI);
				//seg_cluster.setInputCloud (cluster_cloud);
				//seg_cluster.setInputNormals (cluster_normals);
				seg_cluster.setInputCloud (cloud);
				seg_cluster.setInputNormals (cloud_normals);
				seg_cluster.setIndices(aPtr);
				seg_cluster.segment (*inlineers_circle, *coefficients_circle);
				circle_inliners_of_all_clusters[i] = *inlineers_circle; // append the inliner indecies of the detected circle
				circle_coefficients_of_all_clusters[i] = *coefficients_circle;
				//std::cout<< "\n The circle has the following coefficients: \n" <<coefficients_circle->values;

			}

			 //std::vector<int> indices_filtered;
			 //int j = 0;
			 pcl::PointIndices concatinated_circle_inliners;
			 pcl::PointIndices concatinated_clusters;
			 int number_of_clusters_without_circle =0;
		for (int i; i < circle_inliners_of_all_clusters.size(); i++) {
			// concatinate all the indecies
			if (circle_inliners_of_all_clusters[i].indices.size()!= 0){
			concatinated_circle_inliners.indices.insert(
					concatinated_circle_inliners.indices.end(),
					circle_inliners_of_all_clusters[i].indices.begin(),
					circle_inliners_of_all_clusters[i].indices.end());
			// concatinate all the indecies of the clusters
			concatinated_clusters.indices.insert(
					concatinated_clusters.indices.end(),
					clusters[i].indices.begin(), clusters[i].indices.end());
			}
			else
				number_of_clusters_without_circle++;

		}
		// extract the points corresponding to the indices
		pcl::PointIndices::Ptr concatinated_circle_inliners_pointer (new pcl::PointIndices(concatinated_circle_inliners));
			 extract_sub_cloud.setInputCloud(cloud);
			 extract_sub_cloud.setIndices(concatinated_circle_inliners_pointer);
			 extract_sub_cloud.setNegative(false);
			 extract_sub_cloud.filter(*cluster_cloud_circle);
		// extract the points corresponding to the indices
		pcl::PointIndices::Ptr concatinated_clusters_pointer (new pcl::PointIndices(concatinated_clusters));
			 extract_sub_cloud.setIndices(concatinated_clusters_pointer);
			 extract_sub_cloud.filter(*cluster_cloud_complete);

			 // Stats
			 double t_seg_end = omp_get_wtime();
			 std::cout<<"************Stats************\n";
			 std::cout<< "The number of clusters in the current cluster is : "<< clusters.size()<<"\n";
			 std::cout<< "The number of clusters which have no circle is: "<< number_of_clusters_without_circle<<"\n";
			 std::cout << "Segmentation done in " << t_seg_end - t_seg_start << " sec\n";

			 // plots

			 boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_filtered (new pcl::visualization::PCLVisualizer ("3D Viewer2"));
			 viewer_filtered->setBackgroundColor (0, 0, 0);
			 pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color (cloud, 255, 255, 0);
			 viewer_filtered->addPointCloud<pcl::PointXYZ> (cluster_cloud_complete, single_color, "cluster_cloud");
			 //viewer_filtered->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (reg.getColoredCloud (), cloud_normals, 50, 0.05, "normals");
			 viewer_filtered->addPointCloud<pcl::PointXYZ> (cluster_cloud_circle,"CircleCloud");
			 pcl::ModelCoefficients circle_to_cylinder;
			 for (int i; i < circle_inliners_of_all_clusters.size(); i++) {
				 if (circle_inliners_of_all_clusters[i].indices.size()!= 0){

					 Eigen::Vector3d cylinder_vector(
						circle_coefficients_of_all_clusters[i].values[4],
						circle_coefficients_of_all_clusters[i].values[5],
						circle_coefficients_of_all_clusters[i].values[6]);
					 Eigen::Vector3d Zaxis(0,0,1);// assuming a positive z axis from the camera to the scene
					 double dotproduct = Zaxis.dot(cylinder_vector);// it is the cos of the angel since both vectors are normalized there is no need to divide
					 if(dotproduct>0){// the vectors are between |-90 and 90 degrees| from each other then we revers the direction
						 circle_coefficients_of_all_clusters[i].values[4] = -circle_coefficients_of_all_clusters[i].values[4];
						 circle_coefficients_of_all_clusters[i].values[5] = -circle_coefficients_of_all_clusters[i].values[5];
						 circle_coefficients_of_all_clusters[i].values[6] = -circle_coefficients_of_all_clusters[i].values[6];
					 }
					 std::cout<< "The parameters of cylinder "<<i<<" are \n";
					 std::cout<< "x: "<<circle_coefficients_of_all_clusters[i].values[0]<<"\n";
					 std::cout<< "y: "<<circle_coefficients_of_all_clusters[i].values[1]<<"\n";
					 std::cout<< "z: "<<circle_coefficients_of_all_clusters[i].values[2]<<"\n";
					 std::cout<< "radius: "<<circle_coefficients_of_all_clusters[i].values[3]<<"\n";
					 std::cout<< "vector x: "<<circle_coefficients_of_all_clusters[i].values[4]<<"\n";
					 std::cout<< "vector y: "<<circle_coefficients_of_all_clusters[i].values[5]<<"\n";
					 std::cout<< "vector z: "<<circle_coefficients_of_all_clusters[i].values[6]<<"\n";
					 std::cout<<"************************\n";
					 // the order of the Coefficients are different regarding the radis and direction
					 pcl::ModelCoefficients circle_to_cylinder;
					 circle_to_cylinder.values.push_back(circle_coefficients_of_all_clusters[i].values[0]);
					 circle_to_cylinder.values.push_back(circle_coefficients_of_all_clusters[i].values[1]);
					 circle_to_cylinder.values.push_back(circle_coefficients_of_all_clusters[i].values[2]);
					 circle_to_cylinder.values.push_back(circle_coefficients_of_all_clusters[i].values[4]);
					 circle_to_cylinder.values.push_back(circle_coefficients_of_all_clusters[i].values[5]);
					 circle_to_cylinder.values.push_back(circle_coefficients_of_all_clusters[i].values[6]);
					 circle_to_cylinder.values.push_back(circle_coefficients_of_all_clusters[i].values[3]);
					 viewer_filtered->addCylinder(circle_to_cylinder,"Circle"+i);
			 }
			 }


			 boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
			 viewer->setBackgroundColor (0, 0, 0);
			 pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(reg.getColoredCloud ());
			 viewer->addPointCloud<pcl::PointXYZRGB> (reg.getColoredCloud (), rgb, "sample cloud");
			 viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (reg.getColoredCloud (), cloud_normals, 50, 0.05, "normals");
			 viewer->addPointCloud<pcl::PointXYZ> (cluster_cloud_circle,"CircleCloud");
			 while (!viewer->wasStopped() && ! viewer_filtered->wasStopped()){
				 viewer->spinOnce(100);
				 viewer_filtered->spinOnce(100);
			 			   }

		}

	// draw down-sampled and workspace reduced cloud
	cloud_plot = cloud;

  
  // set plotting within handle search on/off  
  bool plots_hands;
  if (plotting_mode_ == PCL_PLOTTING)
		plots_hands = true;
  else
		plots_hands = false;
		
	// find hand configurations
  HandSearch hand_search(finger_width_, hand_outer_diameter_, hand_depth_, hand_height_, init_bite_, num_threads_, 
		num_samples_, plots_hands);
	hand_list = hand_search.findHands(cloud, pts_cam_source, indices, cloud_plot, calculates_antipodal, uses_clustering);

	// remove hands at boundaries of workspace
	if (filters_boundaries_)
  {
    std::cout << "Filtering out hands close to workspace boundaries ...\n";
    hand_list = filterHands(hand_list);
    std::cout << " # hands left: " << hand_list.size() << "\n";
  }

	double t2 = omp_get_wtime();
	std::cout << "Hand localization done in " << t2 - t0 << " sec\n";

	if (plotting_mode_ == PCL_PLOTTING)
	{
		plot_.plotHands(hand_list, cloud_plot, "");
	}
	else if (plotting_mode_ == RVIZ_PLOTTING)
	{
		plot_.plotGraspsRviz(hand_list, visuals_frame_);
	}

	return hand_list;
}





PointCloud::Ptr Localization::PointCloudPreProcessing(const PointCloud::Ptr& cloud_in, PointCloud::Ptr& cloud_plot,	int size_left, bool uses_clustering)
{
	PointCloud::Ptr cloud(new PointCloud);
	if (size_left == 0 || cloud_in->size() == 0) {
		std::cout << "Input cloud is empty!\n";
		std::cout << size_left << std::endl;
		return cloud;
	}
	std::cout << "0 "<< "\n";
	// set camera source for all points (0 = left, 1 = right)
	std::cout << "Generating camera sources for " << cloud_in->size()
			<< " points ...\n";
	Eigen::VectorXi pts_cam_source(cloud_in->size());
	if (size_left == cloud_in->size())
		pts_cam_source << Eigen::VectorXi::Zero(size_left);
	else
		pts_cam_source << Eigen::VectorXi::Zero(size_left), Eigen::VectorXi::Ones(
				cloud_in->size() - size_left);
	std::cout << "before NAN removal" << " ...\n";
	// remove NAN points from the cloud
	std::vector<int> nan_indices;
	pcl::removeNaNFromPointCloud(*cloud_in, *cloud_in, nan_indices);

	// reduce point cloud to workspace
	std::cout << "Filtering workspace ...\n cloud size before filtering: "
			<< cloud_in->size() << "\n";
	filterWorkspace(cloud_in, pts_cam_source, cloud, pts_cam_source);
	std::cout << "cloud size after filtering:" << cloud->size()
			<< " points left\n";

	// store complete cloud for later plotting
	*cloud_plot = *cloud;
	*cloud_ = *cloud;

	// voxelize point cloud
	std::cout << "Voxelizing point cloud\n";
	double t1_voxels = omp_get_wtime();
	voxelizeCloud(cloud, pts_cam_source, cloud, pts_cam_source, 0.003);
	double t2_voxels = omp_get_wtime() - t1_voxels;
	std::cout << " Created " << cloud->points.size() << " voxels in "
			<< t2_voxels << " sec\n";

	// plot camera source for each point in the cloud
	if (plots_camera_sources_)
		plot_.plotCameraSource(pts_cam_source, cloud);
	// visualization
	bool print_pcl_before_and_after_filtering = false;
	if (print_pcl_before_and_after_filtering) {
		std::cout << "cloud size before filtering:";
		plot_.plotCloud(cloud_in);
		std::cout << "cloud size after filtering:";
		plot_.plotCloud(cloud);
	}

	if (uses_clustering) {
		std::cout << "Finding point cloud clusters ... \n";

		// Create the segmentation object for the planar model and set all the parameters
		pcl::SACSegmentation<pcl::PointXYZ> seg;
		pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
		pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(
				new pcl::PointCloud<pcl::PointXYZ>());
		seg.setOptimizeCoefficients(true);
		seg.setModelType(pcl::SACMODEL_PLANE);
		seg.setMethodType(pcl::SAC_RANSAC);
		seg.setMaxIterations(100);
		seg.setDistanceThreshold(0.01);

		// Segment the largest planar component from the remaining cloud
		seg.setInputCloud(cloud);
		seg.segment(*inliers, *coefficients);
		if (inliers->indices.size() == 0) {
			std::cout
					<< " Could not estimate a planar model for the given dataset.\n check localization.cpp"
					<< std::endl;
			return cloud_in;
		}

		std::cout << " PointCloud representing the planar component: "
				<< inliers->indices.size() << " data points." << std::endl;

		// Extract the nonplanar inliers
		pcl::ExtractIndices<pcl::PointXYZ> extract;
		extract.setInputCloud(cloud);
		extract.setIndices(inliers);
		extract.setNegative(true);// this extracts all the points that are not in the inliners set
		std::vector<int> indices_cluster;
		extract.filter(indices_cluster);
		PointCloud::Ptr cloud_cluster(new PointCloud);
		cloud_cluster->points.resize(indices_cluster.size());
		Eigen::VectorXi cluster_cam_source(indices_cluster.size());
		for (int i = 0; i < indices_cluster.size(); i++) {
			cloud_cluster->points[i] = cloud->points[indices_cluster[i]];
			cluster_cam_source[i] = pts_cam_source[indices_cluster[i]];
		}
		// visualization
		bool print_pcl_before_and_after_plane_extraction = false;
		if (print_pcl_before_and_after_plane_extraction) {
			std::cout << "cloud size before plane removal:";
			plot_.plotCloud(cloud);
			std::cout << "cloud size after plane removal:";
			plot_.plotCloud(cloud_cluster);
		}
		// overwrite old cloud with the cloud without the plane
		cloud = cloud_cluster;
		*cloud_plot = *cloud;
		std::cout << " PointCloud representing the non-planar component: "
				<< cloud->points.size() << " data points." << std::endl;
	}
	std::cout << "Finished PreProcessing" << " ...\n";
	return cloud;
}



std::vector<GraspHypothesis> Localization::predictAntipodalHands(const std::vector<GraspHypothesis>& hand_list, 
	const std::string& svm_filename)
{
	double t0 = omp_get_wtime();
	std::vector<GraspHypothesis> antipodal_hands;
	Learning learn(num_threads_);
	Eigen::Matrix<double, 3, 2> cams_mat;
	cams_mat.col(0) = cam_tf_left_.block<3, 1>(0, 3);
	cams_mat.col(1) = cam_tf_right_.block<3, 1>(0, 3);
	antipodal_hands = learn.classify(hand_list, svm_filename, cams_mat);
	std::cout << " runtime: " << omp_get_wtime() - t0 << " sec\n";
	std::cout << antipodal_hands.size() << " antipodal hand configurations found\n"; 
  if (plotting_mode_ == PCL_PLOTTING)
		plot_.plotHands(hand_list, antipodal_hands, cloud_, "Antipodal Hands");
	else if (plotting_mode_ == RVIZ_PLOTTING)
		plot_.plotGraspsRviz(antipodal_hands, visuals_frame_, true);
	return antipodal_hands;
}

std::vector<GraspHypothesis> Localization::localizeHands(const std::string& pcd_filename_left,
	const std::string& pcd_filename_right, bool calculates_antipodal, bool uses_clustering, bool use_suction)
{
	std::vector<int> indices(0);
	return localizeHands(pcd_filename_left, pcd_filename_right, indices, calculates_antipodal, uses_clustering, use_suction);
}

std::vector<GraspHypothesis> Localization::localizeHands(const std::string& pcd_filename_left,
	const std::string& pcd_filename_right, const std::vector<int>& indices, bool calculates_antipodal,
	bool uses_clustering, bool use_suction)
{
	double t0 = omp_get_wtime();

	// load point clouds
	PointCloud::Ptr cloud_left(new PointCloud);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_filename_left, *cloud_left) == -1) //* load the file
	{
		PCL_ERROR("Couldn't read pcd_filename_left file \n");
		std::vector<GraspHypothesis> hand_list(0);
		return hand_list;
	}
	if (pcd_filename_right.length() > 0)
		std::cout << "Loaded left point cloud with " << cloud_left->width * cloud_left->height << " data points.\n";
	else
		std::cout << "Loaded point cloud with " << cloud_left->width * cloud_left->height << " data points.\n";

	PointCloud::Ptr cloud_right(new PointCloud);
	if (pcd_filename_right.length() > 0)
	{
		if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_filename_right, *cloud_right) == -1) //* load the file
		{
			PCL_ERROR("Couldn't read pcd_filename_right file \n");
			std::vector<GraspHypothesis> hand_list(0);
			return hand_list;
		}
		std::cout << "Loaded right point cloud with " << cloud_right->width * cloud_right->height << " data points.\n";
		std::cout << "Loaded both clouds in " << omp_get_wtime() - t0 << " sec\n";
	}
	
	// concatenate point clouds
	std::cout << "Concatenating point clouds ...\n";
	PointCloud::Ptr cloud(new PointCloud);
	*cloud = *cloud_left + *cloud_right;
	if (use_suction)
		return localizeSuctionGrasps(cloud, cloud_left->size(), indices,
				calculates_antipodal, uses_clustering);
	else
		return localizeHands(cloud, cloud_left->size(), indices,
				calculates_antipodal, uses_clustering);
}

void Localization::filterWorkspace(const PointCloud::Ptr& cloud_in, const Eigen::VectorXi& pts_cam_source_in,
	PointCloud::Ptr& cloud_out, Eigen::VectorXi& pts_cam_source_out)
{
	std::vector<int> indices(cloud_in->points.size());
	int c = 0;
	PointCloud::Ptr cloud(new PointCloud);
//  std::cout << "workspace_: " << workspace_.transpose() << "\n";

	for (int i = 0; i < cloud_in->points.size(); i++)
	{
//    std::cout << "i: " << i << "\n";
		const pcl::PointXYZ& p = cloud_in->points[i];
		if (p.x >= workspace_(0) && p.x <= workspace_(1) && p.y >= workspace_(2) && p.y <= workspace_(3)
				&& p.z >= workspace_(4) && p.z <= workspace_(5))
		{
			cloud->points.push_back(p);
			indices[c] = i;
			c++;
		}
	}

	Eigen::VectorXi pts_cam_source(c);
	for (int i = 0; i < pts_cam_source.size(); i++)
	{
		pts_cam_source(i) = pts_cam_source_in(indices[i]);
	}

	cloud_out = cloud;
	pts_cam_source_out = pts_cam_source;
}

void Localization::voxelizeCloud(const PointCloud::Ptr& cloud_in, const Eigen::VectorXi& pts_cam_source_in,
	PointCloud::Ptr& cloud_out, Eigen::VectorXi& pts_cam_source_out, double cell_size)
{
	Eigen::Vector3d min_left, min_right;
	min_left << 10000, 10000, 10000;
	min_right << 10000, 10000, 10000;
	Eigen::Matrix3Xd pts(3, cloud_in->points.size());
	int num_left = 0;
	int num_right = 0;
	for (int i = 0; i < cloud_in->points.size(); i++)
	{
		if (pts_cam_source_in(i) == 0)
		{
			if (cloud_in->points[i].x < min_left(0))
				min_left(0) = cloud_in->points[i].x;
			if (cloud_in->points[i].y < min_left(1))
				min_left(1) = cloud_in->points[i].y;
			if (cloud_in->points[i].z < min_left(2))
				min_left(2) = cloud_in->points[i].z;
			num_left++;
		}
		else if (pts_cam_source_in(i) == 1)
		{
			if (cloud_in->points[i].x < min_right(0))
				min_right(0) = cloud_in->points[i].x;
			if (cloud_in->points[i].y < min_right(1))
				min_right(1) = cloud_in->points[i].y;
			if (cloud_in->points[i].z < min_right(2))
				min_right(2) = cloud_in->points[i].z;
			num_right++;
		}
		pts.col(i) = cloud_in->points[i].getVector3fMap().cast<double>();
	}

	// find the cell that each point falls into
	std::set<Eigen::Vector3i, Localization::UniqueVectorComparator> bins_left;
	std::set<Eigen::Vector3i, Localization::UniqueVectorComparator> bins_right;
	int prev;
	for (int i = 0; i < pts.cols(); i++)
	{
		if (pts_cam_source_in(i) == 0)
		{
			Eigen::Vector3i v = floorVector((pts.col(i) - min_left) / cell_size);
			bins_left.insert(v);
			prev = bins_left.size();
		}
		else if (pts_cam_source_in(i) == 1)
		{
			Eigen::Vector3i v = floorVector((pts.col(i) - min_right) / cell_size);
			bins_right.insert(v);
		}
	}

	// calculate the cell values
	Eigen::Matrix3Xd voxels_left(3, bins_left.size());
	Eigen::Matrix3Xd voxels_right(3, bins_right.size());
	int i = 0;
	for (std::set<Eigen::Vector3i, Localization::UniqueVectorComparator>::iterator it = bins_left.begin();
			it != bins_left.end(); it++)
	{
		voxels_left.col(i) = (*it).cast<double>();
		i++;
	}
	i = 0;
	for (std::set<Eigen::Vector3i, Localization::UniqueVectorComparator>::iterator it = bins_right.begin();
			it != bins_right.end(); it++)
	{
		voxels_right.col(i) = (*it).cast<double>();
		i++;
	}

	voxels_left.row(0) = voxels_left.row(0) * cell_size
			+ Eigen::VectorXd::Ones(voxels_left.cols()) * min_left(0);
	voxels_left.row(1) = voxels_left.row(1) * cell_size
			+ Eigen::VectorXd::Ones(voxels_left.cols()) * min_left(1);
	voxels_left.row(2) = voxels_left.row(2) * cell_size
			+ Eigen::VectorXd::Ones(voxels_left.cols()) * min_left(2);
	voxels_right.row(0) = voxels_right.row(0) * cell_size
			+ Eigen::VectorXd::Ones(voxels_right.cols()) * min_right(0);
	voxels_right.row(1) = voxels_right.row(1) * cell_size
			+ Eigen::VectorXd::Ones(voxels_right.cols()) * min_right(1);
	voxels_right.row(2) = voxels_right.row(2) * cell_size
			+ Eigen::VectorXd::Ones(voxels_right.cols()) * min_right(2);

	PointCloud::Ptr cloud(new PointCloud);
	cloud->resize(bins_left.size() + bins_right.size());
	Eigen::VectorXi pts_cam_source(bins_left.size() + bins_right.size());
	for (int i = 0; i < bins_left.size(); i++)
	{
		pcl::PointXYZ p;
		p.x = voxels_left(0, i);
		p.y = voxels_left(1, i);
		p.z = voxels_left(2, i);
		cloud->points[i] = p;
		pts_cam_source(i) = 0;
	}
	for (int i = 0; i < bins_right.size(); i++)
	{
		pcl::PointXYZ p;
		p.x = voxels_right(0, i);
		p.y = voxels_right(1, i);
		p.z = voxels_right(2, i);
		cloud->points[bins_left.size() + i] = p;
		pts_cam_source(bins_left.size() + i) = 1;
	}

	cloud_out = cloud;
	pts_cam_source_out = pts_cam_source;
}

Eigen::Vector3i Localization::floorVector(const Eigen::Vector3d& a)
{
	Eigen::Vector3i b;
	b << floor(a(0)), floor(a(1)), floor(a(2));
	return b;
}

std::vector<GraspHypothesis> Localization::filterHands(const std::vector<GraspHypothesis>& hand_list)
{
	const double MIN_DIST = 0.02;

	std::vector<GraspHypothesis> filtered_hand_list;

	for (int i = 0; i < hand_list.size(); i++)
	{
		const Eigen::Vector3d& center = hand_list[i].getGraspSurface();
		int k;
		for (k = 0; k < workspace_.size(); k++)
		{
			if (fabs((center(floor(k / 2.0)) - workspace_(k))) < MIN_DIST)
			{
				break;
			}
		}
		if (k == workspace_.size())
		{
			filtered_hand_list.push_back(hand_list[i]);
		}
	}

	return filtered_hand_list;
}

std::vector<Handle> Localization::findHandles(const std::vector<GraspHypothesis>& hand_list, int min_inliers,
	double min_length)
{
	HandleSearch handle_search;
	std::vector<Handle> handles = handle_search.findHandles(hand_list, min_inliers, min_length);
	if (plotting_mode_ == PCL_PLOTTING)
		plot_.plotHandles(handles, cloud_, "Handles");
	else if (plotting_mode_ == RVIZ_PLOTTING)
		plot_.plotHandlesRviz(handles, visuals_frame_);
	return handles;
}

//void addCylindersToPlot(
//		const boost::shared_ptr<pcl::visualization::PCLVisualizer>& plot_obj,
//		std::vector<pcl::PointIndices>& circle_inliners,
//		std::vector<pcl::ModelCoefficients>& circle_coefficients_of_all_clusters) {
//	for (int i; i < circle_inliners.size(); i++) {
//		if (circle_inliners[i].indices.size() != 0) {
//			std::cout << "The parameters of cylinder " << i << " are \n";
//			std::cout << "x: "
//					<< circle_coefficients_of_all_clusters[i].values[0] << "\n";
//			std::cout << "y: "
//					<< circle_coefficients_of_all_clusters[i].values[1] << "\n";
//			std::cout << "z: "
//					<< circle_coefficients_of_all_clusters[i].values[2] << "\n";
//			std::cout << "radius: "
//					<< circle_coefficients_of_all_clusters[i].values[3] << "\n";
//			std::cout << "vector x: "
//					<< circle_coefficients_of_all_clusters[i].values[4] << "\n";
//			std::cout << "vector y: "
//					<< circle_coefficients_of_all_clusters[i].values[5] << "\n";
//			std::cout << "vector z: "
//					<< circle_coefficients_of_all_clusters[i].values[6] << "\n";
//			std::cout << "************************\n";
//			// the order of the Coefficients are different regarding the radius and direction
//			pcl::ModelCoefficients circle_to_cylinder;
//			circle_to_cylinder.values.push_back(
//					circle_coefficients_of_all_clusters[i].values[0]);
//			circle_to_cylinder.values.push_back(
//					circle_coefficients_of_all_clusters[i].values[1]);
//			circle_to_cylinder.values.push_back(
//					circle_coefficients_of_all_clusters[i].values[2]);
//			circle_to_cylinder.values.push_back(
//					circle_coefficients_of_all_clusters[i].values[4]);
//			circle_to_cylinder.values.push_back(
//					circle_coefficients_of_all_clusters[i].values[5]);
//			circle_to_cylinder.values.push_back(
//					circle_coefficients_of_all_clusters[i].values[6]);
//			circle_to_cylinder.values.push_back(
//					circle_coefficients_of_all_clusters[i].values[3]);
//			plot_obj->addCylinder(circle_to_cylinder, "Circle" + i);
//		}
//	}
//}
