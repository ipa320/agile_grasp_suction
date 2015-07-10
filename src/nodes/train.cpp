#include <agile_grasp/learning.h>
#include <agile_grasp/localization.h>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
	if (argc > 3)
	{
		// get list of PCD-files for training
		int num_files = atoi(argv[1]);
    std::string pcd_dir = argv[2];
    std::vector<std::string> files;
    if (num_files == 0)
    {    
      std::string file_name = pcd_dir + "files.txt";
      std::ifstream file(file_name.c_str());
      std::string str;      
      while (std::getline(file, str))
      {
        files.push_back(pcd_dir + str);
        std::cout << files.at(files.size()-1) << "\n";
      }
    }
    else
    {
      for (int i=0; i < num_files; i++)
        files.push_back(pcd_dir + boost::lexical_cast<std::string>(i));
    }
    
    // get list of workspace dimensions if available
		std::string file_name = pcd_dir + "workspace.txt";
		std::cout << file_name;
    boost::filesystem::path file_test(file_name);
    Eigen::MatrixXd workspace_mat(files.size(), 6);
    if( !boost::filesystem::exists(file_test) )
    {
      std::cout << "No workspace.txt file found in pcd directory\n";
      std::cout << " Using standard workspace limits\n";
      Eigen::VectorXd ws(6);
      ws << 0.65, 0.9, -0.1, 0.1, -0.2, 1.0;
      for (int i=0; i < files.size(); i++)
        workspace_mat.row(i) = ws;
    }
    else
    {
      std::ifstream file_ws(file_name.c_str());
      std::string str;
      int t = 0;
      while (std::getline(file_ws, str))
      {
        for (int i = 0; i < 6; ++i)
        {
          int idx = str.find(" ");
          workspace_mat(t,i) = atof(str.substr(0, idx).c_str());
          str = str.substr(idx + 1);
        }
        t++;
      }
		}
    std::cout << "workspace_mat:\n" << workspace_mat << "\n";
    
    std::string svm_file_name = argv[3];
    
    bool plots_hands = false;
    if (argc > 4)
      plots_hands = atoi(argv[4]);

		int num_samples = 1000;
		if (argc > 5)
      num_samples = atoi(argv[5]);
		
		int num_threads = 4;
		if (argc > 6)
      num_threads = atoi(argv[6]);

		// camera poses for 2-camera Baxter setup
		Eigen::Matrix4d base_tf, sqrt_tf;

		base_tf << 	0, 0.445417, 0.895323, 0.21, 
								1, 0, 0, -0.02, 
								0, 0.895323, -0.445417, 0.24, 
								0, 0, 0, 1;

		sqrt_tf << 	0.9366, -0.0162, 0.3500, -0.2863, 
								0.0151, 0.9999, 0.0058, 0.0058, 
								-0.3501, -0.0002, 0.9367, 0.0554, 
								0, 0, 0, 1;

		// set-up parameters for the hand search (fingers)
		Localization loc(num_threads, false, plots_hands);
		loc.setCameraTransforms(base_tf * sqrt_tf.inverse(), base_tf * sqrt_tf);
		loc.setNumSamples(num_samples);
		loc.setNeighborhoodRadiusTaubin(0.08);
		loc.setNeighborhoodRadiusHands(0.08);
		loc.setFingerWidth(0.01);
		loc.setHandOuterDiameter(0.09);
		loc.setHandDepth(0.06);
		loc.setInitBite(0.015);
		loc.setHandHeight(0.02);

		/**  Segmentation(Region Growing) */
		  loc.setNormalRadiusSearch(0.01);
		  loc.setNumOfKdTreeNeighbors(30);
		  loc.setAngleThresholdBetweenNormals(3.5);
		  loc.setCurvatureThreshold(1.0);
		  loc.setMinimumSizeOfClusterAllowed(300);
		  /**  Parameters for segmentation (circle detection) or hand geometry parameters */
		  loc.setMinDetectedRadius(0.02);
		  loc.setMaxDetectedRadius(0.03);
		  loc.setAngleTollerance(4.0);
		  loc.setNormalDistanceWeight(0.1);
		  loc.setMaxNumberOfIterationsCircleDetection(1000);
		  loc.setSegmentationDistanceThreshold(0.004);

		// set-up parameters for the hand search (suction)

		// collect training data for the SVM from each PCD file
		std::cout << "Acquiring training data ...\n";
		std::vector<GraspHypothesis> hand_list;
    std::vector<int> hand_list_sizes(files.size());
    std::cout <<"loading the following pcd images for training: \n";
		for (int i = 0; i < files.size(); i++)
		{
			std::cout << " Creating training data from file " << files[i] << " ...\n";
			boost::filesystem::path pcd_file_left(files[i]+"l_reg.pcd");
			boost::filesystem::path pcd_file_right(files[i]+"r_reg.pcd");
			std::string file_left;
			std::string file_right;
			if (boost::filesystem::exists(pcd_file_left) && boost::filesystem::exists(pcd_file_right))
			{
				file_left = files[i] + "l_reg.pcd";
				file_right = files[i] + "r_reg.pcd";
				std::cout << file_left << "\n";
				std::cout << file_right << "\n";
			}
			else
			{
				std:cout << "only the left pdc was found ... \n";
				file_left = files[i] + ".pcd";
				file_right = "";
				std::cout << file_left << "\n";
			}

			loc.setWorkspace(workspace_mat.row(i));
			bool use_suction = true;
			std::vector<GraspHypothesis> hands = loc.localizeHands(file_left, file_right, true, true,use_suction);// bool calculates_antipodal, bool uses_clustering
			hand_list.insert(hand_list.end(), hands.begin(), hands.end());
      hand_list_sizes[i] = hand_list.size();
      std::cout << i << ") # hands: " << hands.size() << std::endl;
		}
		
		// use the collected data for training the SVM
		std::cout << "Training the SVM ...\n";
		Learning learn;
		Eigen::Matrix<double,3,2> cam_pos;
		cam_pos.col(0) = loc.getCameraTransform(true).block<3,1>(0,3);
		cam_pos.col(1) = loc.getCameraTransform(false).block<3,1>(0,3);
    int max_positives = 20;
    // learn.trainBalanced(hand_list, hand_list_sizes, svm_file_name, cam_pos, max_positives);
//    std::cout <<"HI	";
    learn.train(hand_list, hand_list_sizes, svm_file_name, cam_pos, max_positives);
		// learn.train(hand_list, svm_file_name, cam_pos, false);

		return 0;
	}

	std::cout << "No PCD filenames given!\n";
	std::cout << "Usage: train_svm num_files pcd_directory svm_filename [plots_hands] [num_samples] [num_threads]\n";
	std::cout << "Train an SVM to localize grasps in point clouds.\n\n";
	std::cout << "The standard way is to create a directory that contains *.pcd files for " <<
		"training. Each file needs to be called obji.pcd where i goes from 0 to <num_files>. " <<
		"<pcd_directory> is the location and the root name of the files, for example, " <<
		"/home/userA/data/obj.\n";
	std::cout << "Alternatively, if <num_files> is set to 0, there needs to be a files.txt " << 
		"within pcd_directory that lists all filenames that should be used for training, and " <<
		"a workspace.txt that lists the workspace dimensions for each file.\n";
	
	return (-1);
}

