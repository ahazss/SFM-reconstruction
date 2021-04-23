#include <opencv2\xfeatures2d\nonfree.hpp>
#include <opencv2\xfeatures2d.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\opencv.hpp>
#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>
#include <direct.h>
#include "tinydir.h"

using namespace cv;
using namespace std;

/*
void extract_features(
	vector<string>& image_names,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Mat>& descriptor_for_all,
	vector<vector<Vec3b>>& colors_for_all
)
{
	key_points_for_all.clear();
	descriptor_for_all.clear();
	Mat image;

	//��ȡͼ�񣬻�ȡͼ�������㣬������
	Ptr<Feature2D> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10);
	for (auto it = image_names.begin(); it != image_names.end(); ++it)
	{
		image = imread(*it);
		if (image.empty()) continue;
		
		int scale;
		if (image.rows < 2000) {
			scale = 2000 / image.rows;
			resize(image, image, Size(image.cols * scale, image.rows * scale));
		}
		

		cout << "Extracing features: " << *it << endl;

		vector<KeyPoint> key_points;
		Mat descriptor;
		//ż�������ڴ����ʧ�ܵĴ���
		sift->detectAndCompute(image, noArray(), key_points, descriptor);

		//��������٣����ų���ͼ��
		if (key_points.size() <= 10) continue;

		key_points_for_all.push_back(key_points);
		descriptor_for_all.push_back(descriptor);

		vector<Vec3b> colors(key_points.size());
		for (int i = 0; i < key_points.size(); ++i)
		{
			Point2f& p = key_points[i].pt;
			colors[i] = image.at<Vec3b>(p.y, p.x);
		}
		colors_for_all.push_back(colors);
	}
}

void match_features(Mat& query, Mat& train, vector<DMatch>& matches)
{
	vector<vector<DMatch>> knn_matches;
	BFMatcher matcher(NORM_L2);
	matcher.knnMatch(query, train, knn_matches, 2);

	//��ȡ����Ratio Test����Сƥ��ľ���
	float min_dist = FLT_MAX;
	for (int r = 0; r < knn_matches.size(); ++r)
	{
		//Ratio Test
		if (knn_matches[r][0].distance > 0.7 * knn_matches[r][1].distance)
			continue;

		float dist = knn_matches[r][0].distance;
		if (dist < min_dist) min_dist = dist;
	}

	matches.clear();
	for (size_t r = 0; r < knn_matches.size(); ++r)
	{
		//�ų�������Ratio Test�ĵ��ƥ��������ĵ�
		if (
			knn_matches[r][0].distance > 0.7 * knn_matches[r][1].distance ||
			knn_matches[r][0].distance > 5 * max(min_dist, 10.0f)
			)
			continue;

		//����ƥ���
		matches.push_back(knn_matches[r][0]);
	}
}

void match_features(vector<int>& images_record, vector<int>& matches_record, vector<Mat>& descriptor_for_all, vector<vector<DMatch>>& matches_for_all)
{
	matches_for_all.clear();
	//��ʼ��
	images_record[0] = 1;    //������һ��ͼƬ�ѱ�ƥ��
	matches_record[0] = 0;  
	vector<int> record_for_each;
	vector<vector<DMatch>> dmatch_for_each;
	int currentImg = 0;
	// n��ͼ��ÿ��ͼ��ֱ���δƥ��ͼ����ƥ���������һ��ͼƬ����ƥ��
	int i = 0;
	for (; i < images_record.size() - 1; ++i) {
		cout << "finish matches" << i << endl;
		record_for_each.clear();
		dmatch_for_each.clear();
		//��ÿһ��ͼƬ���ֱ�ƥ��������Լ����ÿһ��ͼƬ����¼
		int j = 0;
		for (; j < images_record.size(); ++j) {
			vector<DMatch> matches;

			if (j == currentImg || images_record[j] == 1) {
				record_for_each.push_back(0);
				dmatch_for_each.push_back(matches);
				continue;
			}
			match_features(descriptor_for_all[currentImg], descriptor_for_all[j], matches);
			dmatch_for_each.push_back(matches);
			record_for_each.push_back(matches.size());
		}
		int result = -1, maxPoints = FLT_MIN;
		for (int k = 0; k < record_for_each.size(); ++k) {
			if (images_record[k] == 1) continue;
			if (record_for_each[k] > maxPoints) {
				maxPoints = record_for_each[k];
				result = k;
			}
		}
		images_record[result] = 1;
		matches_for_all.push_back(dmatch_for_each[result]);
		matches_record[i + 1] = result;
		currentImg = result;
	}
	
}

bool find_transform(Mat& K, vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T, Mat& mask)
{
	//�����ڲξ����ȡ����Ľ���͹������꣨�������꣩
	double focal_length = 0.5 * (K.at<double>(0) + K.at<double>(4));
	Point2d principle_point(K.at<double>(2), K.at<double>(5));

	//����ƥ�����ȡ��������ʹ��RANSAC����һ���ų�ʧ���
	Mat E = findEssentialMat(p1, p2, focal_length, principle_point, RANSAC, 0.999, 1.0, mask);
	if (E.empty()) return false;

	double feasible_count = countNonZero(mask);
	cout << (int)feasible_count << " -in- " << p1.size() << endl;
	//����RANSAC���ԣ�outlier��������50%ʱ������ǲ��ɿ���
	if (feasible_count <= 15 || (feasible_count / p1.size()) < 0.5)
		return false;

	//�ֽⱾ�����󣬻�ȡ��Ա任
	int pass_count = recoverPose(E, p1, p2, R, T, focal_length, principle_point, mask);

	//ͬʱλ���������ǰ���ĵ������Ҫ�㹻��
	if (((double)pass_count) / feasible_count < 0.7)
		return false;

	return true;
}

void get_matched_points(
	vector<KeyPoint>& p1,
	vector<KeyPoint>& p2,
	vector<DMatch> matches,
	vector<Point2f>& out_p1,
	vector<Point2f>& out_p2
)
{
	out_p1.clear();
	out_p2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_p1.push_back(p1[matches[i].queryIdx].pt);
		out_p2.push_back(p2[matches[i].trainIdx].pt);
	}
}

void get_matched_colors(
	vector<Vec3b>& c1,
	vector<Vec3b>& c2,
	vector<DMatch> matches,
	vector<Vec3b>& out_c1,
	vector<Vec3b>& out_c2
)
{
	out_c1.clear();
	out_c2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_c1.push_back(c1[matches[i].queryIdx]);
		out_c2.push_back(c2[matches[i].trainIdx]);
	}
}

void reconstruct(Mat& K, Mat& R1, Mat& T1, Mat& R2, Mat& T2, vector<Point2f>& p1, vector<Point2f>& p2, vector<Point3d>& structure)
{
	//���������ͶӰ����[R T]��triangulatePointsֻ֧��float��
	Mat proj1(3, 4, CV_32FC1);
	Mat proj2(3, 4, CV_32FC1);

	R1.convertTo(proj1(Range(0, 3), Range(0, 3)), CV_32FC1);
	T1.convertTo(proj1.col(3), CV_32FC1);

	R2.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
	T2.convertTo(proj2.col(3), CV_32FC1);

	Mat fK;
	K.convertTo(fK, CV_32FC1);
	proj1 = fK * proj1;
	proj2 = fK * proj2;

	//�����ؽ�
	Mat s;
	triangulatePoints(proj1, proj2, p1, p2, s);

	structure.clear();
	structure.reserve(s.cols);
	for (int i = 0; i < s.cols; ++i)
	{
		Mat_<float> col = s.col(i);
		col /= col(3);	//������꣬��Ҫ�������һ��Ԫ�ز�������������ֵ
		structure.push_back(Point3d(col(0), col(1), col(2)));
	}
}

void maskout_points(vector<Point2f>& p1, Mat& mask)
{
	vector<Point2f> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			p1.push_back(p1_copy[i]);
	}
}

void maskout_colors(vector<Vec3b>& p1, Mat& mask)
{
	vector<Vec3b> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			p1.push_back(p1_copy[i]);
	}
}

void save_structure(string file_name, vector<Mat>& rotations, vector<Mat>& motions, vector<Point3d>& structure, vector<Vec3b>& colors)
{
	int n = (int)rotations.size();

	FileStorage fs(file_name, FileStorage::WRITE);
	fs << "Camera Count" << n;
	fs << "Point Count" << (int)structure.size();

	fs << "Rotations" << "[";
	for (size_t i = 0; i < n; ++i)
	{
		fs << rotations[i];
	}
	fs << "]";

	fs << "Motions" << "[";
	for (size_t i = 0; i < n; ++i)
	{
		fs << motions[i];
	}
	fs << "]";

	fs << "Points" << "[";
	for (size_t i = 0; i < structure.size(); ++i)
	{
		fs << structure[i];
	}
	fs << "]";

	fs << "Colors" << "[";
	for (size_t i = 0; i < colors.size(); ++i)
	{
		fs << colors[i];
	}
	fs << "]";

	fs.release();
}


// matchesΪ��ǰ���ڴ������һ��DMatch�� struct_indeicesΪ��ǰͼƬ�Ŀռ��������ϵ
// structureΪ�ռ���ά�㼯�� key_pointsΪ��һ��ͼƬ�Ĺؼ��㼯��
// object_points:��һ��ͼƬ��ĳ�������Ӧ����ά������, image_points: ��һ��ͼƬ���ж�Ӧ��ά���������
void get_objpoints_and_imgpoints(
	vector<DMatch>& matches,
	vector<int>& struct_indices,
	vector<Point3d>& structure,
	vector<KeyPoint>& key_points,
	vector<Point3f>& object_points,
	vector<Point2f>& image_points)
{
	object_points.clear();
	image_points.clear();

	for (int i = 0; i < matches.size(); ++i)
	{
		int query_idx = matches[i].queryIdx;
		int train_idx = matches[i].trainIdx;
		//˵����Ӧ���Ӧ����ά���������
		int struct_idx = struct_indices[query_idx];

		if (struct_idx < 0) continue;

		object_points.push_back(structure[struct_idx]);
		image_points.push_back(key_points[train_idx].pt);
	}
}

void fusion_structure(
	vector<DMatch>& matches,
	vector<int>& struct_indices,
	vector<int>& next_struct_indices,
	vector<Point3d>& structure,
	vector<Point3d>& next_structure,
	vector<Vec3b>& colors,
	vector<Vec3b>& next_colors
)
{
	for (int i = 0; i < matches.size(); ++i)
	{
		int query_idx = matches[i].queryIdx;
		int train_idx = matches[i].trainIdx;

		int struct_idx = struct_indices[query_idx];
		if (struct_idx >= 0) //���õ��ڿռ����Ѿ����ڣ������ƥ����Ӧ�Ŀռ��Ӧ����ͬһ��������Ҫ��ͬ
		{
			next_struct_indices[train_idx] = struct_idx;
			continue;
		}

		//���õ��ڿռ��в����ڣ����õ���뵽�ṹ�У������ƥ���Ŀռ��������Ϊ�¼���ĵ������
		structure.push_back(next_structure[i]);
		colors.push_back(next_colors[i]);
		struct_indices[query_idx] = structure.size() - 1;
		next_struct_indices[train_idx] = structure.size() - 1;
	}
}

void init_structure(
	vector<int>& matches_record,
	Mat K,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<vector<Vec3b>>& colors_for_all,
	vector<vector<DMatch>>& matches_for_all,
	vector<Point3d>& structure,
	vector<vector<int>>& correspond_struct_idx,
	vector<Vec3b>& colors,
	vector<Mat>& rotations,
	vector<Mat>& motions
)
{
	//����ͷ����ͼ��֮��ı任����
	int index = matches_record[1];
	vector<Point2f> p1, p2;
	vector<Vec3b> c2;
	Mat R, T;	//��ת�����ƽ������
	Mat mask;	//mask�д�����ĵ����ƥ��㣬���������ʧ���
	get_matched_points(key_points_for_all[0], key_points_for_all[index], matches_for_all[0], p1, p2);
	get_matched_colors(colors_for_all[0], colors_for_all[index], matches_for_all[0], colors, c2);
	find_transform(K, p1, p2, R, T, mask);

	//��ͷ����ͼ�������ά�ؽ�
	maskout_points(p1, mask);
	maskout_points(p2, mask);
	maskout_colors(colors, mask);

	Mat R0 = Mat::eye(3, 3, CV_64FC1);
	Mat T0 = Mat::zeros(3, 1, CV_64FC1);
	reconstruct(K, R0, T0, R, T, p1, p2, structure);
	//����任����
	rotations = { R0, R };
	motions = { T0, T };

	//��correspond_struct_idx�Ĵ�С��ʼ��Ϊ��key_points_for_all��ȫһ��(ͼ������)
	correspond_struct_idx.clear();
	correspond_struct_idx.resize(key_points_for_all.size());
	for (int i = 0; i < key_points_for_all.size(); ++i)
	{
		// ��ÿһ��ͼƬ������Ӧ��correspond_struct_idx��С��ʼ��Ϊ��������Ŀ
		correspond_struct_idx[i].resize(key_points_for_all[i].size(), -1);
	}

	//��дͷ����ͼ��Ľṹ����(��������structure�е������Ķ�Ӧ)
	int idx = 0;
	vector<DMatch>& matches = matches_for_all[0];
	for (int i = 0; i < matches.size(); ++i)
	{
		if (mask.at<uchar>(i) == 0)
			continue;

		correspond_struct_idx[0][matches[i].queryIdx] = idx;
		correspond_struct_idx[index][matches[i].trainIdx] = idx;
		++idx;
	}
}

void get_file_names(string dir_name, vector<string>& names)
{
	names.clear();
	tinydir_dir dir;
	tinydir_open(&dir, dir_name.c_str());

	while (dir.has_next)
	{
		tinydir_file file;
		tinydir_readfile(&dir, &file);
		if (!file.is_dir)
		{
			names.push_back(file.path);
		}
		tinydir_next(&dir);
	}
	tinydir_close(&dir);
}


// ����һ�����ۺ���(����ͶӰ���)
struct ReprojectCost
{
	cv::Point2d observation;

	ReprojectCost(cv::Point2d& observation)
		: observation(observation)
	{
	}

	// �����ֱ�Ϊ�ڲΣ���Σ����е��ڿռ��е����꣬���һ�����������������ͶӰ���
	// ע��ΪʹBA����Ч������е���ת�����õ�����ת����������ת����ʹ�Ż���������
	template <typename T>
	bool operator()(const T* const intrinsic, const T* const extrinsic, const T* const pos3d, T* residuals) const
	{
		const T* r = extrinsic;
		const T* t = &extrinsic[3];

		T pos_proj[3];
		ceres::AngleAxisRotatePoint(r, pos3d, pos_proj);

		// Apply the camera translation
		pos_proj[0] += t[0];
		pos_proj[1] += t[1];
		pos_proj[2] += t[2];

		const T x = pos_proj[0] / pos_proj[2];
		const T y = pos_proj[1] / pos_proj[2];

		const T fx = intrinsic[0];
		const T fy = intrinsic[1];
		const T cx = intrinsic[2];
		const T cy = intrinsic[3];

		// Apply intrinsic
		const T u = fx * x + cx;
		const T v = fy * y + cy;

		residuals[0] = u - T(observation.x);
		residuals[1] = v - T(observation.y);

		return true;
	}
};


// ��Ceres Solver���BA
void bundle_adjustment(
	Mat& intrinsic,
	vector<Mat>& extrinsics,
	vector<vector<int>>& correspond_struct_idx,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Point3d>& structure
)
{
	ceres::Problem problem;

	// load extrinsics (rotations and motions)
	for (size_t i = 0; i < extrinsics.size(); ++i)
	{
		problem.AddParameterBlock(extrinsics[i].ptr<double>(), 6);
	}
	// fix the first camera.
	problem.SetParameterBlockConstant(extrinsics[0].ptr<double>());

	// load intrinsic
	problem.AddParameterBlock(intrinsic.ptr<double>(), 4); // fx, fy, cx, cy

	// load points
	ceres::LossFunction* loss_function = new ceres::HuberLoss(4);   // loss function make bundle adjustment robuster.
	for (size_t img_idx = 0; img_idx < correspond_struct_idx.size(); ++img_idx)
	{
		vector<int>& point3d_ids = correspond_struct_idx[img_idx];
		vector<KeyPoint>& key_points = key_points_for_all[img_idx];
		for (size_t point_idx = 0; point_idx < point3d_ids.size(); ++point_idx)
		{
			int point3d_id = point3d_ids[point_idx];
			if (point3d_id < 0)
				continue;

			Point2d observed = key_points[point_idx].pt;
			// ģ������У���һ��Ϊ���ۺ��������ͣ��ڶ���Ϊ���۵�ά�ȣ�ʣ�������ֱ�Ϊ���ۺ�����һ�ڶ����е�����������ά��
			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>(new ReprojectCost(observed));

			problem.AddResidualBlock(
				cost_function,
				loss_function,
				intrinsic.ptr<double>(),            // Intrinsic
				extrinsics[img_idx].ptr<double>(),  // View Rotation and Translation
				&(structure[point3d_id].x)          // Point in 3D space
			);
		}
	}

	// Solve BA
	ceres::Solver::Options ceres_config_options;
	ceres_config_options.minimizer_progress_to_stdout = false;
	ceres_config_options.logging_type = ceres::SILENT;
	ceres_config_options.num_threads = 1;
	ceres_config_options.preconditioner_type = ceres::JACOBI;
	ceres_config_options.linear_solver_type = ceres::SPARSE_SCHUR;
	ceres_config_options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;

	ceres::Solver::Summary summary;
	ceres::Solve(ceres_config_options, &problem, &summary);

	if (!summary.IsSolutionUsable())
	{
		std::cout << "Bundle Adjustment failed." << std::endl;
	}
	else
	{
		// Display statistics about the minimization
		std::cout << std::endl
			<< "Bundle Adjustment statistics (approximated RMSE):\n"
			<< " #views: " << extrinsics.size() << "\n"
			<< " #residuals: " << summary.num_residuals << "\n"
			<< " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
			<< " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
			<< " Time (s): " << summary.total_time_in_seconds << "\n"
			<< std::endl;
	}
}



void main()
{
	string str = _pgmptr;
	for (int i = 0; i < 18; i++) {
		str.pop_back();
	}

	vector<string> img_names;
	get_file_names(str+"\\images", img_names);

	//��������
	Mat K(Matx33d(
		2759.48, 0, 1520.69,
		0, 2764.16, 1006.81,
		0, 0, 1));

	vector<int> images_record(img_names.size(), 0);
	vector<int> matches_record(img_names.size(), 0);
	vector<vector<KeyPoint>> key_points_for_all;
	vector<Mat> descriptor_for_all;
	vector<vector<Vec3b>> colors_for_all;
	vector<vector<DMatch>> matches_for_all;
	//��ȡ����ͼ�������
	extract_features(img_names, key_points_for_all, descriptor_for_all, colors_for_all);
	//������ͼ���������ƥ��
	match_features(images_record, matches_record, descriptor_for_all, matches_for_all);

	vector<Point3d> structure;
	vector<vector<int>> correspond_struct_idx; //�����i��ͼ���е�j���������Ӧ��structure�е������
	vector<Vec3b> colors;
	vector<Mat> rotations;
	vector<Mat> motions;

	//��ʼ���ṹ����ά���ƣ�
	//��ʼ����ʼ����ͼ��ĵ���
	init_structure(
		matches_record,
		K,
		key_points_for_all,
		colors_for_all,
		matches_for_all,
		structure,
		correspond_struct_idx,
		colors,
		rotations,
		motions
	);
	

	//������ʽ�ؽ�ʣ���ͼ��
	//������0��ʼ��ѭ����i����1��ʼ
	for (int i = 1; i < matches_for_all.size(); ++i)
	{
		vector<Point3f> object_points;
		vector<Point2f> image_points;
		Mat r, R, T;
		//Mat mask;
        
		cout << "solving: " << i << endl;
		int idx1 = matches_record[i];
		int idx2 = matches_record[i + 1];

		//��ȡ��i��ͼ����ƥ����Ӧ����ά�㣬�Լ��ڵ�i+1��ͼ���ж�Ӧ�����ص�
		get_objpoints_and_imgpoints(
			matches_for_all[i],
			correspond_struct_idx[idx1],
			structure,
			key_points_for_all[idx2],
			object_points,
			image_points
		);

		//���任����
		solvePnPRansac(object_points, image_points, K, noArray(), r, T);
		//����ת����ת��Ϊ��ת����
		Rodrigues(r, R);
		//����任����
		rotations.push_back(R);
		motions.push_back(T);

		vector<Point2f> p1, p2;
		vector<Vec3b> c1, c2;
		get_matched_points(key_points_for_all[idx1], key_points_for_all[idx2], matches_for_all[i], p1, p2);
		get_matched_colors(colors_for_all[idx1], colors_for_all[idx2], matches_for_all[i], c1, c2);

		//����֮ǰ��õ�R��T������ά�ؽ�
		vector<Point3d> next_structure;
		reconstruct(K, rotations[i], motions[i], R, T, p1, p2, next_structure);

		//���µ��ؽ������֮ǰ���ں�
		fusion_structure(
			matches_for_all[i],
			correspond_struct_idx[idx1],
			correspond_struct_idx[idx2],
			structure,
			next_structure,
			colors,
			c1
		);
	}

	Mat intrinsic(Matx41d(K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2)));
	vector<Mat> extrinsics;
	for (size_t i = 0; i < rotations.size(); ++i)
	{
		Mat extrinsic(6, 1, CV_64FC1);
		Mat r;
		Rodrigues(rotations[i], r);

		r.copyTo(extrinsic.rowRange(0, 3));
		motions[i].copyTo(extrinsic.rowRange(3, 6));

		extrinsics.push_back(extrinsic);
	}

	bundle_adjustment(intrinsic, extrinsics, correspond_struct_idx, key_points_for_all, structure);


	//����
	char buffer[MAX_PATH];
	getcwd(buffer, MAX_PATH);
	string folderpath = buffer;
	folderpath = folderpath + "\\result";
	mkdir(folderpath.c_str());

	save_structure(folderpath+"\\structure.yml", rotations, motions, structure, colors);

	system("pause");
}
*/