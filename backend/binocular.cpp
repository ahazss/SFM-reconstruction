#include <opencv2\xfeatures2d\nonfree.hpp>
#include <opencv2\xfeatures2d.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <iostream>
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
		if (knn_matches[r][0].distance > 0.6 * knn_matches[r][1].distance)
			continue;

		float dist = knn_matches[r][0].distance;
		if (dist < min_dist) min_dist = dist;
	}

	matches.clear();
	for (size_t r = 0; r < knn_matches.size(); ++r)
	{
		//�ų�������Ratio Test�ĵ��ƥ��������ĵ�
		if (
			knn_matches[r][0].distance > 0.6 * knn_matches[r][1].distance ||
			knn_matches[r][0].distance > 5 * max(min_dist, 10.0f)
			)
			continue;

		//����ƥ���
		matches.push_back(knn_matches[r][0]);
	}
}

void match_features(vector<Mat>& descriptor_for_all, vector<vector<DMatch>>& matches_for_all)
{
	matches_for_all.clear();
	// n��ͼ������˳���� n-1 ��ƥ��
	// 1��2ƥ�䣬2��3ƥ�䣬3��4ƥ�䣬�Դ�����
	for (int i = 0; i < descriptor_for_all.size() - 1; ++i)
	{
		cout << "Matching images " << i << " - " << i + 1 << endl;
		vector<DMatch> matches;
		match_features(descriptor_for_all[i], descriptor_for_all[i + 1], matches);
		matches_for_all.push_back(matches);
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
	if (feasible_count <= 15 || (feasible_count / p1.size()) < 0.6)
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

void reconstruct(Mat& K, Mat& R1, Mat& T1, Mat& R2, Mat& T2, vector<Point2f>& p1, vector<Point2f>& p2, vector<Point3f>& structure)
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
		structure.push_back(Point3f(col(0), col(1), col(2)));
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

void save_structure(string file_name, vector<Mat>& rotations, vector<Mat>& motions, vector<Point3f>& structure, vector<Vec3b>& colors)
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


// matchesΪ��ǰ���ڴ�������һ��DMatch�� struct_indeicesΪ��ǰͼƬ�Ŀռ��������ϵ
// structureΪ�ռ���ά�㼯�� key_pointsΪ��һ��ͼƬ�Ĺؼ��㼯��
// object_points:��һ��ͼƬ��ĳ�������Ӧ����ά������, image_points: ��һ��ͼƬ���ж�Ӧ��ά���������
void get_objpoints_and_imgpoints(
	vector<DMatch>& matches,
	vector<int>& struct_indices,
	vector<Point3f>& structure,
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
	vector<Point3f>& structure,
	vector<Point3f>& next_structure,
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
	Mat K,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<vector<Vec3b>>& colors_for_all,
	vector<vector<DMatch>>& matches_for_all,
	vector<Point3f>& structure,
	vector<vector<int>>& correspond_struct_idx,
	vector<Vec3b>& colors,
	vector<Mat>& rotations,
	vector<Mat>& motions
)
{
	//����ͷ����ͼ��֮��ı任����
	vector<Point2f> p1, p2;
	vector<Vec3b> c2;
	Mat R, T;	//��ת�����ƽ������
	Mat mask;	//mask�д�����ĵ����ƥ��㣬���������ʧ���
	get_matched_points(key_points_for_all[0], key_points_for_all[1], matches_for_all[0], p1, p2);
	get_matched_colors(colors_for_all[0], colors_for_all[1], matches_for_all[0], colors, c2);
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
		correspond_struct_idx[1][matches[i].trainIdx] = idx;
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

void main()
{
	vector<string> img_names;
	get_file_names("images", img_names);

	//��������
	Mat K(Matx33d(
		2759.48, 0, 1520.69,
		0, 2764.16, 1006.81,
		0, 0, 1));

	vector<vector<KeyPoint>> key_points_for_all;
	vector<Mat> descriptor_for_all;
	vector<vector<Vec3b>> colors_for_all;
	vector<vector<DMatch>> matches_for_all;
	//��ȡ����ͼ�������
	extract_features(img_names, key_points_for_all, descriptor_for_all, colors_for_all);
	//������ͼ�����˳�ε�����ƥ��
	match_features(descriptor_for_all, matches_for_all);

	vector<Point3f> structure;
	vector<vector<int>> correspond_struct_idx; //�����i��ͼ���е�j���������Ӧ��structure�е������
	vector<Vec3b> colors;
	vector<Mat> rotations;
	vector<Mat> motions;

	//��ʼ���ṹ����ά���ƣ�
	//��ʼ����ʼ����ͼ��ĵ���
	init_structure(
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

		//��ȡ��i��ͼ����ƥ����Ӧ����ά�㣬�Լ��ڵ�i+1��ͼ���ж�Ӧ�����ص�
		get_objpoints_and_imgpoints(
			matches_for_all[i],
			correspond_struct_idx[i],
			structure,
			key_points_for_all[i + 1],
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
		get_matched_points(key_points_for_all[i], key_points_for_all[i + 1], matches_for_all[i], p1, p2);
		get_matched_colors(colors_for_all[i], colors_for_all[i + 1], matches_for_all[i], c1, c2);

		//����֮ǰ��õ�R��T������ά�ؽ�
		vector<Point3f> next_structure;
		reconstruct(K, rotations[i], motions[i], R, T, p1, p2, next_structure);

		//���µ��ؽ������֮ǰ���ں�
		fusion_structure(
			matches_for_all[i],
			correspond_struct_idx[i],
			correspond_struct_idx[i + 1],
			structure,
			next_structure,
			colors,
			c1
		);
	}

	//����
	save_structure(".\\Viewer\\structure.yml", rotations, motions, structure, colors);
	save_structure(".\\Viewer\\structure.json", rotations, motions, structure, colors);
}*/