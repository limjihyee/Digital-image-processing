#include <iostream>
#include <iomanip>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/imgcodecs.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

//Stitcher�� �̿��ϴ� ���
void ex_panorama_simple()
{
	Mat img;
	vector<Mat> imgs;
	img = imread("left.jpg", IMREAD_COLOR);
	// ���� ũ�Ⱑ �ʹ� Ŀ�� ũ�� ������
	resize(img, img, Size(800, 800), 0, 0);
	imgs.push_back(img);
	img = imread("center.jpg", IMREAD_COLOR);
	// ���� ũ�Ⱑ �ʹ� Ŀ�� ũ�� ������
	resize(img, img, Size(800, 800), 0, 0);
	imgs.push_back(img);
	img = imread("right.jpg", IMREAD_COLOR);
	// ���� ũ�Ⱑ �ʹ� Ŀ�� ũ�� ������
	resize(img, img, Size(800, 800), 0, 0);
	imgs.push_back(img);

	Mat result;
	Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA, false);
	Stitcher::Status status = stitcher->stitch(imgs, result);
	if (status != Stitcher::OK) {
		cout << "Can't stitch images, error code = " << int(status) << endl; // ����Ǵ� ������ �ʹ� ������ error code 1�� ���
		exit(-1);
	}

	imshow("ex_panorama_simple_result", result);
	waitKey();
}

//�ܰ躰 ���� ���
Mat makePanorama(Mat img_l, Mat img_r, int thresh_dist, int min_matches){
	//<Gray scale�� ��ȯ>
	Mat img_gray_l, img_gray_r;
	cvtColor(img_l, img_gray_l, COLOR_BGR2GRAY);
	cvtColor(img_r, img_gray_r, COLOR_BGR2GRAY);

	//<Ư¡��(key point) ����>
	Ptr<SurfFeatureDetector> Detector = SURF::create(300);
	vector<KeyPoint> kpts_obj, kpts_scene;
	Detector->detect(img_gray_l, kpts_obj);
	Detector->detect(img_gray_r, kpts_scene);

	//<Ư¡�� �ð�ȭ>
	Mat img_kpts_l, img_kpts_r;
	drawKeypoints(img_gray_l, kpts_obj, img_kpts_l,
		Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_gray_r, kpts_scene, img_kpts_r,
		Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	//<�����(descriptor) ����>
	Ptr<SurfDescriptorExtractor> Extractor = 
		SURF::create(100, 4, 3, false, true);

	Mat img_des_obj, img_des_scene;
	Extractor->compute(img_gray_l, kpts_obj, img_des_obj);
	Extractor->compute(img_gray_r, kpts_scene, img_des_scene);

	//<����ڸ� �̿��� Ư¡�� ��Ī>
	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(img_des_obj, img_des_scene, matches);

	//<��Ī ��� �ð�ȭ>
	Mat img_matches;
	drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene,
		matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//<��Ī ��� ����>
	// ��Ī �Ÿ��� ���� ����� ��Ī ����� �����ϴ� ����
	// �ּ� ��Ī �Ÿ��� 3�� �Ǵ� ����� ��Ī ��� 60�̻� ���� ����
	double dist_max = matches[0].distance;
	double dist_min = matches[0].distance;
	double dist;
	for (int i = 0; i < img_des_obj.rows; i++) {
		dist = matches[i].distance;
		if (dist < dist_min)dist_min = dist;
		if (dist > dist_max)dist_max = dist;
	}
	printf("max_dist : %f \n", dist_max); // max �� ��ǻ� ���ʿ�
	printf("min_dist : %f \n", dist_min);

	vector<DMatch> matches_good;
	do {
		vector<DMatch> good_matches2;
		for (int i = 0; i < img_des_obj.rows; i++) {
			if (matches[i].distance < thresh_dist * dist_min)
				good_matches2.push_back(matches[i]);
		}
		matches_good = good_matches2;
		thresh_dist -= 1;
	} while (thresh_dist != 2 && matches_good.size() > min_matches);

	//<����� ��Ī ��� �ð�ȭ>
	Mat img_matches_good;
	drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene,
		matches_good, img_matches_good, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	// <��Ī ��� ��ǥ ����>
	vector<Point2f> obj, scene;
	for (int i = 0; i < matches_good.size(); i++) {
		obj.push_back(kpts_obj[matches_good[i].queryIdx].pt);	//img1
		scene.push_back(kpts_scene[matches_good[i].trainIdx].pt);	//img2
	}

	// < ��Ī ����κ��� homography ����� ����>
	Mat mat_homo = findHomography(scene, obj, RANSAC);
	// �̻�ġ ���Ÿ� ���� RANSAC �߰�

	// <Homograpy ����� �̿��� ���� ����ȯ>
	Mat img_result;
	warpPerspective(img_r, img_result, mat_homo, 
		Size(img_l.cols * 2, img_l.rows * 1.2), INTER_CUBIC);
	//������ �߸��� ���� �����ϱ� ���� ���������� �ο�

	// <���� ����� ����ȯ�� ���� ���� ��ü>
	Mat img_pano;
	img_pano = img_result.clone();
	Mat roi(img_pano, Rect(0, 0, img_l.cols, img_l.rows));
	img_l.copyTo(roi);

	// <���� ���� �߶󳻱�>
	int cut_x = 0, cut_y = 0;
	for (int y = 0; y < img_pano.rows; y++) {
		for (int x = 0; x < img_pano.cols; x++) {
			if (img_pano.at<Vec3b>(y, x)[0] == 0 &&
				img_pano.at<Vec3b>(y, x)[1] == 0 &&
				img_pano.at<Vec3b>(y, x)[2] == 0) {
				continue;
			}
			if (cut_x < x)	cut_x = x;
			if (cut_y < y)	cut_y = y;
		}
	}
	Mat img_pano_cut;
	img_pano_cut = img_pano(Range(0, cut_y), Range(0, cut_x));

	return img_pano_cut;
}

void ex_panorama()
{
	Mat matImage1 = imread("center.jpg", IMREAD_COLOR);	//left
	Mat matImage2 = imread("left.jpg", IMREAD_COLOR);	//center
	Mat matImage3 = imread("right.jpg", IMREAD_COLOR);	//right

	// ���� ũ�Ⱑ �ʹ� Ŀ�� ũ�� ������
	resize(matImage1, matImage1, Size(800, 800), 0, 0);
	resize(matImage2, matImage2, Size(800, 800), 0, 0);
	resize(matImage3, matImage3, Size(800, 800), 0, 0);

	if (matImage1.empty() || matImage2.empty() || matImage3.empty())exit(-1);

	Mat result;
	flip(matImage1, matImage1, 1);
	flip(matImage2, matImage2, 1);
	result = makePanorama(matImage1, matImage2, 3, 60);
	flip(result, result, 1);
	result = makePanorama(result, matImage3, 3, 60);


	imshow("ex_panorama_result2", result);
	waitKey(0);
	destroyAllWindows();
}

// ex2�� �����ϱ� ���� �Լ� (1������ makePanorma�� ����)
void find(Mat img_obj, Mat img_scene, int thresh_dist, int min_matches)
{
	//<Gray scale�� ��ȯ>
	Mat img_gray_l, img_gray_r;
	cvtColor(img_obj, img_gray_l, COLOR_BGR2GRAY);
	cvtColor(img_scene, img_gray_r, COLOR_BGR2GRAY);

	//<Ư¡��(key point) ����>
	Ptr<SiftFeatureDetector> Detector = SIFT::create(300);
	vector<KeyPoint> kpts_obj, kpts_scene;
	Detector->detect(img_gray_l, kpts_obj);
	Detector->detect(img_gray_r, kpts_scene);

	//<Ư¡�� �ð�ȭ>
	Mat img_kpts_l, img_kpts_r;
	drawKeypoints(img_gray_l, kpts_obj, img_kpts_l,
		Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_gray_r, kpts_scene, img_kpts_r,
		Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	//<�����(descriptor) ����>
	Ptr<SiftDescriptorExtractor> Extractor = SIFT::create();
	Mat img_des_obj, img_des_scene;
	Extractor->compute(img_gray_l, kpts_obj, img_des_obj);
	Extractor->compute(img_gray_r, kpts_scene, img_des_scene);

	//<����ڸ� �̿��� Ư¡�� ��Ī>
	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(img_des_obj, img_des_scene, matches);

	//<��Ī ��� �ð�ȭ>
	Mat img_matches;
	drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene,
		matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//<��Ī ��� ����>
	// ��Ī �Ÿ��� ���� ����� ��Ī ����� �����ϴ� ����
	// �ּ� ��Ī �Ÿ��� 3�� �Ǵ� ����� ��Ī ��� 60�̻� ���� ����
	double dist_max = matches[0].distance;
	double dist_min = matches[0].distance;
	double dist;
	for (int i = 0; i < img_des_obj.rows; i++) {
		dist = matches[i].distance;
		if (dist < dist_min)dist_min = dist;
		if (dist > dist_max)dist_max = dist;
	}

	printf("max_dist : %f \n", dist_max); // max�� ��ǻ� ���ʿ�
	printf("min_dist : %f \n", dist_min);

	vector<DMatch> good_matches;
	do {
		for (int i = 0; i < img_des_obj.rows; i++) {
			if (matches[i].distance < thresh_dist * dist_min)
				good_matches.push_back(matches[i]);
		}
		thresh_dist -= 1;
	} while (thresh_dist != 2 && good_matches.size() > min_matches);

	//<����� ��Ī ��� �ð�ȭ>
	Mat img_matches_good;
	drawMatches(img_obj, kpts_obj, img_scene, kpts_scene,
		good_matches, img_matches_good, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	waitKey(0);

	//<��Ī ��� ��ǥ ����>
	vector<Point2f> obj, scene;
	for (int i = 0; i < good_matches.size(); i++) {
		obj.push_back(kpts_obj[good_matches[i].queryIdx].pt);
		scene.push_back(kpts_scene[good_matches[i].trainIdx].pt);
	}

	//<��Ī ����κ��� homography ����� ����>
	Mat mat_homo = findHomography(obj, scene, RANSAC);
	//�̻�ġ ���Ÿ� ���� RANSAC �߰�

	//ã������ ��ü�� corner������ ����ִ� obj_corners ����
	vector<Point2f> obj_corners(4);
	obj_corners[0] = Point2f(0, 0);
	obj_corners[1] = Point2f((float)img_obj.cols, 0);
	obj_corners[2] = Point2f((float)img_obj.cols, (float)img_obj.rows);
	obj_corners[3] = Point2f(0, (float)img_obj.rows);

	//Homography ����� �̿��� obj_corners�� ���� ��ȯ ��� ����
	vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, mat_homo);

	//scene_corners�� ����� ��ġ��ǥ�� �̿��� scene���� ��Ī�� ��ü�� ������ �׷���
	for (int i = 0; i < 4; i++) {
		line(img_matches_good, scene_corners[i] + Point2f((float)img_obj.cols, 0), scene_corners[(i + 1) % 4] + Point2f((float)img_obj.cols, 0), Scalar(0, 255, 255), 4);
	}

	resize(img_matches_good, img_matches_good, Size(800, 800), 0, 0);
	imshow("result", img_matches_good);
	waitKey(0);
	destroyAllWindows();
}

void ex1() {
	ex_panorama_simple();
	ex_panorama();
}

void ex2() {
	Mat Book1 = imread("Book1.jpg", IMREAD_COLOR);
	Mat Book2 = imread("Book2.jpg", IMREAD_COLOR);
	Mat Book3 = imread("Book3.jpg", IMREAD_COLOR);
	Mat Scene = imread("Scene.jpg", IMREAD_COLOR);

	find(Book1, Scene, 3, 60);
	find(Book2, Scene, 3, 60);
	find(Book3, Scene, 3, 60);
}

int main() {
	ex2();

	return 0;
}