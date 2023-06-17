#pragma once
#include <iostream>
#include <cmath>
#include <iomanip>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

void createClustersInfo(Mat imgInput, int n_cluster, vector<Scalar>& clustersCenters, vector<vector<Point>>& ptlnClusters);
double computeColorDistance(Scalar pixel, Scalar clusterPixel);
double adjustClusterCenters(Mat src_img, int n_cluster, vector<Scalar>& clustersCenters, vector<vector<Point>> ptlnClusters, double& oldCenter, double newCenter);
Mat MyKmeans(Mat src_img, int n_cluster);
void findAssociatedCluster(Mat imgInput, int n_cluster, vector<Scalar> clustersCenters, vector<vector<Point>>& ptlnClusters);
Mat applyFinalClusterTolmage(Mat src_img, int n_cluster,
	vector<vector<Point>>ptInClusters,
	vector<Scalar>clustersCenters);
Mat MyBGR2HSV(Mat src_img);
void con_Color(Mat src_img);
Mat inRange(Mat hsv_img, Scalar th1, Scalar th2);


// ���� �� ��ȭ
void CvColorModels(Mat bgr_img) {
	Mat gray_img, rgb_img, hsv_img, yuv_img, xyz_img;

	cvtColor(bgr_img, gray_img, cv::COLOR_BGR2GRAY);
	cvtColor(bgr_img, rgb_img, cv::COLOR_BGR2RGB);
	cvtColor(bgr_img, hsv_img, cv::COLOR_BGR2HSV);
	cvtColor(bgr_img, yuv_img, cv::COLOR_BGR2YCrCb);
	cvtColor(bgr_img, xyz_img, cv::COLOR_BGR2XYZ);

	Mat print_img;
	bgr_img.copyTo(print_img);
	cvtColor(gray_img, gray_img, cv::COLOR_GRAY2BGR);
	hconcat(print_img, gray_img, print_img);
	hconcat(print_img, rgb_img, print_img);
	hconcat(print_img, hsv_img, print_img);
	hconcat(print_img, yuv_img, print_img);
	hconcat(print_img, xyz_img, print_img);

	imshow("results", print_img);
	imwrite("CvColorModels.png", print_img);

	waitKey(0);
}

//���� ���� �и�
Mat GetYCvCr(Mat src_img) {
	double b, g, r, y, cb, cr;
	Mat dst_img;
	src_img.copyTo(dst_img);

	// <ȭ�� �ε���>
	for (int row = 0; row < dst_img.rows; row++) {
		for (int col = 0; col < dst_img.cols; col++) {
			// <BGR ���>
			// openCV�� Mat�� BGR�� ������ ������ ����
			b = (double)dst_img.at<Vec3b>(row, col)[0];
			g = (double)dst_img.at<Vec3b>(row, col)[1];
			r = (double)dst_img.at<Vec3b>(row, col)[2];

			// <���� ��ȯ ���>
			// ��Ȯ�� ����� ���� double �ڷ��� ���
			y = 0.2627 * r + 0.678 * g + 0.0593 * b;
			cb = -0.13963 * r - 0.36037 * g + 0.5 * b;
			cr = 0.5 * r - 0.45979 * g - 0.04021 * b;

			// <�����÷ο� ����>
			y = y > 255.0 ? 255.0 : y < 0 ? 0 : y;
			cb = cb > 255.0 ? 255.0 : cb < 0 ? 0 : cb;
			cr = cr > 255.0 ? 255.0 : cr < 0 ? 0 : cr;

			//<��ȯ�� ���� ����>
			// double �ڷ����� ���� ���� �ڷ������� ��ȯ
			dst_img.at<Vec3b>(row, col)[0] = (uchar)y;
			dst_img.at<Vec3b>(row, col)[1] = (uchar)cb;
			dst_img.at<Vec3b>(row, col)[2] = (uchar)cr;
		}
	}
	return dst_img;
}

//1������ �̿�Ǵ� opencv�� k-means clustering

Mat CvKMeans(Mat src_img, int k) {
	// <2���� ���� -> 1���� ����>
	Mat samples(src_img.rows * src_img.cols, src_img.channels(), CV_32F);
	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			if (src_img.channels() == 3) {
				for (int z = 0; z < src_img.channels(); z++) {
					samples.at<float>(y + x * src_img.rows, z) =
						(float)src_img.at<Vec3b>(y, x)[z];
				}
			}
			else {
				samples.at<float>(y + x * src_img.rows) =
					(float)src_img.at<uchar>(y, x);
			}
		}
	}

	// <opencv k-means ����
	Mat labels; // �����Ǻ� ����� ��� 1���� ����
	Mat centers; // �� ������ �߾Ӱ�(��ǥ��)
	int attempts = 5;
	kmeans(samples, k, labels,
		TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001),
		attempts, KMEANS_PP_CENTERS,
		centers);

	// <1���� ���� -> 2���� ����>
	Mat dst_img(src_img.size(), src_img.type());
	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			int cluster_idx = labels.at<int>(y + x * src_img.rows, 0);
			if (src_img.channels() == 3) {
				for (int z = 0; z < src_img.channels(); z++) {
					dst_img.at<Vec3b>(y, x)[z] =
						(uchar)centers.at<float>(cluster_idx, z);
					// �����Ǻ� ����� ���� �� ������ �߾Ӱ����� ��� ����
				}
			}
			else {
				dst_img.at<uchar>(y, x) =
					(uchar)centers.at<float>(cluster_idx, 0);
			}
		}
	}

	return dst_img;
}

// MyKMeans
Mat MyKmeans(Mat src_img, int n_cluster){
	vector<Scalar>clustersCenters;	//���� �߾Ӱ� ����
	vector<vector<Point>>ptInClusters;	//������ ��ǥ ����
	double threshold = 0.001;
	double oldCenter = INFINITY;
	double newCenter = 0;
	double diffChange = oldCenter - newCenter;	//���� ������ ��ȭ��

	//<�ʱ⼳��>
	//���� �߾Ӱ��� "�������� �Ҵ�" �� ������ ��ǥ���� ������ ���� �Ҵ�
	createClustersInfo(src_img, n_cluster, clustersCenters, ptInClusters);

	//<�߾Ӱ� ���� �� ȭ�Һ� ���� �Ǻ�>
	//�ݺ����� ������� ���� �߾Ӱ� ����
	//������ �Ӱ谪���� ���� ������ ��ȭ�� ���� ������ �ݺ�
	while (diffChange > threshold) {
		//<�ʱ�ȭ>
		newCenter = 0;
		for (int k = 0; k < n_cluster; k++) { ptInClusters[k].clear(); }

		//<������ ���� �߾Ӱ��� �������� ���� Ž��>
		findAssociatedCluster(src_img, n_cluster, clustersCenters, ptInClusters);

		//<���� �߾Ӱ� ����>
		diffChange = adjustClusterCenters(src_img, n_cluster, clustersCenters, ptInClusters, oldCenter, newCenter);
	}

	//<���� �߾Ӱ����θ� �̷���� ���� ����>
	Mat dst_img = applyFinalClusterTolmage(src_img, n_cluster, ptInClusters, clustersCenters);

	return dst_img;
}

// createClustersInfo
void createClustersInfo(Mat imgInput, int n_cluster,
	vector<Scalar>& clustersCenters, vector<vector<Point>>& ptInClusters){
	RNG random(cv::getTickCount());	//OpenCV���� ������ ���� �����ϴ� �Լ�

	for (int k = 0; k < n_cluster; k++) {	//������ ���
		//<������ ��ǥ ȹ��>
		Point centerKPoint;
		centerKPoint.x = random.uniform(0, imgInput.cols);
		centerKPoint.y = random.uniform(0, imgInput.rows);
		Scalar centerPixel = imgInput.at<Vec3b>(centerKPoint.y, centerKPoint.x);

		//<������ ��ǥ�� ȭ�Ұ����� ������ �߾Ӱ� ����>
		Scalar centerK(centerPixel.val[0], centerPixel.val[1], centerPixel.val[2]);
		clustersCenters.push_back(centerK);

		vector<Point>ptInClustersK;
		ptInClusters.push_back(ptInClustersK);
	}
}

// findAssociatedCluster
void findAssociatedCluster(Mat imgInput, int n_cluster,
	vector<Scalar>clustersCenters, vector<vector<Point>>& ptInClusters)
{
	for (int r = 0; r < imgInput.rows; r++) {
		for (int c = 0; c < imgInput.cols; c++) {
			double minDistance = INFINITY;
			int closestClusterIndex = 0;
			Scalar pixel = imgInput.at<Vec3b>(r, c);

			for (int k = 0; k < n_cluster; k++) {	//������ ���
				//<�� ���� �߾Ӱ����� ���̸� ���>
				Scalar clusterPixel = clustersCenters[k];
				double distance = computeColorDistance(pixel, clusterPixel);

				//<���̰� ���� ���� �������� ��ǥ�� ������ �Ǻ�>
				if (distance < minDistance) {
					minDistance = distance;
					closestClusterIndex = k;
				}
			}

			//<��ǥ ����>
			ptInClusters[closestClusterIndex].push_back(Point(c, r));
		}
	}
}
double computeColorDistance(Scalar pixel, Scalar clusterPixel)
{
	double diffBlue = pixel.val[0] - clusterPixel[0];
	double diffGreen = pixel.val[1] - clusterPixel[1];
	double diffRed = pixel.val[2] - clusterPixel[2];

	double distance = sqrt(pow(diffBlue, 2) + pow(diffGreen, 2) + pow(diffRed, 2));
	//Euclidian distance

	return distance;
}

//adjustClusterCenters
double adjustClusterCenters(Mat src_img, int n_cluster,
	vector<Scalar>& clustersCenters, vector<vector<Point>>ptInClusters,
	double& oldCenter, double newCenter)
{
	double diffChange;

	for (int k = 0; k < n_cluster; k++) {	//������ ���
		vector<Point>ptInCluster = ptInClusters[k];
		double newBlue = 0;
		double newGreen = 0;
		double newRed = 0;

		//<��հ� ���>
		for (int i = 0; i < ptInCluster.size(); i++) {
			Scalar pixel = src_img.at<Vec3b>(ptInCluster[i].y, ptInCluster[i].x);
			newBlue += pixel.val[0];
			newGreen += pixel.val[1];
			newRed += pixel.val[2];
		}
		newBlue /= ptInCluster.size();
		newGreen /= ptInCluster.size();
		newRed /= ptInCluster.size();

		//<����� ��հ����� ���� �߾Ӱ� ��ü>
		Scalar newPixel(newBlue, newGreen, newRed);
		newCenter += computeColorDistance(newPixel, clustersCenters[k]);
		//��� ������ ���� ��հ��� ���� ���
		clustersCenters[k] = newPixel;
	}
	newCenter /= n_cluster;
	diffChange = abs(oldCenter - newCenter);
	//��� ������ ���� ��հ� ��ȭ�� ���

	oldCenter = newCenter;

	return diffChange;
}

// applyFinalClusterTolmage
Mat applyFinalClusterTolmage(Mat src_img, int n_cluster,
	vector<vector<Point>>ptInClusters,
	vector<Scalar>clustersCenters){

	Mat dst_img(src_img.size(), src_img.type());

	for (int k = 0; k < n_cluster; k++) {	//��� ������ ���� ����
		vector<Point>ptInCluster = ptInClusters[k];	//������ ��ǥ��

		for (int j = 0; j < ptInCluster.size(); j++) {
			//������ ��ǥ ��ġ�� �ִ� ȭ�� ���� �ش� ���� �߾Ӱ����� ��ü
			dst_img.at<Vec3b>(ptInCluster[j])[0] = clustersCenters[k].val[0];
			dst_img.at<Vec3b>(ptInCluster[j])[1] = clustersCenters[k].val[1];
			dst_img.at<Vec3b>(ptInCluster[j])[2] = clustersCenters[k].val[2];
		}
	}

	return dst_img;
}

Mat MyBGR2HSV(Mat src_img)
{
	double b, g, r, h, s, v;
	Mat dst_img(src_img.size(), src_img.type());

	//<ȭ�� �ε���>
	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			//<BGR ���>
			b = (double)src_img.at<Vec3b>(y, x)[0];
			g = (double)src_img.at<Vec3b>(y, x)[1];
			r = (double)src_img.at<Vec3b>(y, x)[2];

			double min_2 = min(r, min(g, b));
			double max_2 = max(r, min(g, b));
			v = max_2;
			s = (v == 0) ? 0 : ((max_2 - min_2) / max_2)*255;

			if(max_2 == r) {	
				h = 60 * (0 + (g - b) / (max_2 - min_2));
			}
			else if (max_2 == g) {	
				h = 60 * (2 + (b - r) / (max_2 - min_2));
			}
			else if (max_2 == b) {	
				h = 60 * (4 + (r - g) / (max_2 - min_2));
			}

			if (h < 0) {
				h += 360;
			}
			h = (double)(h / 2);

			//<�����÷ο� ����>
			h = (h > 180.0) ? 180.0 : (h < 0) ? 0 : h;	
			s = (s > 255.0) ? 255.0 : (s < 0) ? 0 : s;	
			v = (v > 255.0) ? 255.0 : (v < 0) ? 0 : v;	

			dst_img.at<Vec3b>(y, x)[0] = (uchar)h;
			dst_img.at<Vec3b>(y, x)[1] = (uchar)s;
			dst_img.at<Vec3b>(y, x)[2] = (uchar)v;
		}
	}
	return dst_img;
}

void con_Color(Mat src_img)
{
	double h, s, v;
	int countR = 0, countY = 0, countG = 0;
	int countB = 0, countV = 0, countP = 0;
	int maxC;

	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {

			h = (double)src_img.at<Vec3b>(y, x)[0];
			s = (double)src_img.at<Vec3b>(y, x)[1];
			v = (double)src_img.at<Vec3b>(y, x)[2];

			if (s >= 100) {
				if ((h >= 0 && h < 15) || (h >= 170 && h <= 180)) {	//Red
					countR++;
				}
				else if (h >= 20 && h < 40) {	//Yellow
					countY++;
				}
				else if (h >= 45 && h < 80) {	//Green
					countG++;
				}
				else if (h >= 80 && h < 108) {	//blue
					countB++;
				}
				else if (h >= 108 && h < 145) {	//violet
					countV++;
				}
				else if (h >= 145 && h < 175) {	//pink
					countP++;
				}
			}
		}
	}

	// ������ ī��Ʈ �ؼ� ���� ū ���� �ִ� �� �� ���� ������ ���� ���ǿ� �ش�Ǹ� �÷��� ���ǿ� �°� �����Ѵ�.
	maxC = max({ countR, countY, countG, countB, countV, countP });
	cout << "The fruit has ";
	if (maxC == countR)
		cout << "Red" << endl;
	else if (maxC == countY)
		cout << "Yello" << endl;
	else if (maxC == countG)
		cout << "Green" << endl;
	else if (maxC == countB)
		cout << "Blue" << endl;
	else if (maxC == countV)
		cout << "Violet" << endl;
	else if (maxC == countP)
		cout << "pink" << endl;
}

Mat inRange(Mat hsv_img, Scalar th1, Scalar th2)
{
	double h, s, v;

	Mat dst_img(hsv_img.size(), hsv_img.type());

	for (int y = 0; y < dst_img.rows; y++) {
		for (int x = 0; x < dst_img.cols; x++) {
			h = (double)hsv_img.at<Vec3b>(y, x)[0];
			s = (double)hsv_img.at<Vec3b>(y, x)[1];
			v = (double)hsv_img.at<Vec3b>(y, x)[2];
			int index = (y * dst_img.cols + x) * 3;

			if ((h >= th1[0] && h <= th2[0])
				&& (s >= th1[1] && s <= th2[1])
				&& (v >= th1[2] && v <= th2[2])) {
				h = 255;
				s = 255;
				v = 255;
			}
			else {
				h = 0;
				s = 0;
				v = 0;
			}
			dst_img.at<Vec3b>(y, x)[0] = (uchar)h;
			dst_img.at<Vec3b>(y, x)[1] = (uchar)s;
			dst_img.at<Vec3b>(y, x)[2] = (uchar)v;
		}
	}

	return dst_img;
}