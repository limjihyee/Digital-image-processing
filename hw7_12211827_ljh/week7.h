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


// 색상 모델 변화
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

//색차 정보 분리
Mat GetYCvCr(Mat src_img) {
	double b, g, r, y, cb, cr;
	Mat dst_img;
	src_img.copyTo(dst_img);

	// <화소 인덱싱>
	for (int row = 0; row < dst_img.rows; row++) {
		for (int col = 0; col < dst_img.cols; col++) {
			// <BGR 취득>
			// openCV의 Mat은 BGR의 순서를 가짐에 유의
			b = (double)dst_img.at<Vec3b>(row, col)[0];
			g = (double)dst_img.at<Vec3b>(row, col)[1];
			r = (double)dst_img.at<Vec3b>(row, col)[2];

			// <색상 변환 계산>
			// 정확한 계산을 위해 double 자료형 사용
			y = 0.2627 * r + 0.678 * g + 0.0593 * b;
			cb = -0.13963 * r - 0.36037 * g + 0.5 * b;
			cr = 0.5 * r - 0.45979 * g - 0.04021 * b;

			// <오버플로우 방지>
			y = y > 255.0 ? 255.0 : y < 0 ? 0 : y;
			cb = cb > 255.0 ? 255.0 : cb < 0 ? 0 : cb;
			cr = cr > 255.0 ? 255.0 : cr < 0 ? 0 : cr;

			//<변환된 색상 대입>
			// double 자료형의 값을 본래 자료형으로 변환
			dst_img.at<Vec3b>(row, col)[0] = (uchar)y;
			dst_img.at<Vec3b>(row, col)[1] = (uchar)cb;
			dst_img.at<Vec3b>(row, col)[2] = (uchar)cr;
		}
	}
	return dst_img;
}

//1번에서 이용되는 opencv의 k-means clustering

Mat CvKMeans(Mat src_img, int k) {
	// <2차원 영상 -> 1차원 벡터>
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

	// <opencv k-means 수행
	Mat labels; // 군집판별 결과가 담길 1차원 벡터
	Mat centers; // 각 군집의 중앙값(대표값)
	int attempts = 5;
	kmeans(samples, k, labels,
		TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001),
		attempts, KMEANS_PP_CENTERS,
		centers);

	// <1차원 벡터 -> 2차원 영상>
	Mat dst_img(src_img.size(), src_img.type());
	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			int cluster_idx = labels.at<int>(y + x * src_img.rows, 0);
			if (src_img.channels() == 3) {
				for (int z = 0; z < src_img.channels(); z++) {
					dst_img.at<Vec3b>(y, x)[z] =
						(uchar)centers.at<float>(cluster_idx, z);
					// 군집판별 결과에 따라 각 군집의 중앙값으로 결과 생성
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
	vector<Scalar>clustersCenters;	//군집 중앙값 벡터
	vector<vector<Point>>ptInClusters;	//군집별 좌표 벡터
	double threshold = 0.001;
	double oldCenter = INFINITY;
	double newCenter = 0;
	double diffChange = oldCenter - newCenter;	//군집 조정의 변화량

	//<초기설정>
	//군집 중앙값을 "무작위로 할당" 및 군집별 좌표값을 저장할 벡터 할당
	createClustersInfo(src_img, n_cluster, clustersCenters, ptInClusters);

	//<중앙값 조정 및 화소별 군집 판별>
	//반복적인 방법으로 군집 중앙값 조정
	//설정한 임계값보다 군집 조정의 변화가 작을 때까지 반복
	while (diffChange > threshold) {
		//<초기화>
		newCenter = 0;
		for (int k = 0; k < n_cluster; k++) { ptInClusters[k].clear(); }

		//<현재의 군집 중앙값을 기준으로 군집 탐색>
		findAssociatedCluster(src_img, n_cluster, clustersCenters, ptInClusters);

		//<군집 중앙값 조절>
		diffChange = adjustClusterCenters(src_img, n_cluster, clustersCenters, ptInClusters, oldCenter, newCenter);
	}

	//<군집 중앙값으로만 이루어진 영상 생성>
	Mat dst_img = applyFinalClusterTolmage(src_img, n_cluster, ptInClusters, clustersCenters);

	return dst_img;
}

// createClustersInfo
void createClustersInfo(Mat imgInput, int n_cluster,
	vector<Scalar>& clustersCenters, vector<vector<Point>>& ptInClusters){
	RNG random(cv::getTickCount());	//OpenCV에서 무작위 값을 설정하는 함수

	for (int k = 0; k < n_cluster; k++) {	//군집별 계산
		//<무작위 좌표 획득>
		Point centerKPoint;
		centerKPoint.x = random.uniform(0, imgInput.cols);
		centerKPoint.y = random.uniform(0, imgInput.rows);
		Scalar centerPixel = imgInput.at<Vec3b>(centerKPoint.y, centerKPoint.x);

		//<무작위 좌표의 화소값으로 군집별 중앙값 설정>
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

			for (int k = 0; k < n_cluster; k++) {	//군집별 계산
				//<각 군집 중앙값과의 차이를 계산>
				Scalar clusterPixel = clustersCenters[k];
				double distance = computeColorDistance(pixel, clusterPixel);

				//<차이가 가장 적은 군집으로 좌표의 군집을 판별>
				if (distance < minDistance) {
					minDistance = distance;
					closestClusterIndex = k;
				}
			}

			//<좌표 저장>
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

	for (int k = 0; k < n_cluster; k++) {	//군집별 계산
		vector<Point>ptInCluster = ptInClusters[k];
		double newBlue = 0;
		double newGreen = 0;
		double newRed = 0;

		//<평균값 계산>
		for (int i = 0; i < ptInCluster.size(); i++) {
			Scalar pixel = src_img.at<Vec3b>(ptInCluster[i].y, ptInCluster[i].x);
			newBlue += pixel.val[0];
			newGreen += pixel.val[1];
			newRed += pixel.val[2];
		}
		newBlue /= ptInCluster.size();
		newGreen /= ptInCluster.size();
		newRed /= ptInCluster.size();

		//<계산한 평균값으로 군집 중앙값 대체>
		Scalar newPixel(newBlue, newGreen, newRed);
		newCenter += computeColorDistance(newPixel, clustersCenters[k]);
		//모든 군집에 대한 평균값도 같이 계산
		clustersCenters[k] = newPixel;
	}
	newCenter /= n_cluster;
	diffChange = abs(oldCenter - newCenter);
	//모든 군집에 대한 평균값 변화량 계산

	oldCenter = newCenter;

	return diffChange;
}

// applyFinalClusterTolmage
Mat applyFinalClusterTolmage(Mat src_img, int n_cluster,
	vector<vector<Point>>ptInClusters,
	vector<Scalar>clustersCenters){

	Mat dst_img(src_img.size(), src_img.type());

	for (int k = 0; k < n_cluster; k++) {	//모든 군집에 대해 수행
		vector<Point>ptInCluster = ptInClusters[k];	//군집별 좌표들

		for (int j = 0; j < ptInCluster.size(); j++) {
			//군집별 좌표 위치에 있는 화소 값을 해당 군집 중앙값으로 대체
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

	//<화소 인덱싱>
	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			//<BGR 취득>
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

			//<오버플로우 방지>
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

	// 각각을 카운트 해서 가장 큰 값이 있는 곳 그 곳이 다음과 같은 조건에 해당되면 컬러를 조건에 맞게 지정한다.
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