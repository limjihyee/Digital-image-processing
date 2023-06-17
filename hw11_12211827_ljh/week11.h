#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
// 4, 5 line에서 추가한 라이브러리는 corner detection을 하기 위한 라이브러리
#include "opencv2/features2d.hpp"
#include <string.h>

using namespace std;
using namespace cv;

void cvHarrisCorner() {
	// harris corner etecion으로 corner 또한 여러 scene variation에 안정적인 특징점이 gradient 변화 기반으로 탐색하기 위해
	Mat img = imread("ship.png");
	if (img.empty()) {
		cout << "Empty image!\n";
		exit(-1);
	}

	resize(img, img, Size(500, 500), 0, 0, INTER_CUBIC);

	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	// < Do Harris corner  detection>
	Mat harr;
	cornerHarris(gray, harr, 2, 3, 0.05, BORDER_DEFAULT);
	normalize(harr, harr, 0, .55, NORM_MINMAX, CV_32FC1, Mat());

	// <Get abs for Harris visualization>
	Mat harr_abs;
	convertScaleAbs(harr, harr_abs);

	//<Print corners>
	int thresh = 125;
	Mat result = img.clone();
	for (int y = 0; y < harr.rows; y += 1) {
		for (int x = 0; x < harr.cols; x += 1) {
			if ((int)harr.at<float>(y, x) > thresh)
				circle(result, Point(x, y), 7, Scalar(255, 0, 255), 0, 4, 0);
		}
	}

	imshow("Source image", img);
	imshow("Harris image", harr_abs);
	imshow("Target image", result);
	waitKey(0);
	destroyAllWindows();
}

void cvFeatureSIFT(Mat img)
{
	resize(img, img, Size(512, 512), 0, 0);

	Mat gray;
	Mat result;
	cvtColor(img, gray, CV_BGR2GRAY);

	Ptr<SiftFeatureDetector> detector = SiftFeatureDetector::create();
	vector<KeyPoint> keypoints;
	detector->detect(gray, keypoints);

	drawKeypoints(img, keypoints, result);

	imwrite("result.jpg", result);
	imshow("result", result);
	waitKey(0);
	destroyAllWindows();
}

void cvBlobDetection() {
	// 라플라시안 가우시안의 형태에 따라 scale의 blob을 탐색한다. (시그마의 값이 커질 수록 큰 blob이 탐색된다.
		Mat img = imread("circle.jpg", IMREAD_COLOR);

		//<Set paras>
		SimpleBlobDetector::Params params;
		params.minThreshold = 10;
		params.maxThreshold = 300;
		params.filterByArea = true;
		params.minArea = 10;
		params.filterByCircularity = true;
		params.minCircularity = 0.895;
		params.filterByConvexity = true;
		params.minConvexity = 0.9;
		params.filterByInertia = true;
		params.minInertiaRatio = 0.01;

		//<Set Blob detector>
		Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

		//<Detect Blobs>
		std::vector<KeyPoint> keypoints;
		detector->detect(img, keypoints);

		//<Draw Blobs>
		Mat result;
		drawKeypoints(img, keypoints, result, Scalar(255, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

		imshow("keypoints", result);
		waitKey(0);
		destroyAllWindows();
	}

Mat warpPers(Mat img)
{
	resize(img, img, Size(512, 512), 0, 0);
	Point2f src_p[4], dst_p[4];

	src_p[0] = Point2f(0, 0);
	src_p[1] = Point2f(512, 0);
	src_p[2] = Point2f(0, 512);
	src_p[3] = Point2f(512, 512);

	dst_p[0] = Point2f(0, 0);
	dst_p[1] = Point2f(512, 0);
	dst_p[2] = Point2f(0, 512);
	dst_p[3] = Point2f(412, 412);

	Mat pers_mat = getPerspectiveTransform(src_p, dst_p);
	warpPerspective(img, img, pers_mat, Size(512, 512));
	imshow("투시변환", img);

	img.convertTo(img, -1, 1, 120);

	imshow("밝기까지", img);
	waitKey(0);
	destroyAllWindows();

	return img;
}