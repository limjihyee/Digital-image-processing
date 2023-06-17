#include <iostream>
#include <cmath>
#include <iomanip>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <string.h>

using namespace std;
using namespace cv;

void cvFlip() {
	Mat src = imread("Lenna.png", 1);
	Mat dst_x, dst_y, dst_xy;

	flip(src, dst_x, 0);
	flip(src, dst_y, 1);
	flip(src, dst_xy, -1);

	imwrite("nonflip.jpg", src);
	imwrite("xflip.jpg", dst_x);
	imwrite("yflip.jpg", dst_y);
	imwrite("xyflip.jpg", dst_xy);

	imshow("nonflip", src);
	imshow("xflip", dst_x);
	imshow("yflip", dst_y);
	imshow("xyflip", dst_xy);
	waitKey(0);

	destroyAllWindows();
}

void cvRotation() {
	Mat src = imread("Lenna.png", 1);
	Mat dst, matrix;

	Point center = Point(src.cols / 2, src.rows / 2);
	matrix = getRotationMatrix2D(center, 45.0, 1.0);
	warpAffine(src, dst, matrix, src.size());

	imwrite("nonrot.jpg", src);
	imwrite("rot.jpg", dst);

	imshow("nonrot", src);
	imshow("rot", dst);
	waitKey(0);

	destroyAllWindows();
}

Mat myRotation(Point center, float angle, float scale) {

	cout << "여기서는 cvRotation과 동일하게 45도 회전과 scale 값은 1.0으로 한다." << endl;

	double alpha = cos(angle * CV_PI / 180) * scale;
	double beta = sin(angle * CV_PI / 180) * scale;

	Mat dst = (Mat_<double>(2, 3) <<
		alpha, beta, (1 - alpha) * center.x - beta * center.y, -beta, alpha, beta * center.x + (1 - alpha) * center.y);

	return dst;
}

void myRotation_ex() {
	Mat src = imread("Lenna.png", 1);
	Mat dst, matrix;

	Point center = Point(src.cols / 2, src.rows / 2);

	matrix = myRotation(center, 45, 1);
	warpAffine(src, dst, matrix, src.size());

	imshow("Rotation 직접 구현", dst);
	waitKey(0);
	destroyAllWindows();
}

void cvAffine() {
	Mat src = imread("Lenna.png", 1);
	Mat dst, matrix;

	Point2f srcTri[3];
	srcTri[0] = Point2f(0.f, 0.f);
	srcTri[1] = Point2f(src.cols - 1.f, 0.f);
	srcTri[2] = Point2f(0.f, src.rows - 1.f);

	Point2f dstTri[3];
	dstTri[0] = Point2f(0.f, src.rows * 0.33f);
	dstTri[1] = Point2f(src.cols * 0.85f, src.rows * 0.25f);
	dstTri[2] = Point2f(src.cols * 0.15f, src.rows * 0.7f);

	matrix = getAffineTransform(srcTri, dstTri);
	warpAffine(src, dst, matrix, src.size());

	imwrite("nonaff.jpg", src);
	imwrite("aff.jpg", dst);

	imshow("nonaff", src);
	imshow("aff", dst);
	waitKey(0);

	destroyAllWindows();
}

void cvPerspective() {
	Mat src = imread("Lenna.png", 1);
	Mat dst, matrix;

	Point2f srcQuad[4];
	srcQuad[0] = Point2f(0.f, 0.f);
	srcQuad[1] = Point2f(src.cols - 1.f, 0.f);
	srcQuad[2] = Point2f(0.f, src.rows - 1.f);
	srcQuad[3] = Point2f(src.cols - 1.f, src.rows - 1.f);

	Point2f dstQuad[4];
	dstQuad[0] = Point2f(0.f, src.rows * 0.33f);
	dstQuad[1] = Point2f(src.cols * 0.85f, src.rows * 0.25f);
	dstQuad[2] = Point2f(src.cols * 0.15f, src.rows * 0.7f);
	dstQuad[3] = Point2f(src.cols * 0.85f, src.rows * 0.7f);

	matrix = getPerspectiveTransform(srcQuad, dstQuad);
	warpPerspective(src, dst, matrix, src.size());

	imwrite("nonper.jpg", src);
	imwrite("per.jpg", dst);

	imshow("nonper", src);
	imshow("per", dst);
	waitKey(0);

	destroyAllWindows();
}

Mat myTransMat() {
	Mat matrix1 = (Mat_<double>(3, 3) <<
		1, tan(45 * CV_PI / 180), 0,
		0, 1, 0,
		0, 0, 1);
	Mat matrix2 = (Mat_<double>(3, 3) <<
		1, 0, -256,
		0, 1, 0,
		0, 0, 1);
	Mat matrix3 = (Mat_<double>(3, 3) <<
		0.5, 0, 0,
		0, 0.5, 0,
		0, 0, 1);
	return matrix3 * matrix2 * matrix1;
}

void cvPerspective2() {
	Mat src = imread( "Lenna.png", 1 );
	Mat dst, matrix;

	matrix = myTransMat();
	warpPerspective(src, dst, matrix, src.size());

	imwrite("nonper.jpg", src);
	imwrite("per.jpg", dst);

	imshow("nonper", src);
	imshow("per", dst);
	waitKey(0);

	destroyAllWindows();

}

void ex1() {
	cvRotation(); // getRotationMatrix()
	myRotation_ex();
}

vector<KeyPoint>myDetection(Mat img) {

	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	Mat harr;
	cornerHarris(gray, harr, 2, 3, 0.05, BORDER_DEFAULT);
	normalize(harr, harr, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

	Mat harr_abs;
	convertScaleAbs(harr, harr_abs);

	int thresh = 107;
	Mat result = img.clone();
	int cnt = 0;
	for (int y = 0; y < harr.rows; y++) {
		for (int x = 0; x < harr.cols; x++) {
			if ((int)harr.at<float>(y, x) > thresh) {
				circle(result, Point(x, y), 7, Scalar(0, 0, 255), 0, 4, 0);
			}
		}
	}

	SimpleBlobDetector::Params params;
	params.minThreshold = 65;
	params.maxThreshold = 1000;
	params.filterByArea = true;
	params.minArea = 50;
	params.maxArea = 500;
	params.filterByCircularity = true;
	params.minCircularity = 0.1;
	params.filterByConvexity = false;
	params.minConvexity = 0.2;
	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;
	params.minDistBetweenBlobs = 2;


	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	std::vector<KeyPoint> keypoints;
	detector->detect(result, keypoints);

	Mat result_2;
	drawKeypoints(img, keypoints, result_2, Scalar(255, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	imshow("result", result);
	imshow("result2", result_2);
	waitKey(0);
	destroyAllWindows();

	return keypoints;
}

void ex2(){

	Mat img = imread("card_per.png", 1);
	Mat dst, matrix;
	
	int width = img.cols;
	int height = img.rows;

	vector<KeyPoint>corner = myDetection(img);

	Point2f point[4];

	point[0] = corner[2].pt;
	point[3] = corner[1].pt;
	point[1] = corner[3].pt;
	point[2] = corner[0].pt;

	Point2f shift_point[4];

	shift_point[1] = Point2f(10, height / 2 - height / 4);
	shift_point[0] = Point2f(width - 10, height / 2 - height / 4);
	shift_point[2] = Point2f(10, height / 2 + height / 4);
	shift_point[3] = Point2f(width - 10, height / 2 + height / 4);

	matrix = getPerspectiveTransform(point, shift_point);
	warpPerspective(img, dst, matrix, img.size());

	imshow("img", img);
	imshow("dst", dst);
	waitKey(0);
	destroyAllWindows();
}

int main() {
	ex2();
}