#include <iostream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
using namespace cv;
using namespace std;

// img2.jpg가 아래로 갈수록 어두워 지는 프로그램 작성하기 (산술 연산과 for 문으로 구현)

//Histogram 분석
Mat GetHistogram(Mat& src)
{
	Mat histogram;
	const int* channel_numbers = { 0 };
	float channel_range[] = { 0.0, 255.0 };
	const float* channel_ranges = channel_range;
	int number_bins = 255;

	//히스토그램 계산
	calcHist(&src, 1, channel_numbers, Mat(), histogram, 1, &number_bins, &channel_ranges);

	//히스토그램 plot
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / number_bins);

	//정규화
	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	//값과 값을 잇는 선을 그리는 방식으로 plot
	for (int i = 1; i < number_bins; i++) {
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(histogram.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(histogram.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	return histImage;
}
int dark_down(Mat img, int y, int x); //prototype

int main() {
	Mat src_img = imread("img2.jpg", 0); //이미지 영상 흑백으로 처리

	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			// 픽셀 값에 접근을 하여 어둡게 만든다
			src_img.at<uchar>(y, x) = dark_down(src_img, y, x);
		}
	}

	imshow("Test window", src_img);
	imshow("histogram window", GetHistogram(src_img));
	waitKey(0); 
	destroyWindow("Test windhow");

	return 0;
}

int dark_down(Mat img, int y, int x) {
	int e = img.at<uchar>(y, x) - (y / 2); 
	/* 왼쪽 꼭지점이 0, 0 으로 제일 어둡다.
	따라서 원하는 부분의 픽셀 값에 접근을 하고 경계값을 높이의 중간 값 즉 y/2로 하여 밝기를 조절
	*/

	if (e < 0)
		return 0; // 값이 0보다 작게 되는 경우는 0으로 하여 픽셀 변화가 없도록 하였다.
	else
		return e;
}