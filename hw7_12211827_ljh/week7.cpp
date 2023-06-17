#include "week7.h"

void ex1() {
	Mat src_img = imread("apple.jpeg", 1);

	Mat hsv_img = MyBGR2HSV(src_img);
	Mat hsv2_img;
	Mat mask2_img;

	Mat mask_img = inRange(hsv_img, Scalar(50, 50, 0), Scalar(50, 100, 100)); // 마스크 생성
	Mat dst_img;

	bitwise_and(src_img, mask_img, dst_img);

	con_Color(hsv_img);
	imshow("original사진을 HSV로 변환한 사진", hsv_img);
	waitKey(0);
	destroyAllWindows();
}

void ex2() {
	//opencv 이용
	Mat src_img = imread("beach.jpg", 1);
	resize(src_img, src_img, Size(src_img.cols / 2, src_img.rows / 2));
	Mat beach = CvKMeans(src_img, 20);

	cout << "k = 20"; //k의 값을 2, 5, 10, 20으로 나눠 cluster 하자
	imshow("beach에 opencv를 이용한 k-mean 사진", beach);
	waitKey(0);
	destroyAllWindows();
}

void ex3() {
	
	Mat src_img = imread("apple.jpeg", 1);
	Mat apple_New = MyKmeans(src_img, 20);
	imshow("apple에 Myk-mean 적용사진", apple_New);

	Mat hsv_img = MyBGR2HSV(apple_New);
	Mat hsv2_img;
	Mat mask2_img;

	Mat mask_img = inRange(hsv_img, Scalar(50, 50, 0), Scalar(50, 100, 100)); // 마스크 생성
	Mat dst_img;

	bitwise_and(apple_New, mask_img, dst_img);

	con_Color(hsv_img);
	imshow("HSV로 변환한 사진", hsv_img);

	waitKey(0);
	destroyAllWindows();
}


int main()
{
	ex3();

	return 0;
}