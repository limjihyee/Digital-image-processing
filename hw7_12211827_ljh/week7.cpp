#include "week7.h"

void ex1() {
	Mat src_img = imread("apple.jpeg", 1);

	Mat hsv_img = MyBGR2HSV(src_img);
	Mat hsv2_img;
	Mat mask2_img;

	Mat mask_img = inRange(hsv_img, Scalar(50, 50, 0), Scalar(50, 100, 100)); // ����ũ ����
	Mat dst_img;

	bitwise_and(src_img, mask_img, dst_img);

	con_Color(hsv_img);
	imshow("original������ HSV�� ��ȯ�� ����", hsv_img);
	waitKey(0);
	destroyAllWindows();
}

void ex2() {
	//opencv �̿�
	Mat src_img = imread("beach.jpg", 1);
	resize(src_img, src_img, Size(src_img.cols / 2, src_img.rows / 2));
	Mat beach = CvKMeans(src_img, 20);

	cout << "k = 20"; //k�� ���� 2, 5, 10, 20���� ���� cluster ����
	imshow("beach�� opencv�� �̿��� k-mean ����", beach);
	waitKey(0);
	destroyAllWindows();
}

void ex3() {
	
	Mat src_img = imread("apple.jpeg", 1);
	Mat apple_New = MyKmeans(src_img, 20);
	imshow("apple�� Myk-mean �������", apple_New);

	Mat hsv_img = MyBGR2HSV(apple_New);
	Mat hsv2_img;
	Mat mask2_img;

	Mat mask_img = inRange(hsv_img, Scalar(50, 50, 0), Scalar(50, 100, 100)); // ����ũ ����
	Mat dst_img;

	bitwise_and(apple_New, mask_img, dst_img);

	con_Color(hsv_img);
	imshow("HSV�� ��ȯ�� ����", hsv_img);

	waitKey(0);
	destroyAllWindows();
}


int main()
{
	ex3();

	return 0;
}