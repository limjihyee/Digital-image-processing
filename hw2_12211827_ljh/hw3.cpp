#include <iostream>
#include "opencv2/core/core.hpp" // Mat class�� ���� data structure �� ��� ��ƾ�� �����ϴ� ���
#include "opencv2/highgui/highgui.hpp" // GUI�� ���õ� ��Ҹ� �����ϴ� ���(imshow ��)
#include "opencv2/imgproc/imgproc.hpp" // ���� �̹��� ó�� �Լ��� �����ϴ� ���
using namespace cv;
using namespace std;

int main()
{
	Mat src3 = imread("img3.jpg", 1);	//ȭ������� ���ּ�
	Mat src4 = imread("img4.jpg", 1);	//������ ���� ���
	Mat src5 = imread("img5.jpg", 1);	//SPACEX �ΰ�

	resize(src3, src3, Size(src4.cols, src4.rows));
	// ȭ���� ���� ���� ����� ������ ����

	Mat final_img;
	subtract(src3, src4, final_img);

	Mat logo = final_img(Rect(500, 550, src5.cols, src5.rows));

	// mask�� ����� ���ؼ� logo �̹����� gray scales�� �ٲٰ� binary image�� ��ȯ
	// mask�� logo�κ��� ���(255) ������ ������(0)
	// mask_inv�� logo�κ��� ������ ������ ����̴�.

	//1��(binarization ���� �κ�)
	Mat gray_img, binary_img1, binary_img2, img1_fg, img2_bg, logo_final;
	cvtColor(src5, gray_img, CV_BGR2GRAY); // ����ȭ�� ���ؼ� color���� grayscale �������� ��ȯ

	
	threshold(gray_img, binary_img1, 210, 255, THRESH_BINARY); // �Ӱ谪 ���� ����ȭ, thresh�� 127 �׸��� maximum�� 255�̴�. 
															   // ����� ������ ��� ���� ���������� ��ȯ (����ũ ��� ��)
	
	
	//binary_img2�� binary_img�� ���� ����ȯ�� ������ ����
	bitwise_not(binary_img1, binary_img2); //binary_img2�� ����ũ ��� iverse ��
	
	bitwise_and(logo, logo, img1_fg, binary_img1);	//mask�� img1_fg ���� logo�� logo�� and ���� ��Ű�� binary_img1 mask�� �Ͽ� 
	// binary_img1�� �����Ͽ���. �̷��� �ΰ��� ���ڸ� �������� �ϰ� ��濡�� final�� �Ϻκ��� ����ȴ�.
	bitwise_and(src5, src5, img2_bg, binary_img2);	//mask�� img2_bg
	
	add (img1_fg, img2_bg, logo_final);

	// image���� �ΰ� �� �κп� logo_final�� �־��ֱ�
	final_img(Rect(500, 550, src5.cols, src5.rows)) = logo_final + 1;
	imshow("hw3", final_img);
	waitKey(0);
	// �� �ΰ��� img1_fg�� img2_bg�� ����


	return 0;
}