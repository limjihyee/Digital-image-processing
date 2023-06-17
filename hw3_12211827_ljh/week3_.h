#include <iostream>
#include <cmath>
#include <iomanip>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

//9x9 ����ũ�� ���� convolution
int myKernerConv9x9(uchar* arr, double kernel[][9], int x, int y, int width, int height) // uchar �׷��� �̹����� ����
{
	int sum = 0;
	int sumKernel = 0;

	//Ư�� ȭ���� ��� �̿�ȭ�ҿ� ���� ����ϵ��� �ݺ��� ����
	//9x9 kernel�� ����
	for (int j = -4; j <= 4; j++) { // 9x9 �ϱ� ��� ������ 9���� -4~4
		for (int i = -4; i <= 4; i++) {
			if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width) {
				//���� �����ڸ����� ���� ���� ȭ�Ҹ� ���� �ʵ��� �ϴ� ���ǹ�
				sum += arr[(y + j) * width + (x + i)] * kernel[i + 4][j + 4];
				sumKernel += kernel[i + 4][j + 4];
			}
		}
	}

	if (sumKernel != 0) { return sum / sumKernel; }	//���� 1�� ����ȭ�ǵ��� �Ͽ�, ������ ��� ��ȭ�� ����******
	else return sum;
}

//3x3 ����ũ�� ���� convolution
int myKernerConv3x3(uchar* arr, int kernel[][3], int x, int y, int width, int height) {
	int sum = 0;
	int sumKernel = 0;

	//Ư�� ȭ���� ��� �̿�ȭ�ҿ� ���� ����ϵ��� �ݺ��� ����
	for (int j = -1; j <= 1; j++) {
		for (int i = -1; i <= 1; i++) {
			if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width) {
				//���� �����ڸ����� ���� ���� ȭ�Ҹ� ���� �ʵ��� �ϴ� ���ǹ�
				sum += arr[(y + j) * width + (x + i)] * kernel[i + 1][j + 1];
				sumKernel += kernel[i + 1][j + 1];
			}
		}
	}

	if (sumKernel != 0) { return sum / sumKernel; }	//���� 1�� ����ȭ�ǵ��� �Ͽ�, ������ ��� ��ȭ�� ����******
	else return sum;
}

//Gaussian filter
Mat myGaussianFilter(Mat srcImg)
{
	int width = srcImg.cols;
	int height = srcImg.rows;
	double kernel[9][9];	//9x9 ������ Gaussian ����ũ �迭 // 9x9 ������ ����ũ �迭�� ��� ���� �� ������ �Լ��� ���� �ۼ� 
	double sigma = 2.0;	//9x9 Gaussian mask�� ũ�� -> sigma = 2 // sigma�� Ŀ�� ���� ���� �������⿡ 1�� Ƽ�� ���� ���� 2�� ����
	double r;
	double s = 2 * sigma * sigma;
	const int PI = 3.14;

	// ����þ� ���Ϳ� ���� ������ �Ʒ������� ǥ��
	for (int j = -4; j <= 4; j++) {
		for (int i = -4; i <= 4; i++) {
			r = sqrt(i * i + j * j); // ������ sqrt�̿�
			kernel[i + 4][j + 4] = (exp(-(r * r) / s)) / (PI * s);
		}
	}

	Mat dstImg(srcImg.size(), CV_8UC1);
	//CV_8UC1 : 2^8bits(0~255), one(gray) channel
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++)
			dstData[y * width + x] = myKernerConv9x9(srcData, kernel, x, y, width, height);
		//�ռ� ������ convolution�� ����ũ �迭�� �Է��� ���
	}
	return dstImg;
}

//Gaussian filter about color image
Mat myGaussianFilter_color(Mat srcImg)
{
	int width = srcImg.cols;
	int height = srcImg.rows;
	double sigma = 2.0;	//9x9 Gaussian mask�� ũ�� -> sigma = 2
	double r;
	double s = 2 * sigma * sigma;
	const int PI = 3.14;
	double kernel[9][9];	//9x9 ������ Gaussian ����ũ �迭

	for (int j = -4; j <= 4; j++) {
		for (int i = -4; i <= 4; i++) {
			r = sqrt(i * i + j * j);
			kernel[i + 4][j + 4] = (exp(-(r * r) / s)) / (PI * s);
		}
	}

	Mat dstImg(srcImg.size(), CV_8UC3);
	//CV_8UC1 : 2^8bits(0~255), one(gray) channel
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	Mat RGB[3];
	split(srcImg, RGB);	//split()�Լ� : �̹����� ä�κ�(R,G,B)�� �и�
	uchar* BlueData = RGB[0].data; // ��� �̹����̱⿡ uchar�� �̿��Ͽ� ���
	uchar* GreenData = RGB[1].data;
	uchar* RedData = RGB[2].data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int index = (y * width + x) * 3;
			dstData[index + 0] = myKernerConv9x9(BlueData, kernel, x, y, width, height);
			dstData[index + 1] = myKernerConv9x9(GreenData, kernel, x, y, width, height);
			dstData[index + 2] = myKernerConv9x9(RedData, kernel, x, y, width, height);
			//�ռ� ������ convolution�� ����ũ �迭�� �Է��� ���
		}
	}
	return dstImg;
}

//Histogram
Mat GetHistogram(Mat& src)
{
	Mat histogram;
	const int* channel_numbers = { 0 };
	float channel_range[] = { 0.0, 255.0 };
	const float* channel_ranges = channel_range;
	int number_bins = 255;

	// ������׷� ���
	calcHist(&src, 1, channel_numbers, Mat(), histogram, 1, &number_bins, &channel_ranges);

	//������׷� plot
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / number_bins);

	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat()); // ����ȭ

	for (int i = 1; i < number_bins; i++) {
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(histogram.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(histogram.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0); // ���� ���� �մ� ���� �׸��� ������� plot
	}
	return histImage;
}

//Generate Salts and Pepper noise
Mat salt_and_pepper_noise(Mat img) {
	srand(time(NULL));
	int black, white;
	int height = img.rows;
	int width = img.cols;
	uchar Black = 0;
	uchar White = 255;
	Mat dst_img = img;

	cout << "Enter how many black dots you wanna mask: "; cin >> black;
	cout << "Enter how many white dots you wanna mask: "; cin >> white;

	for (int i = 0; i < black; i++) {
		img.at<uchar>(rand() % height, rand() % width) = Black;
	}

	for (int i = 0; i < white; i++) {
		img.at<uchar>(rand() % height, rand() % width) = White;
	}

	return dst_img;
}

//45���� 135���� �밢 edge�� �����ϴ� Sobel filter
Mat mySobelFilter(Mat srcImg) {
	int kernelX[3][3] = { 0, 1, 2,
						-1, 0, 1,
						-2, -1, 0 };	//45���� ���� Sobel kernel ����ũ(���ι���)
	int kernelY[3][3] = { -2, -1, 0,
							-1, 0, 1,
							0, 1, 2 };//135���� ���� Sobel kernel ����ũ(���ι���)

	//����ũ ���� 0�� �ǹǷ� 1�� ����ȭ�ϴ� ������ �ʿ� X
	Mat dstImg(srcImg.size(), CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;
	int width = srcImg.cols;
	int height = srcImg.rows;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++)
			dstData[y * width + x] = (abs(myKernerConv3x3(srcData, kernelX, x, y, width, height))
				+ abs(myKernerConv3x3(srcData, kernelY, x, y, width, height))) / 2;
		//�� edge ����� ���밪�� �� ���·� ������� ����
	}

	return dstImg;
}


Mat mySampling(Mat srcImg) {
	int width = srcImg.cols / 2;
	int height = srcImg.rows / 2;
	//���� ���ΰ� �Է� ������ ������ ������ ���� ����

	Mat dstImg(height, width, CV_8UC3);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstImg.at<Vec3b>(y, x) = srcImg.at<Vec3b>(y * 2, x * 2); // 2�� �������� �ε��� �� ū ������ ���� ���� ����, ���⼱ �÷������� �����ϱ� vec3b��
		}
	}

	return dstImg;
}

//Gaussian pyramid
//���� ���� �ػ��� ������ �ݺ������� ���� (���� ������ �����ϴ� ��)
vector<Mat> myGaussianPyramid(Mat srcImg) {
	vector<Mat> Vec;	//���� ������ ��Ƽ� �����ϱ� ���� STL�� vector �����̳� ���

	Vec.push_back(srcImg);
	for (int i = 0; i < 4; i++) {
		srcImg = mySampling(srcImg);	//�ռ� ������ down sampling
		srcImg = myGaussianFilter_color(srcImg); //�ռ� ������ gaussian filtering color�� ��

		Vec.push_back(srcImg);	//vector �����̳ʿ� �ϳ��� ó������� ����
	}

	return Vec;
}

//Laplacian pyramid
//���� �ػ��� ����� �۾��� ���� ���� �������� �����ϴ� ���
// up sampling�ϰ� �� ������ ���Ͽ� ���� �ػ��� ���� ����
vector<Mat> myLaplacianPyramid(Mat srcImg) {
	vector<Mat> Vec;

	for (int i = 0; i < 4; i++) {
		if (i != 3) {
			Mat highImg = srcImg;	//�����ϱ� ���� ������ ���

			srcImg = mySampling(srcImg);
			srcImg = myGaussianFilter_color(srcImg);

			Mat lowImg = srcImg;
			resize(lowImg, lowImg, highImg.size());
			//�۾��� ������ ����� ������ ũ��� Ȯ��
			Vec.push_back(highImg - lowImg + 128);
			//�� ������ �����̳ʿ� ����
			//128 ������ ���� �� ���󿡼� �����÷ο츦 �����ϱ� ����
		}
		else
			Vec.push_back(srcImg);
	}

	return Vec;
}