#include <iostream>
#include <cmath>
#include <iomanip>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

//9x9 마스크에 대한 convolution
int myKernerConv9x9(uchar* arr, double kernel[][9], int x, int y, int width, int height) // uchar 그레이 이미지에 접근
{
	int sum = 0;
	int sumKernel = 0;

	//특정 화소의 모든 이웃화소에 대해 계산하도록 반복문 구성
	//9x9 kernel을 생성
	for (int j = -4; j <= 4; j++) { // 9x9 니까 행렬 각각이 9개씩 -4~4
		for (int i = -4; i <= 4; i++) {
			if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width) {
				//영상 가장자리에서 영상 밖의 화소를 읽지 않도록 하는 조건문
				sum += arr[(y + j) * width + (x + i)] * kernel[i + 4][j + 4];
				sumKernel += kernel[i + 4][j + 4];
			}
		}
	}

	if (sumKernel != 0) { return sum / sumKernel; }	//합이 1로 정규화되도록 하여, 영상의 밝기 변화를 방지******
	else return sum;
}

//3x3 마스크에 대한 convolution
int myKernerConv3x3(uchar* arr, int kernel[][3], int x, int y, int width, int height) {
	int sum = 0;
	int sumKernel = 0;

	//특정 화소의 모든 이웃화소에 대해 계산하도록 반복문 구성
	for (int j = -1; j <= 1; j++) {
		for (int i = -1; i <= 1; i++) {
			if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width) {
				//영상 가장자리에서 영상 밖의 화소를 읽지 않도록 하는 조건문
				sum += arr[(y + j) * width + (x + i)] * kernel[i + 1][j + 1];
				sumKernel += kernel[i + 1][j + 1];
			}
		}
	}

	if (sumKernel != 0) { return sum / sumKernel; }	//합이 1로 정규화되도록 하여, 영상의 밝기 변화를 방지******
	else return sum;
}

//Gaussian filter
Mat myGaussianFilter(Mat srcImg)
{
	int width = srcImg.cols;
	int height = srcImg.rows;
	double kernel[9][9];	//9x9 형태의 Gaussian 마스크 배열 // 9x9 형태의 마스크 배열을 모두 넣을 수 없으니 함수로 따로 작성 
	double sigma = 2.0;	//9x9 Gaussian mask의 크기 -> sigma = 2 // sigma가 커질 수록 블러가 많아지기에 1은 티가 별로 없어 2로 설정
	double r;
	double s = 2 * sigma * sigma;
	const int PI = 3.14;

	// 가우시안 필터에 대한 공식을 아래식으로 표현
	for (int j = -4; j <= 4; j++) {
		for (int i = -4; i <= 4; i++) {
			r = sqrt(i * i + j * j); // 제곱은 sqrt이용
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
		//앞서 구현한 convolution에 마스크 배열을 입력해 사용
	}
	return dstImg;
}

//Gaussian filter about color image
Mat myGaussianFilter_color(Mat srcImg)
{
	int width = srcImg.cols;
	int height = srcImg.rows;
	double sigma = 2.0;	//9x9 Gaussian mask의 크기 -> sigma = 2
	double r;
	double s = 2 * sigma * sigma;
	const int PI = 3.14;
	double kernel[9][9];	//9x9 형태의 Gaussian 마스크 배열

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
	split(srcImg, RGB);	//split()함수 : 이미지를 채널별(R,G,B)로 분리
	uchar* BlueData = RGB[0].data; // 흑백 이미지이기에 uchar을 이용하여 사용
	uchar* GreenData = RGB[1].data;
	uchar* RedData = RGB[2].data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int index = (y * width + x) * 3;
			dstData[index + 0] = myKernerConv9x9(BlueData, kernel, x, y, width, height);
			dstData[index + 1] = myKernerConv9x9(GreenData, kernel, x, y, width, height);
			dstData[index + 2] = myKernerConv9x9(RedData, kernel, x, y, width, height);
			//앞서 구현한 convolution에 마스크 배열을 입력해 사용
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

	// 히스토그램 계산
	calcHist(&src, 1, channel_numbers, Mat(), histogram, 1, &number_bins, &channel_ranges);

	//히스토그램 plot
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / number_bins);

	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat()); // 정규화

	for (int i = 1; i < number_bins; i++) {
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(histogram.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(histogram.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0); // 값과 값을 잇는 선을 그리는 방식으로 plot
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

//45도와 135도의 대각 edge를 검출하는 Sobel filter
Mat mySobelFilter(Mat srcImg) {
	int kernelX[3][3] = { 0, 1, 2,
						-1, 0, 1,
						-2, -1, 0 };	//45도에 대한 Sobel kernel 마스크(가로방향)
	int kernelY[3][3] = { -2, -1, 0,
							-1, 0, 1,
							0, 1, 2 };//135도에 대한 Sobel kernel 마스크(세로방향)

	//마스크 합이 0이 되므로 1로 정규화하는 과정은 필요 X
	Mat dstImg(srcImg.size(), CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;
	int width = srcImg.cols;
	int height = srcImg.rows;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++)
			dstData[y * width + x] = (abs(myKernerConv3x3(srcData, kernelX, x, y, width, height))
				+ abs(myKernerConv3x3(srcData, kernelY, x, y, width, height))) / 2;
		//두 edge 결과의 절대값의 합 형태로 최종결과 도출
	}

	return dstImg;
}


Mat mySampling(Mat srcImg) {
	int width = srcImg.cols / 2;
	int height = srcImg.rows / 2;
	//가로 세로가 입력 영상의 절반인 영상을 먼저 생성

	Mat dstImg(height, width, CV_8UC3);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstImg.at<Vec3b>(y, x) = srcImg.at<Vec3b>(y * 2, x * 2); // 2배 간격으로 인덱싱 해 큰 영상을 작은 영상에 대입, 여기선 컬러영상을 받으니까 vec3b로
		}
	}

	return dstImg;
}

//Gaussian pyramid
//점차 작은 해상도의 영상을 반복적으로 생성 (작은 영상을 생성하는 것)
vector<Mat> myGaussianPyramid(Mat srcImg) {
	vector<Mat> Vec;	//여러 영상을 모아서 저장하기 위해 STL의 vector 컨테이너 사용

	Vec.push_back(srcImg);
	for (int i = 0; i < 4; i++) {
		srcImg = mySampling(srcImg);	//앞서 구현한 down sampling
		srcImg = myGaussianFilter_color(srcImg); //앞서 구현한 gaussian filtering color일 때

		Vec.push_back(srcImg);	//vector 컨테이너에 하나씩 처리결과를 삽입
	}

	return Vec;
}

//Laplacian pyramid
//높은 해상도의 영상과 작아진 영상 간의 차영상을 저장하는 방식
// up sampling하고 차 영상을 더하여 높은 해상도의 영상 복원
vector<Mat> myLaplacianPyramid(Mat srcImg) {
	vector<Mat> Vec;

	for (int i = 0; i < 4; i++) {
		if (i != 3) {
			Mat highImg = srcImg;	//수행하기 이전 영상을 백업

			srcImg = mySampling(srcImg);
			srcImg = myGaussianFilter_color(srcImg);

			Mat lowImg = srcImg;
			resize(lowImg, lowImg, highImg.size());
			//작아진 영상을 백업한 영상의 크기로 확대
			Vec.push_back(highImg - lowImg + 128);
			//차 영상을 컨테이너에 삽입
			//128 더해준 것은 차 영상에서 오버플로우를 방지하기 위함
		}
		else
			Vec.push_back(srcImg);
	}

	return Vec;
}