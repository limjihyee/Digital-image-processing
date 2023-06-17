#include <iostream>
#include <cmath>
#include <iomanip>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

//padding
Mat padding(Mat img) {
	int dftRows = getOptimalDFTSize(img.rows);
	int dftCols = getOptimalDFTSize(img.cols);

	Mat padded;
	copyMakeBorder(img, padded, 0, dftRows - img.rows, 0, dftCols - img.cols, BORDER_CONSTANT, Scalar::all(0));
	return padded;
}

// 2D DFT -> discrete fourier transform으로 discrete한 이미지를 주파수 영역에서 fourier transform 하는 함수이다.(주파수 영역으로 변환)
Mat doDft(Mat srcImg) {
	Mat floatImg;
	srcImg.convertTo(floatImg, CV_32F); // cv_32f: 32 bit floating point number으로 하나의픽셀을 표현하기 위해 32bit를 활용하겠다.
	// converto : 인수를 사용해서 지정된 값 개체를 지정한 형식으로 변환하다. floatImg의 urchar형을 float 형으로 변환한다.

	Mat complexImg;
	dft(floatImg, complexImg, DFT_COMPLEX_OUTPUT);

	return complexImg;
}

// Magnitude  영상 취득
Mat getMagnitude(Mat complexImg) {
	Mat planes[2];
	split(complexImg, planes); // 실수부, 허수부 분리

	Mat magImg;
	magnitude(planes[0], planes[1], magImg);
	magImg += Scalar::all(1);
	log(magImg, magImg); 
	//magnitude 취득
	// log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))

	return magImg;
}

// 정규화 -> 각 데이터에 따른 최솟값과 최댓값의 격차가 다르기 때문에 영향을 고르게 받게 하기 위한 같은 범위로 격차를 정규화 시킨다.
// magnitude imput array a, inputarray y, outputarray magnitude 이렇게가 magnitude에 들어가는 데이터의 값, 강노 p.6
Mat myNormalize(Mat src) {
	Mat dst;
	src.copyTo(dst);
	normalize(dst, dst, 0, 255, NORM_MINMAX);
	dst.convertTo(dst, CV_8UC1);

	return dst;
}

// phase 영상 취득
Mat getPhase(Mat complexImg) {
	Mat planes[2];
	split(complexImg, planes);
	//실수부 허수부 분리

	Mat phaImg;
	phase(planes[0], planes[1], phaImg);
	// phase 취득

	return phaImg;
}

// 좌표계 중앙 이동 ( 이론상 중앙이 0,0인데 실습상으로 왼쪽 위가 0,0으로 조정이 필요)
// fourier transform으로 한 주파수 영역으로 바뀐 이미지는 주파수 성분이 다 다르기 때문에 퓨리에 결과도 다르다. 따라서 중앙으로 모이게 하는 과정 필요(정규화 가능)
// 퓨리에 변환과 중앙 정렬해준 이미지에서 magnitude와 phase를 얻자.
Mat centralize(Mat complex)
{
	Mat planes[2];
	split(complex, planes);
	int cx = planes[0].cols / 2;
	int cy = planes[1].rows / 2;

	Mat q0Re(planes[0], Rect(0, 0, cx, cy));
	Mat q1Re(planes[0], Rect(cx, 0, cx, cy));
	Mat q2Re(planes[0], Rect(0, cy, cx, cy));
	Mat q3Re(planes[0], Rect(cx, cy, cx, cy));

	Mat tmp;
	q0Re.copyTo(tmp);
	q3Re.copyTo(q0Re);
	tmp.copyTo(q3Re);
	q1Re.copyTo(tmp);
	q2Re.copyTo(q1Re);
	tmp.copyTo(q2Re);

	Mat q0Im(planes[1], Rect(0, 0, cx, cy));
	Mat q1Im(planes[1], Rect(cx, 0, cx, cy));
	Mat q2Im(planes[1], Rect(0, cy, cx, cy));
	Mat q3Im(planes[1], Rect(cx, cy, cx, cy));

	q0Im.copyTo(tmp);
	q3Im.copyTo(q0Im);
	tmp.copyTo(q3Im);
	q1Im.copyTo(tmp);
	q2Im.copyTo(q1Im);
	tmp.copyTo(q2Im);

	Mat centerComplex;
	merge(planes, 2, centerComplex);

	return centerComplex;
}

// 2D IDFT
Mat setComplex(Mat magImg, Mat phaImg) {
	exp(magImg, magImg);
	magImg -= Scalar::all(1);
	// magnitude 계산을 반대로 수행

	Mat planes[2];
	polarToCart(magImg, phaImg, planes[0], planes[1]);
	// 극 좌표계 -> 직교 좌표계 (각도와 크기로부터 2차원 좌표)

	Mat complexImg;
	merge(planes, 2, complexImg);
	// 실수부, 허수부 합체

	return complexImg;
}

Mat doIdft(Mat complexImg) // inverse fourier transform으로 bandpass filter를 통과한 주파수 이미지를 원상으로 되돌리는 역할
// idft를 이용한 원본 영상 취득
{
	Mat idftcvt;
	idft(complexImg, idftcvt);
	//IDFT를 이용한 원본 영상 취득

	Mat planes[2];
	split(idftcvt, planes);

	Mat dst_img;
	magnitude(planes[0], planes[1], dst_img);
	normalize(dst_img, dst_img, 255, 0, NORM_MINMAX);
	dst_img.convertTo(dst_img, CV_8UC1);
	//일반 영상의 type과 표현범위로 변환

	return dst_img;
}

// Low pass filtering(LPF)
Mat doLPF(Mat srcImg) {
	// <DFT>
	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	// <LPF>
	double minVal, maxVal; // 최대값 최소값를 나타내는 변수 설정
	Point minLoc, maxLoc; // 위치를 나타내는 포인터 변수 설정
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc); // minMaxLoc (얻고자 하는 행렬, 최소값, 최대값, 최소값 위치, 최대값 위치)로 얻고 최대 최소값을 얻는다. -> 이후 정규화 하려고
	normalize(magImg, magImg, 0, 1, NORM_MINMAX); // 위에서 얻은 값을 정규화 한다.

	Mat maskImg = Mat::zeros(magImg.size(), CV_32F); // maskimg는 검정색 동그라미 1개를 만들고 주파수 성분의 이미지를 검정색으로 가려주는 역할
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 20, Scalar::all(1), -1, -1, 0);

	Mat magImg2;
	multiply(magImg, maskImg, magImg2);

	// <IDFT>
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);
}

// High pass filtering (HPF)
Mat doHPF(Mat srcImg)
{
	//DFT
	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	//RFF
	double minVal, maxVal; // 최대값 최소값를 나타내는 변수 설정
	Point minLoc, maxLoc; // 위치를 나타내는 포인터 변수 설정
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc); // minMaxLoc (얻고자 하는 행렬, 최소값, 최대값, 최소값 위치, 최대값 위치)로 얻고 최대 최소값을 얻는다. -> 이후 정규화 하려고
	normalize(magImg, magImg, 0, 1, NORM_MINMAX); // 위에서 얻은 값을 정규화 한다.

	Mat maskImg = Mat::ones(magImg.size(), CV_32F); // maskimg는 검정색 동그라미 1개를 만들고 주파수 성분의 이미지를 검정색으로 가려주는 역할
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 50, Scalar::all(0), -1, -1, 0);

	Mat magImg2;
	multiply(maskImg, magImg, magImg2);

	//IDFT
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);
}

// img1에 band pass filter 적용시켜라
// bpf는 특정 영역의 주파수만 통과시키는 필터이다. (LPF와 HPF 의 혼합), 특정 주파수를 보여주고 싶으면 주파수 성분에서 낮은 부분과 높은 부분을 가려준다.
//-> 가운데와 가장자리 즉 고주파 저주파 부분을 검정색 마스크로 가림

Mat doBPF(Mat srcImg)
{
	//DFT
	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	//BPF
	double minVal, maxVal; // 최대값 최소값를 나타내는 변수 설정
	Point minLoc, maxLoc; // 위치를 나타내는 포인터 변수 설정
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc); // minMaxLoc (얻고자 하는 행렬, 최소값, 최대값, 최소값 위치, 최대값 위치)로 얻고 최대 최소값을 얻는다. -> 이후 정규화 하려고
	normalize(magImg, magImg, 0, 1, NORM_MINMAX); // 위에서 얻은 값을 정규화 한다.

	Mat maskImg = Mat::ones(magImg.size(), CV_32F); // maskimg는 검정색 동그라미 2개를 만들고 주파수 성분의 이미지를 검정색으로 가려주는 역할
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 50, Scalar::all(1), -1, -1, 0); // 저주파에서 동그라미 생성
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 20, Scalar::all(0), -1, -1, 0); // 고주파에서 동그라미 생성

	Mat magImg2;
	multiply(magImg, maskImg, magImg2);

	//IDFT
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);
}

//1번. img1에 band pass filter를 적용시켜라
void ex1() {
	Mat img = imread("img1.jpg", 0); // img를 그대로 읽기
	Mat band_img = doBPF(img); // 읽은 img를 bandfilter 적용 시켜서 band_img로 한다.

	imshow("img", img);
	imshow("band_img", band_img);
	waitKey(0);
	destroyAllWindows();
}

// 2번을 위한 sobel filter 구현
int myKernelConv3x3(uchar* arr, int kernel[][3], int x, int y, int width, int height) {
	int sum = 0;
	int sumKernel = 0;

	//특정 화소의 모든 아웃화소에 대해 계산하도록 반복문 구성
	for (int j = -1; j <= 1; j++) {
		for (int i = -1; i <= 1; i++) {
			if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width) {
				// 영상 가장자리에서 영상 밖의 화소를 읽지 않도록 하는 조건문
				sum += arr[(y + j) * width + (x + i)] * kernel[i + 1][j + 1];
				sumKernel += kernel[i + 1][j + 1];
			}
		}
	}
	if (sumKernel != 0) {
		return sum / sumKernel;
	} // 합이 1로 정규화 되도록 해 영상의 밝기변화 방지
	else {
		return sum;
	}
}

Mat mySobelFilter(Mat srcImg) {
	int kernelX[3][3] = { -1, 0, 1,
		-2, 0, 2,
		-1, 0, 1 }; // 가로 방향으로 sobel 마스크
	int kernelY[3][3] = { -1, -2, -1,
		0, 0, 0,
		1, 2, 1 }; // 세로 방향으로 sobel 마스크
// 마스크 합이 0이 되므로 1로 정규화하는 과정은 필요없음

	Mat dstImg(srcImg.size(), CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;
	int width = srcImg.cols;
	int height = srcImg.rows;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstData[y * width + x] = (abs(myKernelConv3x3(srcData, kernelX, x, y, width, height)) +
				abs(myKernelConv3x3(srcData, kernelY, x, y, width, height))) / 2;
			// 두 에지 결과의 절대값 합 형태로 최종 결과 도출
		}
	}
	return dstImg;
}

Mat doFDF(Mat srcImg) {

	//DFT
	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	//BPF
	double minVal, maxVal; // 최대값 최소값를 나타내는 변수 설정
	Point minLoc, maxLoc; // 위치를 나타내는 포인터 변수 설정
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc); // minMaxLoc (얻고자 하는 행렬, 최소값, 최대값, 최소값 위치, 최대값 위치)로 얻고 최대 최소값을 얻는다. -> 이후 정규화 하려고
	normalize(magImg, magImg, 0, 1, NORM_MINMAX); // 위에서 얻은 값을 정규화 한다.
	
	Mat maskImg = Mat::ones(magImg.size(), CV_32F); // maskimg는 검정색 동그라미 1개를 만들고 주파수 성분의 이미지를 검정색으로 가려주는 역할

	// line과 circle 필터를 만들어서 주파수 성분에서 생성하고 부자연스러운 직선 줄무늬를 자연스럽게 보이도록 할 경우 사용한다.
	// line(input output array, pt1, pt2, scalar(b,g,r), thickness(굵기))로 나타난다. pt1, pt2는 직선의 양 끝 좌표를 의미한다.
	line(maskImg, Point(maskImg.rows / 2 - 50), Point(maskImg.rows / 2 + 50), Scalar::all(1), 50);
	line(maskImg, Point(maskImg.rows / 2 + 50), Point(maskImg.rows / 2 - 50), Scalar::all(1), 50);
	line(maskImg, Point(maskImg.cols / 2 - 50), Point(maskImg.cols / 2 + 50), Scalar::all(1), 20);
	line(maskImg, Point(maskImg.cols / 2 + 50), Point(maskImg.cols / 2 - 50), Scalar::all(1), 20);

	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 20, Scalar::all(0), -1, -1, 0);

	Mat magImg2;
	multiply(maskImg, magImg, magImg2);

	//IDFT
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);

}

void ex2() {

	// spatial domain에서 sobel filter 구현
	Mat img2 = imread("img2.jpg", 0); // img를 그대로 읽기
	Mat sp_img2 = mySobelFilter(img2);
	imshow("spatial domain", sp_img2);

	// frequency domain에서 sobel filter 구현
	Mat fre_img2 = doFDF(img2);
	imshow("frequency domain", fre_img2);

	// 그냥 high pass filter
	Mat hig_img2 = doHPF(img2);
	imshow("high pass filter domain", hig_img2);

	waitKey(0);
	destroyAllWindows();
}

// 3번 img3에서 나타나는 flickering 현상을 frequency domain filtering을 통해 제거해라
Mat doRDF(Mat srcImg) {

	//DFT
	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	//BPF
	double minVal, maxVal; // 최대값 최소값를 나타내는 변수 설정
	Point minLoc, maxLoc; // 위치를 나타내는 포인터 변수 설정
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc); // minMaxLoc (얻고자 하는 행렬, 최소값, 최대값, 최소값 위치, 최대값 위치)로 얻고 최대 최소값을 얻는다. -> 이후 정규화 하려고
	normalize(magImg, magImg, 0, 1, NORM_MINMAX); // 위에서 얻은 값을 정규화 한다.

	Mat maskImg = Mat::ones(magImg.size(), CV_32F); // maskimg는 검정색 동그라미 1개를 만들고 주파수 성분의 이미지를 검정색으로 가려주는 역할

	rectangle(maskImg, Rect(Point(maskImg.cols / 2 - 10, 0), Point(maskImg.cols / 2 + 10, maskImg.rows)), Scalar::all(0), -1);
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 3, Scalar::all(1), -1, -1, 0);

	Mat magImg2;
	multiply(maskImg, magImg, magImg2);

	//IDFT
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);

}

void ex3() {
	Mat ex3_img = imread("img3.jpg", 0); // 색 그대로 읽기
	Mat rem_img = doRDF(ex3_img);
	imshow("remove flickering frequency domain", rem_img);

	waitKey(0);
	destroyAllWindows();
}

int main() {
	ex3();
}