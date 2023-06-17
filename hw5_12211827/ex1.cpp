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

// 2D DFT -> discrete fourier transform���� discrete�� �̹����� ���ļ� �������� fourier transform �ϴ� �Լ��̴�.(���ļ� �������� ��ȯ)
Mat doDft(Mat srcImg) {
	Mat floatImg;
	srcImg.convertTo(floatImg, CV_32F); // cv_32f: 32 bit floating point number���� �ϳ����ȼ��� ǥ���ϱ� ���� 32bit�� Ȱ���ϰڴ�.
	// converto : �μ��� ����ؼ� ������ �� ��ü�� ������ �������� ��ȯ�ϴ�. floatImg�� urchar���� float ������ ��ȯ�Ѵ�.

	Mat complexImg;
	dft(floatImg, complexImg, DFT_COMPLEX_OUTPUT);

	return complexImg;
}

// Magnitude  ���� ���
Mat getMagnitude(Mat complexImg) {
	Mat planes[2];
	split(complexImg, planes); // �Ǽ���, ����� �и�

	Mat magImg;
	magnitude(planes[0], planes[1], magImg);
	magImg += Scalar::all(1);
	log(magImg, magImg); 
	//magnitude ���
	// log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))

	return magImg;
}

// ����ȭ -> �� �����Ϳ� ���� �ּڰ��� �ִ��� ������ �ٸ��� ������ ������ ���� �ް� �ϱ� ���� ���� ������ ������ ����ȭ ��Ų��.
// magnitude imput array a, inputarray y, outputarray magnitude �̷��԰� magnitude�� ���� �������� ��, ���� p.6
Mat myNormalize(Mat src) {
	Mat dst;
	src.copyTo(dst);
	normalize(dst, dst, 0, 255, NORM_MINMAX);
	dst.convertTo(dst, CV_8UC1);

	return dst;
}

// phase ���� ���
Mat getPhase(Mat complexImg) {
	Mat planes[2];
	split(complexImg, planes);
	//�Ǽ��� ����� �и�

	Mat phaImg;
	phase(planes[0], planes[1], phaImg);
	// phase ���

	return phaImg;
}

// ��ǥ�� �߾� �̵� ( �̷л� �߾��� 0,0�ε� �ǽ������� ���� ���� 0,0���� ������ �ʿ�)
// fourier transform���� �� ���ļ� �������� �ٲ� �̹����� ���ļ� ������ �� �ٸ��� ������ ǻ���� ����� �ٸ���. ���� �߾����� ���̰� �ϴ� ���� �ʿ�(����ȭ ����)
// ǻ���� ��ȯ�� �߾� �������� �̹������� magnitude�� phase�� ����.
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
	// magnitude ����� �ݴ�� ����

	Mat planes[2];
	polarToCart(magImg, phaImg, planes[0], planes[1]);
	// �� ��ǥ�� -> ���� ��ǥ�� (������ ũ��κ��� 2���� ��ǥ)

	Mat complexImg;
	merge(planes, 2, complexImg);
	// �Ǽ���, ����� ��ü

	return complexImg;
}

Mat doIdft(Mat complexImg) // inverse fourier transform���� bandpass filter�� ����� ���ļ� �̹����� �������� �ǵ����� ����
// idft�� �̿��� ���� ���� ���
{
	Mat idftcvt;
	idft(complexImg, idftcvt);
	//IDFT�� �̿��� ���� ���� ���

	Mat planes[2];
	split(idftcvt, planes);

	Mat dst_img;
	magnitude(planes[0], planes[1], dst_img);
	normalize(dst_img, dst_img, 255, 0, NORM_MINMAX);
	dst_img.convertTo(dst_img, CV_8UC1);
	//�Ϲ� ������ type�� ǥ�������� ��ȯ

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
	double minVal, maxVal; // �ִ밪 �ּҰ��� ��Ÿ���� ���� ����
	Point minLoc, maxLoc; // ��ġ�� ��Ÿ���� ������ ���� ����
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc); // minMaxLoc (����� �ϴ� ���, �ּҰ�, �ִ밪, �ּҰ� ��ġ, �ִ밪 ��ġ)�� ��� �ִ� �ּҰ��� ��´�. -> ���� ����ȭ �Ϸ���
	normalize(magImg, magImg, 0, 1, NORM_MINMAX); // ������ ���� ���� ����ȭ �Ѵ�.

	Mat maskImg = Mat::zeros(magImg.size(), CV_32F); // maskimg�� ������ ���׶�� 1���� ����� ���ļ� ������ �̹����� ���������� �����ִ� ����
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
	double minVal, maxVal; // �ִ밪 �ּҰ��� ��Ÿ���� ���� ����
	Point minLoc, maxLoc; // ��ġ�� ��Ÿ���� ������ ���� ����
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc); // minMaxLoc (����� �ϴ� ���, �ּҰ�, �ִ밪, �ּҰ� ��ġ, �ִ밪 ��ġ)�� ��� �ִ� �ּҰ��� ��´�. -> ���� ����ȭ �Ϸ���
	normalize(magImg, magImg, 0, 1, NORM_MINMAX); // ������ ���� ���� ����ȭ �Ѵ�.

	Mat maskImg = Mat::ones(magImg.size(), CV_32F); // maskimg�� ������ ���׶�� 1���� ����� ���ļ� ������ �̹����� ���������� �����ִ� ����
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 50, Scalar::all(0), -1, -1, 0);

	Mat magImg2;
	multiply(maskImg, magImg, magImg2);

	//IDFT
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);
}

// img1�� band pass filter ������Ѷ�
// bpf�� Ư�� ������ ���ļ��� �����Ű�� �����̴�. (LPF�� HPF �� ȥ��), Ư�� ���ļ��� �����ְ� ������ ���ļ� ���п��� ���� �κа� ���� �κ��� �����ش�.
//-> ����� �����ڸ� �� ������ ������ �κ��� ������ ����ũ�� ����

Mat doBPF(Mat srcImg)
{
	//DFT
	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	//BPF
	double minVal, maxVal; // �ִ밪 �ּҰ��� ��Ÿ���� ���� ����
	Point minLoc, maxLoc; // ��ġ�� ��Ÿ���� ������ ���� ����
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc); // minMaxLoc (����� �ϴ� ���, �ּҰ�, �ִ밪, �ּҰ� ��ġ, �ִ밪 ��ġ)�� ��� �ִ� �ּҰ��� ��´�. -> ���� ����ȭ �Ϸ���
	normalize(magImg, magImg, 0, 1, NORM_MINMAX); // ������ ���� ���� ����ȭ �Ѵ�.

	Mat maskImg = Mat::ones(magImg.size(), CV_32F); // maskimg�� ������ ���׶�� 2���� ����� ���ļ� ������ �̹����� ���������� �����ִ� ����
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 50, Scalar::all(1), -1, -1, 0); // �����Ŀ��� ���׶�� ����
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 20, Scalar::all(0), -1, -1, 0); // �����Ŀ��� ���׶�� ����

	Mat magImg2;
	multiply(magImg, maskImg, magImg2);

	//IDFT
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);
}

//1��. img1�� band pass filter�� ������Ѷ�
void ex1() {
	Mat img = imread("img1.jpg", 0); // img�� �״�� �б�
	Mat band_img = doBPF(img); // ���� img�� bandfilter ���� ���Ѽ� band_img�� �Ѵ�.

	imshow("img", img);
	imshow("band_img", band_img);
	waitKey(0);
	destroyAllWindows();
}

// 2���� ���� sobel filter ����
int myKernelConv3x3(uchar* arr, int kernel[][3], int x, int y, int width, int height) {
	int sum = 0;
	int sumKernel = 0;

	//Ư�� ȭ���� ��� �ƿ�ȭ�ҿ� ���� ����ϵ��� �ݺ��� ����
	for (int j = -1; j <= 1; j++) {
		for (int i = -1; i <= 1; i++) {
			if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width) {
				// ���� �����ڸ����� ���� ���� ȭ�Ҹ� ���� �ʵ��� �ϴ� ���ǹ�
				sum += arr[(y + j) * width + (x + i)] * kernel[i + 1][j + 1];
				sumKernel += kernel[i + 1][j + 1];
			}
		}
	}
	if (sumKernel != 0) {
		return sum / sumKernel;
	} // ���� 1�� ����ȭ �ǵ��� �� ������ ��⺯ȭ ����
	else {
		return sum;
	}
}

Mat mySobelFilter(Mat srcImg) {
	int kernelX[3][3] = { -1, 0, 1,
		-2, 0, 2,
		-1, 0, 1 }; // ���� �������� sobel ����ũ
	int kernelY[3][3] = { -1, -2, -1,
		0, 0, 0,
		1, 2, 1 }; // ���� �������� sobel ����ũ
// ����ũ ���� 0�� �ǹǷ� 1�� ����ȭ�ϴ� ������ �ʿ����

	Mat dstImg(srcImg.size(), CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;
	int width = srcImg.cols;
	int height = srcImg.rows;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstData[y * width + x] = (abs(myKernelConv3x3(srcData, kernelX, x, y, width, height)) +
				abs(myKernelConv3x3(srcData, kernelY, x, y, width, height))) / 2;
			// �� ���� ����� ���밪 �� ���·� ���� ��� ����
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
	double minVal, maxVal; // �ִ밪 �ּҰ��� ��Ÿ���� ���� ����
	Point minLoc, maxLoc; // ��ġ�� ��Ÿ���� ������ ���� ����
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc); // minMaxLoc (����� �ϴ� ���, �ּҰ�, �ִ밪, �ּҰ� ��ġ, �ִ밪 ��ġ)�� ��� �ִ� �ּҰ��� ��´�. -> ���� ����ȭ �Ϸ���
	normalize(magImg, magImg, 0, 1, NORM_MINMAX); // ������ ���� ���� ����ȭ �Ѵ�.
	
	Mat maskImg = Mat::ones(magImg.size(), CV_32F); // maskimg�� ������ ���׶�� 1���� ����� ���ļ� ������ �̹����� ���������� �����ִ� ����

	// line�� circle ���͸� ���� ���ļ� ���п��� �����ϰ� ���ڿ������� ���� �ٹ��̸� �ڿ������� ���̵��� �� ��� ����Ѵ�.
	// line(input output array, pt1, pt2, scalar(b,g,r), thickness(����))�� ��Ÿ����. pt1, pt2�� ������ �� �� ��ǥ�� �ǹ��Ѵ�.
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

	// spatial domain���� sobel filter ����
	Mat img2 = imread("img2.jpg", 0); // img�� �״�� �б�
	Mat sp_img2 = mySobelFilter(img2);
	imshow("spatial domain", sp_img2);

	// frequency domain���� sobel filter ����
	Mat fre_img2 = doFDF(img2);
	imshow("frequency domain", fre_img2);

	// �׳� high pass filter
	Mat hig_img2 = doHPF(img2);
	imshow("high pass filter domain", hig_img2);

	waitKey(0);
	destroyAllWindows();
}

// 3�� img3���� ��Ÿ���� flickering ������ frequency domain filtering�� ���� �����ض�
Mat doRDF(Mat srcImg) {

	//DFT
	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	//BPF
	double minVal, maxVal; // �ִ밪 �ּҰ��� ��Ÿ���� ���� ����
	Point minLoc, maxLoc; // ��ġ�� ��Ÿ���� ������ ���� ����
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc); // minMaxLoc (����� �ϴ� ���, �ּҰ�, �ִ밪, �ּҰ� ��ġ, �ִ밪 ��ġ)�� ��� �ִ� �ּҰ��� ��´�. -> ���� ����ȭ �Ϸ���
	normalize(magImg, magImg, 0, 1, NORM_MINMAX); // ������ ���� ���� ����ȭ �Ѵ�.

	Mat maskImg = Mat::ones(magImg.size(), CV_32F); // maskimg�� ������ ���׶�� 1���� ����� ���ļ� ������ �̹����� ���������� �����ִ� ����

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
	Mat ex3_img = imread("img3.jpg", 0); // �� �״�� �б�
	Mat rem_img = doRDF(ex3_img);
	imshow("remove flickering frequency domain", rem_img);

	waitKey(0);
	destroyAllWindows();
}

int main() {
	ex3();
}