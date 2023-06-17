#include <iostream>
#include <cmath>
#include <iomanip>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

//// kernel convolution review ���� �Ҵ� �� ������ ��� Ŀ�� �������
double gaussian2D(float c, float r, double sigma) {
	return exp(-(pow(c, 2) + pow(r, 2)) / (2 * pow(sigma, 2)))
		/ (2 * CV_PI * pow(sigma, 2));
}

float distance(int x, int y, int i, int j) {
	return float(sqrt(pow(x - i, 2) + pow(y - j, 2)));
}

double gaussian(float x, double sigma) {
	return exp(-(pow(x, 2)) / (2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));
}
//
//void myGaussian(const Mat& src_img, Mat& dst_img, Size size) {
//	// <Ŀ�� ����>
//	Mat kn = Mat::zeros(size, CV_32FC1);
//	double sigma = 1.0;
//	float* kn_data = (float*)kn.data;
//	for(int c = 0; c < kn.cols; c++) {
//		for (int r = 0; r < kn.rows; r++) {
//			kn_data[r*kn.cols + c] =
//				(float)gaussian2D((float)(c-kn.cols / 2),
//				(float)(r - kn.rows / 2), sigma);
//		}
//	}
//
//	// <Ŀ�� ������� ����>
//	myKernelConv(src_img, dst_img, kn);
//}
//
//void myKernelConv(const Mat& src_img, Mat& dst_img, const Mat& kn) {
//	dst_img = Mat::zeros(src_img.size(), CV_8UC1);
//
//	int wd = src_img.cols; int hg = src_img.rows;
//	int kwd = kn.cols; int khg = kn.rows;
//	int rad_w = kwd / 2; int rad_h = khg / 2;
//
//	float* kn_data = (float*)kn.data;
//	uchar* src_data = (uchar*)src_img.data;
//	uchar* dst_data = (uchar*)dst_img.data;
//
//	float wei, tmp, sum;
//
//	// <�ȼ� �ε��� (�����ڸ� ����) >
//	for (int c = rad_w + 1; c < wd - rad_w; c++) {
//		for (int r = rad_h + 1; r < hg - rad_h; r++) {
//			tmp = 0.f;
//			sum = 0.f;
//			//<Ŀ�� �ε���>
//			for (int kc = -rad_w; kc <= rad_w; kc++) {
//				for (int kr = -rad_h; kr <= rad_h; kr++) {
//					wei + (float)kn_data[(kr + rad_h) * kwd + (kc + rad_w)];
//					tmp += wei * (float)src_data[(r + kr) * wd + (c + kr)];
//					sum += wei;
//				}
//			}
//			if (sum != 0.f) tmp = abs(tmp) / sum; // ����ȭ �� overflow ����
//			else tmp = abs(tmp);
//
//			if (tmp > 255.f) tmp = 255.f; // overflow ����
//
//			dst_data[r * wd + c] = (uchar)tmp;
//		}
//	}
//}

//// ������� ���� ���� ����

// Median Filter
// median filter�� �̿��Ͽ� salt and pepper noise �����ϱ�
// �߰��� �����Ͽ� �װͺ��� �ʹ� ũ�ų� ���� �� ����

// 5x5
void myMedian(const Mat& src_img, Mat& dst_img, const Size& kn_size) {
	dst_img = Mat::zeros(src_img.size(), CV_8UC1);

	int wd = src_img.cols; int hg = src_img.rows;
	int kwd = kn_size.width; int khg = kn_size.height;
	int rad_w = kwd / 2; int rad_h = khg / 2;

	uchar* src_data = (uchar*)src_img.data;
	uchar* dst_data = (uchar*)dst_img.data;

	float* table = new float[kwd * khg](); // Ŀ�� ���̺� ���� �Ҵ�

	// <�ȼ� �ε��� (�����ڸ� ����) >
	for (int c = rad_w + 1; c < wd - rad_w; c++) {
		for (int r = rad_h + 1; r < hg - rad_h; r++) {

			// <Ŀ�� �ε���>
			for (int kc = -rad_w; kc <= rad_w; kc++) {
				for (int kr = -rad_h; kr <= rad_h; kr++) {

					int sw1 = (r + kr) * wd + (c + kc);
					int sw2 = (kr + rad_h) * kwd + (kc + rad_w);

					table[sw2] = (float)src_data[sw1];
				}
			}
			for (int i = 0; i < kwd * khg - 1; i++) {
				for (int j = i + 1; j < kwd * khg - 1; j++) {
					if (table[i] > table[j]) {
						float tmp = table[i];
						table[i] = table[j];
						table[j] = tmp;
					}
				}
			}
			int mid = kwd * khg / 2 + 1;

			dst_data[r * wd + c] = table[mid];
		}
	}
	delete table; // ���� �Ҵ� ����
}

void doMedianEx() {
	cout << "--- doMedianEX() --- \n" << endl;

	// <�Է�>
	Mat src_img = imread("salt_pepper.png", 0);
	if (!src_img.data) printf("No image data \n");

	// <Median ���͸� ����>
	Mat dst_img;
	myMedian(src_img, dst_img, Size(3, 3));

	// <���>
	Mat result_img;
	hconcat(src_img, dst_img, result_img);
	imshow("doMedianEX()", result_img);
	waitKey(0);
}

void bilateral(const Mat& src_img, Mat& dst_img,
	int c, int r, int diameter, double sig_r, double sig_s);

// Bilateral FIlter
void myBilateral(const Mat& src_img, Mat& dst_img,
	int diameter, double sig_r, double sig_s) {
	dst_img = Mat::zeros(src_img.size(), CV_8UC1);

	Mat guide_img = Mat::zeros(src_img.size(), CV_64F);
	int wh = src_img.cols; int hg = src_img.rows;
	int radius = diameter / 2;

	// <�ȼ� �ε��� (�����ڸ� ����)>
	for (int c = radius + 1; c < hg - radius; c++) {
		for (int r = radius + 1; r < wh - radius; r++) {
			bilateral(src_img, guide_img, c, r, diameter, sig_r, sig_s);
			//ȭ�Һ� bilateral ��� ����
		}
	}
	guide_img.convertTo(dst_img, CV_8UC1); // Mat type ��ȯ
}

void doBilateralEX() {
	cout << "--- doBiladeralEX() --- \n" << endl;

	// <�Է�>
	Mat src_img = imread("rock.png", 0);
	Mat dst_img;
	if (!src_img.data) printf("No image data \n");

	// < Bilateral ���͸� ����>
	myBilateral(src_img, dst_img, 5, 25.0, 50.0);

	// <���>
	Mat result_img;
	hconcat(src_img, dst_img, result_img);
	imshow("doBilateralEx()", result_img);
	waitKey(0);
}

void bilateral(const Mat& src_img, Mat& dst_img,
	int c, int r, int diameter, double sig_r, double sig_s) {

	int radius = diameter / 2;

	double gr, gs, wei;
	double tmp = 0;
	double sum = 0;

	// Ŀ�� �ε���
	for (int kc = -radius; kc <= radius; kc++) {
		for (int kr = -radius; kr <= radius; kr++) {
			//rage calc
			gr = gaussian2D((float)src_img.at<uchar>(c + kc, r + kr) - (float)src_img.at<uchar>(c, r),
				(float)src_img.at<uchar>(c + kc, r + kr) - (float)src_img.at<uchar>(c, r), sig_r);
			//spatial calc
			gs = gaussian2D(distance(c, r, c + kc, r + kr), distance(c, r, c + kc, r + kr), sig_s);
			wei = gr * gs;
			tmp += src_img.at<uchar>(c + kc, r + kr) * wei;
			sum += wei;
		}
	}
	dst_img.at<double>(c, r) = tmp / sum; //����ȭ
}


// ���� �� �� �ּ� ó���ϰ� �ϳ��� �� ��
//hw1�� main()
//int main() {
//	
//	// hw1 5x5 median filter
//	Mat src_img = imread("salt_pepper2.png", 0);
//	Mat dst_img, final_img;
//	myMedian(src_img, dst_img, Size(5, 5)); // 5x5��
//
//	hconcat(src_img, dst_img, final_img);
//	imshow("5x5 median", final_img);
// 
// // 3x3 median filter
//	doMedianEx();
//	waitKey(0);
//	destroyAllWindows();
//	}

//hw2�� main()
//void myBilateral(const Mat& src_img, Mat& dst_img,
//int diameter, double sig_r, double sig_s)
//int main() {
//	Mat src_img = imread("rock.png", 0);
//	Mat dst_img1, dst_img2, dst_img3, dst_img4, dst_img5, dst_img6, dst_img7, dst_img8, dst_img9;
//	if (!src_img.data)printf("no image data\n");
//
//	//Bilateral ���͸� ����
//	myBilateral(src_img, dst_img1, 5, 0.2, 2);
//	myBilateral(src_img, dst_img2, 5, 2000, 2);
//	myBilateral(src_img, dst_img3, 5, 1000000000, 2);
//	myBilateral(src_img, dst_img4, 5, 0.2, 5000);
//	myBilateral(src_img, dst_img5, 5, 2000, 5000);
//	myBilateral(src_img, dst_img6, 5, 1000000000, 5000);
//	myBilateral(src_img, dst_img7, 5, 0.2, 100000000);
//	myBilateral(src_img, dst_img8, 5, 2000, 100000000);
//	myBilateral(src_img, dst_img9, 5, 1000000000, 100000000);
//
//	Mat result1, result2, result3;
//	hconcat(dst_img1, dst_img2, result1);
//	hconcat(result1, dst_img3, result1);
//	hconcat(dst_img4, dst_img5, result2);
//	hconcat(result2, dst_img6, result2);
//	hconcat(dst_img7, dst_img8, result3);
//	hconcat(result3, dst_img9, result3);
//
//	Mat result;
//	vconcat(result1, result2, result);
//	vconcat(result, result3, result);
//
//	imshow("ex2_bilateral()", result);
//
//	waitKey(0);
//}

//hw3�� main()
int main() {
	// hw3() Canny edge detection
	Mat src_img = imread("edge_test.jpg", 0);

	//Canny(InputArray image, OutputArray edges, double �ּ�threshold, double �ִ�threshold) threshold = [0~255]
	Mat dst_img;
	int min_threshold = 240;
	int max_threshold = 255;

	clock_t start_c, end_c;
	float res;

	start_c = clock();
	Canny(src_img, dst_img, min_threshold, max_threshold);
	end_c = clock();
	res = (float)(end_c - start_c);


	cout << "�ּ� ��谪 " << min_threshold << "�ִ� ��谪 " << max_threshold << endl
		<< "ó���ð� " << res << endl;
	Mat result_img;
	hconcat(src_img, dst_img, result_img);
	imshow("hw2()", result_img);
	waitKey(0);
	destroyAllWindows();
}