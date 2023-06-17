#include <iostream>
#include "opencv2/core/core.hpp" // Mat class�� ���� data structure �� ��� ��ƾ�� �����ϴ� ���
#include "opencv2/highgui/highgui.hpp" // GUI�� ���õ� ��Ҹ� �����ϴ� ���(imshow ��)
#include "opencv2/imgproc/imgproc.hpp" // ���� �̹��� ó�� �Լ��� �����ϴ� ���
using namespace cv;
using namespace std;

// img2.jpg�� �Ʒ��� ������ ��ο� ���� ���α׷� �ۼ��ϱ� (��� ����� for ������ ����)

//Histogram �м�
Mat GetHistogram(Mat& src)
{
	Mat histogram;
	const int* channel_numbers = { 0 };
	float channel_range[] = { 0.0, 255.0 };
	const float* channel_ranges = channel_range;
	int number_bins = 255;

	//������׷� ���
	calcHist(&src, 1, channel_numbers, Mat(), histogram, 1, &number_bins, &channel_ranges);

	//������׷� plot
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / number_bins);

	//����ȭ
	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	//���� ���� �մ� ���� �׸��� ������� plot
	for (int i = 1; i < number_bins; i++) {
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(histogram.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(histogram.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	return histImage;
}
int dark_down(Mat img, int y, int x); //prototype

int main() {
	Mat src_img = imread("img2.jpg", 0); //�̹��� ���� ������� ó��

	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			// �ȼ� ���� ������ �Ͽ� ��Ӱ� �����
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
	/* ���� �������� 0, 0 ���� ���� ��Ӵ�.
	���� ���ϴ� �κ��� �ȼ� ���� ������ �ϰ� ��谪�� ������ �߰� �� �� y/2�� �Ͽ� ��⸦ ����
	*/

	if (e < 0)
		return 0; // ���� 0���� �۰� �Ǵ� ���� 0���� �Ͽ� �ȼ� ��ȭ�� ������ �Ͽ���.
	else
		return e;
}