#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "week3_.h"

using namespace cv;
using namespace std;

//9x9 Gaussian filter�� �����ϰ� ����� Ȯ���� ��
//9x9 Gaussian filter�� �������� �� ������׷��� ��� ���ϴ��� Ȯ���� ��
 //1��
int main() {
	Mat img = imread("gear.jpg", 0);
	Mat GaussianImg = myGaussianFilter(img);

	imshow("Original img", img);
	imshow("Gaussian image", GaussianImg);

	Mat hist_img = GetHistogram(img);
	Mat hist_GaussianImg = GetHistogram(GaussianImg);
	imshow("Histogram Original img", hist_img);
	imshow("Histogram Gaussian image", hist_GaussianImg);

	waitKey(0);
	destroyAllWindows();
}

//2��
//���� Salt and peppter noise�� �ְ� ������ 9x9 Gaussian filter�� �����غ� ��
int main() {
	Mat src_img = imread("gear.jpg", 0);
	Mat dst_img = salt_and_pepper_noise(src_img);

	imshow("salted image", dst_img);
	imshow("salted histogram", GetHistogram(dst_img));
	dst_img = myGaussianFilter(dst_img);
	imshow("gaussian image", dst_img);
	imshow("gaussian histogram", GetHistogram(dst_img));
	waitKey(0);
	destroyAllWindows();
}


//3��
//45���� 135���� �밢 edge�� �����ϴ� Sobel filter�� �����ϰ� ����� Ȯ���� ��
int main() {
	Mat img = imread("gear.jpg", 0);
	Mat SobelImg = mySobelFilter(img);

	imshow("Sobel image", SobelImg);
	waitKey(0);
	destroyAllWindows();

}

//4��
//�÷����� ���� Gaussain pyramid�� �����ϰ� ����� Ȯ���� ��
int main() {
	Mat img = imread("gear.jpg", 1);
	vector<Mat> GaussianPyramid = myGaussianPyramid(img);

	for (int i = 0; i < GaussianPyramid.size(); i++)
	{
		imshow("Gaussian pyramid", GaussianPyramid[i]);
		waitKey(0);
		destroyAllWindows();
	}
}

//5��
int main() {
	//�÷����� ���� Laplacian pyramid�� �����ϰ�
	//������ ������ ����� Ȯ���� ��
	Mat src_img = imread("gear.jpg", 1);
	Mat dst_img;
	vector<Mat> VecLap = myLaplacianPyramid(src_img);

	reverse(VecLap.begin(), VecLap.end()); //���� ������� ó���ϱ� ���� vector�� ������ �ݴ�� ����

	for (int i = 0; i < VecLap.size(); i++) {
		//vector�� ũ�⸸ŭ �ݺ�
		if (i == 0) {
			dst_img = VecLap[i];	//���� ���� ����
			//���� ���� ���� != �� �����̸� �ٷ� �ҷ���
		}
		else {
			resize(dst_img, dst_img, VecLap[i].size()); //���� ������ Ȯ��(up-sampling)
			dst_img = dst_img + VecLap[i] - 128;
			//�� ������ �ٽ� ���Ͽ� ū �������� ����
			//overflow ���������� ���ߴ� 128�� �ٽ� ����
		}

		string fname = "lap_pyr" + to_string(i) + ".png";
		imwrite(fname, dst_img);
		imshow("reverse Laplacian pyramid", dst_img);
		waitKey(0);
		destroyAllWindows();
	}
}