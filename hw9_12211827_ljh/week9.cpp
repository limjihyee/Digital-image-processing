#include <iostream>
#include <cmath>
#include <iomanip>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <string.h>

using namespace std;
using namespace cv;

// opencv�� �̿��� ���� (mean shift clustering)
void exCvMeanShift() {
	Mat img = imread("fruits.png");
	if (img.empty()) exit(-1);
	cout << "-----exCvMeanShift() -----" << endl;

	resize(img, img, Size(256, 256), 0, 0, CV_INTER_AREA);
	imshow("Src", img);
	imwrite("exCvMeanShift_src.jpg", img);

	pyrMeanShiftFiltering(img, img, 8, 16);

	imshow("Dst", img);
	waitKey();
	destroyAllWindows();
	imwrite("exCvMeanShift_dst.jpg", img);
}

// low-level ���� (mean shift clustering)
//������ ���Ǽ��� ���� ����  point class ����

class Point5D {
	//Mean shift ������ ���� ���� ����Ʈ(�ȼ�) Ŭ����
public:
	float x, y, l, u, v; // ����Ʈ�� ��ǥ�� LUV ��

	Point5D();
	~Point5D();

	void accumPt(Point5D); // ����Ʈ ����
	void copyPt(Point5D); // ����Ʈ ����
	float getColorDist(Point5D); // ���� �Ÿ� ���
	float getSpatialDist(Point5D); //��ǥ �Ÿ� ���
	void scalePt(float);// ����Ʈ �����ϸ� �Լ�(��տ�)
	void setPt(float, float, float, float, float); // ����Ʈ�� �����Լ�
	void printPt();
};

Point5D::Point5D()
{
}

Point5D::~Point5D()
{
}

void Point5D::accumPt(Point5D Pt) {
	x += Pt.x;
	y += Pt.y;
	l += Pt.l;
	u += Pt.u;
	v += Pt.v;
}

void Point5D::copyPt(Point5D Pt) {
	x = Pt.x;
	y = Pt.y;
	l = Pt.l;
	u = Pt.u;
	v = Pt.v;
}

float Point5D::getColorDist(Point5D Pt) {
	return sqrt(pow(l - Pt.l, 2) + pow(u - Pt.u, 2) + pow(v - Pt.v, 2));
}

float Point5D::getSpatialDist(Point5D Pt) {
	return sqrt(pow(x - Pt.x, 2) + pow(y - Pt.y, 2));
}

void Point5D::scalePt(float scale) {
	x *= scale;
	y *= scale;
	l *= scale;
	u *= scale;
	v *= scale;
}

void Point5D::setPt(float px, float py, float pl, float pa, float pb) {
	x = px;
	y = py;
	l = pl;
	u = pa;
	v = pb;
}

void Point5D::printPt() {
	cout << x << " " << y << " " << l << " " << u << " " << v << endl;
}

//Mean shift clustering
class MeanShift {
	/* Mean shift Ŭ���� */
public:
	float bw_spatial = 8; // Spatial bandwidth
	float bw_color = 16; // Color bandwidth
	float min_shift_color = 0.1; // �ּ� �÷���ȭ
	float min_shift_spatial = 0.1; // �ּ� ��ġ��ȭ
	int max_steps = 10; // �ִ� �ݺ�Ƚ��
	vector<Mat> img_split; //ä�κ��� ���ҵǴ� Mat
	MeanShift(float, float, float, float, int); // Bandwidth ������ ���� ������
	void doFiltering(Mat&); // Mean shift filtering �Լ�
};
MeanShift::MeanShift(float bs, float bc, float msc, float mss, int ms) {
	/* ������ */
	bw_spatial = bs;
	bw_color = bc;
	max_steps = ms;
	min_shift_color = msc;
	min_shift_spatial = mss;
}

void MeanShift::doFiltering(Mat& Img) {
	int height = Img.rows;
	int width = Img.cols;
	split(Img, img_split);

	Point5D pt, pt_prev, pt_cur, pt_sum;

	int pad_left, pad_right, pad_top, pad_bottom;
	size_t n_pt, step;

	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			pad_left = (col - bw_spatial) > 0 ? (col - bw_spatial) : 0;
			pad_right = (col + bw_spatial) < width ? (col + bw_spatial) : width;
			pad_top = (row - bw_spatial) > 0 ? (row - bw_spatial) : 0;
			pad_bottom = (row + bw_spatial) < height ? (row + bw_spatial) : height;

			// <���� �ȼ� ����>
			pt_cur.setPt(row, col,
				(float)img_split[0].at<uchar>(row, col),
				(float)img_split[1].at<uchar>(row, col),
				(float)img_split[2].at<uchar>(row, col));

			// <�ֺ��ȼ� Ž��>
			step = 0;
			do {
				pt_prev.copyPt(pt_cur);
				pt_sum.setPt(0, 0, 0, 0, 0);
				n_pt = 0;
				for (int hx = pad_top; hx < pad_bottom; hx++) {
					for (int hy = pad_left; hy < pad_right; hy++) {
						pt.setPt(hx, hy,
							(float)img_split[0].at<uchar>(hx, hy),
							(float)img_split[1].at<uchar>(hx, hy),
							(float)img_split[2].at<uchar>(hx, hy));

						// <Color bandwidth �ȿ��� ����>
						if (pt.getColorDist(pt_cur) < bw_color) {
							pt_sum.accumPt(pt);
							n_pt++;
						}
					}
				}

				// <��������� ������� �����ȼ� ����>
				pt_sum.scalePt(1.0 / n_pt); // ������� ���
				pt_cur.copyPt(pt_sum);
				step++;
			} 
			while ((pt_cur.getColorDist(pt_prev) > min_shift_color) &&
				(pt_cur.getSpatialDist(pt_prev) > min_shift_spatial) &&
				(step < max_steps));
			//��ȭ�� �ּ� ������ ������ ������ �ݺ�
			//�ִ� �ݺ�Ƚ�� ���ǵ� ����

			// <��� �ȼ� ����>
			Img.at<Vec3b>(row, col) = Vec3b(pt_cur.l, pt_cur.u, pt_cur.v);
		}
	}
}

void exMyMeanShift() {
	Mat img = imread("fruits.png");
	if (img.empty()) exit(-1);
	cout << "-----exMyMeanShift() -----" << endl;

	resize(img, img, Size(256, 256), 0, 0, CV_INTER_AREA);
	imshow("Src", img);
	imwrite("exMyMeanShift_src.jpg", img);

	cvtColor(img, img, CV_RGB2Luv);

	MeanShift MSProc(8, 16, 0.1, 0.1, 10);
	MSProc.doFiltering(img);

	cvtColor(img, img, CV_Luv2RGB);

	imshow("Dst", img);
	waitKey();
	destroyAllWindows();
	imwrite("exMyMeanShift_dst.jpg", img);
}

void Grab(Mat img, int x1, int y1, int x2, int y2)
{
	Rect rect = Rect(Point(x1, y1), Point(x2, y2));

	Mat rect2, result, bg_model, fg_model;

	img.copyTo(rect2);
	rectangle(rect2, rect, Scalar(255, 0, 0), 5);

	grabCut(img, result, rect, bg_model, fg_model, 5, GC_INIT_WITH_RECT);

	compare(result, GC_PR_FGD, result, CMP_EQ);
	//GC_PR_FGDF: Crabcut class foreground �ȼ�
	//CMP_EQ: compare �ɼ�(equal)

	Mat mask(img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	img.copyTo(mask, result);

	imshow("rect", rect2);
	imshow("mask", mask);
	imshow("result", result);
	waitKey();
	destroyAllWindows();
}

void ex1() {
	exCvMeanShift(); // opencv �̿��� Mean-shift clustering;

	exMyMeanShift(); // Low-level�� Mean-shift clustering
}

void ex2() {
	////1�� ����
	//Mat img = imread("girl.png", 1);
	//Grab(img, 20, 20, 140, 140);

	//// 2�� �ܽ���
	//Mat img = imread("hamster.jpg", 1);
	//Grab(img, 300, 100, 700, 600);

	////3�� ����
	//Mat img = imread("girl2.png", 1);
	//Grab(img, 50, 10, 300, 589);

	////4�� �عٶ��
	//Mat img = imread("sunflower.jpg", 1);
	//Grab(img, 50, 20, 260, 360);

	////5�� ���
	//Mat img = imread("apple.jpg", 1);
	//Grab(img, 100, 200, 400, 400); 

	//6�� �罿
	Mat img = imread("animal.jpg", 1);
	Grab(img, 200, 10, 1000, 750);
}

int main() {
	ex2();
}