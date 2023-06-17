#include <iostream>
#include <iomanip>
#include <string.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/photo.hpp"
#include "opencv2/imgcodecs.hpp"
using namespace std;
using namespace cv;

Mat GetHistogram(Mat& src);

void readImagesAndTimes(vector<Mat>& images, vector<float>& times)
{
	int numImages = 4;
	static const float timesArray[] = { 1 / 30.0f, 0.25f, 2.5f, 15.0f };
	times.assign(timesArray, timesArray + numImages);
	static const char* filenames[] = { "img_0.033.jpg", "img_0.25.jpg", "img_2.5.jpg", "img_15.jpg" };
	for (int i = 0; i < numImages; i++) {
		Mat im = imread(filenames[i]);
		images.push_back(im);
		Mat hist_pic = GetHistogram(im);
		imshow(filenames[i], hist_pic);
		waitKey(0);
	}
	destroyAllWindows();
}

void readImagesAndTimes2(vector<Mat>& images, vector<float>& times)
{
	int numImages = 4;
	static const float timesArray[] = { 0.0002f, 0.001f, 0.01f, 0.1f };
	times.assign(timesArray, timesArray + numImages);
	static const char* filenames[] = { "0.0002.png", "0.001.png", "0.01.png", "0.1.png" };
	for (int i = 0; i < numImages; i++) {
		Mat im = imread(filenames[i]);
		images.push_back(im);
		Mat hist_image = GetHistogram(im);
		imshow(filenames[i], hist_image);
		waitKey(0);
	}
	destroyAllWindows();
}

//Histogram 분석
Mat GetHistogram(Mat& src)
{
	cvtColor(src, src, COLOR_RGB2GRAY);
	Mat histogram;
	const int* channel_numbers = { 0 };
	float channel_range[] = { 0.0, 255.0 };
	const float* channel_ranges = channel_range;
	int number_bins = 255;
	calcHist(&src, 1, channel_numbers, Mat(), histogram, 1, &number_bins, &channel_ranges);
	int hist_w = 500;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / number_bins);

	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0)); normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	for (int i = 1; i < number_bins; i++) {
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(histogram.at<float>(i - 1))), Point(bin_w * (i), hist_h - cvRound(histogram.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
	}
	return histImage;
}

void ex1()
{
	// <영상, 노출 시간 불러오기>
	cout << "Reading images and exposure times ..." << endl;
	vector<Mat> images;
	vector<float> times;
	readImagesAndTimes(images, times);
	cout << "finished" << endl;

	// <영상 정렬>
	cout << "Aligning images ..." << endl;
	Ptr<AlignMTB> alignMTB = createAlignMTB();
	alignMTB->process(images, images);

	// <Camera response function(CRF) 복원>
	cout << "Calculating Camera Response Function ..." << endl;
	Mat responseDebevec;
	Ptr<CalibrateDebevec> calibrateDebevec = createCalibrateDebevec();
	calibrateDebevec->process(images, responseDebevec, times);
	cout << "----- CRF -----" << endl;
	cout << responseDebevec << endl;

	// <24 bit 표현 범위로 이미지 병합>
	cout << "Merging images into one HDR image ..." << endl;
	Mat hdrDebevec;
	Ptr<MergeDebevec> mergeDevevec = createMergeDebevec();
	mergeDevevec->process(images, hdrDebevec, times, responseDebevec);
	imwrite("hdrDevevec.jpg", hdrDebevec);
	cout << "saved hdrDebevec.hdf" << endl;

	// <Drago 톤맵>
	cout << "Tonemaping using Drago's method ...";
	Mat IdrDrago;
	Ptr<TonemapDrago> tonemapDrago = createTonemapDrago(1.0f, 0.7f, 0.85f);
	tonemapDrago->process(hdrDebevec, IdrDrago);
	IdrDrago = 3 * IdrDrago;
	IdrDrago = IdrDrago*255;
	imwrite("Idr-Drago.jpg", IdrDrago);
	cout << "saved Idr-Drago.jpg" << endl;
	Mat hist_Drago = GetHistogram(IdrDrago);
	imshow("Drago Histogram", hist_Drago);
	waitKey(0);

	// <Reinhard 톤맵>
	cout << "Tonemaping using Reinhard's method ...";
	Mat IdrReinhard;
	Ptr<TonemapReinhard> tonemapReinhard = createTonemapReinhard(1.5f, 0, 0, 0);
	tonemapReinhard->process(hdrDebevec, IdrReinhard);
	IdrReinhard = IdrReinhard * 255;
	imwrite("Idr-Reinhard.jpg", IdrReinhard);
	cout << "saved Idr-Reinhard.jpg" << endl;
	Mat hist_Reinhard = GetHistogram(IdrReinhard);
	imshow("Reinhard Histogram", hist_Reinhard);
	waitKey(0);

	// <Mantiuk 톤맵>
	cout << "Tonemaping using Mantiuk's method ...";
	Mat IdrMantiuk;
	Ptr<TonemapMantiuk> tonemapMantiuk = createTonemapMantiuk(2.2f, 0.85f, 1.2f);
	tonemapMantiuk->process(hdrDebevec, IdrMantiuk);
	IdrMantiuk = 3 * IdrMantiuk;
	IdrMantiuk = IdrMantiuk * 255;
	imwrite("Idr-Mantiuk.jpg", IdrMantiuk);
	cout << "saved Idr-Mantiuk.jpg" << endl;
	Mat hist_Mantiuk = GetHistogram(IdrMantiuk);
	imshow("Mantiuk Histogram", hist_Mantiuk);
	waitKey(0);
}

void ex2()
{
	// <영상, 노출 시간 불러오기>
	cout << "Reading images and exposure times ..." << endl;
	vector<Mat> images;
	vector<float> times;
	readImagesAndTimes2(images, times);
	cout << "finished" << endl;

	// <영상 정렬>
	cout << "Aligning images ..." << endl;
	Ptr<AlignMTB> alignMTB = createAlignMTB();
	alignMTB->process(images, images);

	// <Camera response function(CRF) 복원>
	cout << "Calculating Camera Response Function ..." << endl;
	Mat responseDebevec;
	Ptr<CalibrateDebevec> calibrateDebevec = createCalibrateDebevec();
	calibrateDebevec->process(images, responseDebevec, times);
	cout << "----- CRF -----" << endl;
	cout << responseDebevec << endl;

	// <24 bit 표현 범위로 이미지 병합>
	cout << "Merging images into one HDR image ..." << endl;
	Mat hdrDebevec;
	Ptr<MergeDebevec> mergeDevevec = createMergeDebevec();
	mergeDevevec->process(images, hdrDebevec, times, responseDebevec);
	cout << "saved hdrDebevec.hdf" << endl;

	// <Drago 톤맵>
	cout << "Tonemaping using Drago's method ...";
	Mat IdrDrago;
	Ptr<TonemapDrago> tonemapDrago = createTonemapDrago(1.0f, 0.7f, 0.85f);
	tonemapDrago->process(hdrDebevec, IdrDrago);
	IdrDrago = 3 * IdrDrago;
	IdrDrago = IdrDrago * 255;
	imwrite("Idr-Drago.jpg", IdrDrago);
	cout << "saved Idr-Drago.jpg" << endl;
	Mat hist_Drago = GetHistogram(IdrDrago);
	imshow("Drago Histogram", hist_Drago);
	waitKey(0);

	// <Reinhard 톤맵>
	cout << "Tonemaping using Reinhard's method ...";
	Mat IdrReinhard;
	Ptr<TonemapReinhard> tonemapReinhard = createTonemapReinhard(1.5f, 0, 0, 0);
	tonemapReinhard->process(hdrDebevec, IdrReinhard);
	IdrReinhard = IdrReinhard * 255;
	imwrite("Idr-Reinhard.jpg", IdrReinhard);
	cout << "saved Idr-Reinhard.jpg" << endl;
	Mat hist_Reinhard = GetHistogram(IdrReinhard);
	imshow("Reinhard Histogram", hist_Reinhard);
	waitKey(0);

	// <Mantiuk 톤맵>
	cout << "Tonemaping using Mantiuk's method ...";
	Mat IdrMantiuk;
	Ptr<TonemapMantiuk> tonemapMantiuk = createTonemapMantiuk(2.2f, 0.85f, 1.2f);
	tonemapMantiuk->process(hdrDebevec, IdrMantiuk);
	IdrMantiuk = 3 * IdrMantiuk;
	IdrMantiuk = IdrMantiuk * 255;
	imwrite("Idr-Mantiuk.jpg", IdrMantiuk);
	cout << "saved Idr-Mantiuk.jpg" << endl;
	Mat hist_Mantiuk = GetHistogram(IdrMantiuk);
	imshow("Mantiuk Histogram", hist_Mantiuk);
	waitKey(0);
}

int main()
{
	//ex1();
	ex2();
	return 0;
}