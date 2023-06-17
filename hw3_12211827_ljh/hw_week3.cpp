#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "week3_.h"

using namespace cv;
using namespace std;

//9x9 Gaussian filter를 구현하고 결과를 확인할 것
//9x9 Gaussian filter를 적용했을 때 히스토그램이 어떻게 변하는지 확인할 것
 //1번
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

//2번
//영상에 Salt and peppter noise를 주고 구현한 9x9 Gaussian filter를 적용해볼 것
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


//3번
//45도와 135도의 대각 edge를 검출하는 Sobel filter를 구현하고 결과를 확인할 것
int main() {
	Mat img = imread("gear.jpg", 0);
	Mat SobelImg = mySobelFilter(img);

	imshow("Sobel image", SobelImg);
	waitKey(0);
	destroyAllWindows();

}

//4번
//컬러영상에 대한 Gaussain pyramid를 구축하고 결과를 확인할 것
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

//5번
int main() {
	//컬러영상에 대한 Laplacian pyramid를 구축하고
	//복원을 수행한 결과를 확인할 것
	Mat src_img = imread("gear.jpg", 1);
	Mat dst_img;
	vector<Mat> VecLap = myLaplacianPyramid(src_img);

	reverse(VecLap.begin(), VecLap.end()); //작은 영상부터 처리하기 위해 vector의 순서를 반대로 해줌

	for (int i = 0; i < VecLap.size(); i++) {
		//vector의 크기만큼 반복
		if (i == 0) {
			dst_img = VecLap[i];	//가장 작은 영상
			//가장 작은 영상 != 차 영상이면 바로 불러옴
		}
		else {
			resize(dst_img, dst_img, VecLap[i].size()); //작은 영상을 확대(up-sampling)
			dst_img = dst_img + VecLap[i] - 128;
			//차 영상을 다시 더하여 큰 영상으로 복원
			//overflow 방지용으로 더했던 128을 다시 빼줌
		}

		string fname = "lap_pyr" + to_string(i) + ".png";
		imwrite(fname, dst_img);
		imshow("reverse Laplacian pyramid", dst_img);
		waitKey(0);
		destroyAllWindows();
	}
}