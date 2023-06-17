#include <iostream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
using namespace cv;
using namespace std;

int main()
{
	Mat src3 = imread("img3.jpg", 1);	//화성배경의 우주선
	Mat src4 = imread("img4.jpg", 1);	//원형의 검은 배경
	Mat src5 = imread("img5.jpg", 1);	//SPACEX 로고

	resize(src3, src3, Size(src4.cols, src4.rows));
	// 화성과 원형 검은 배경의 사이즈 맞춤

	Mat final_img;
	subtract(src3, src4, final_img);

	Mat logo = final_img(Rect(500, 550, src5.cols, src5.rows));

	// mask를 만들기 위해서 logo 이미지를 gray scales로 바꾸고 binary image로 전환
	// mask는 logo부분이 흰색(255) 바탕은 검정색(0)
	// mask_inv는 logo부분이 검은색 바탕이 흰색이다.

	//1번(binarization 강노 부분)
	Mat gray_img, binary_img1, binary_img2, img1_fg, img2_bg, logo_final;
	cvtColor(src5, gray_img, CV_BGR2GRAY); // 이진화를 위해서 color에서 grayscale 영상으로 변환

	
	threshold(gray_img, binary_img1, 210, 255, THRESH_BINARY); // 임계값 지정 이진화, thresh를 127 그리고 maximum이 255이다. 
															   // 흰색을 제외한 모든 색을 검정색으로 변환 (마스크 결과 값)
	
	
	//binary_img2는 binary_img의 색을 역변환한 영상을 저장
	bitwise_not(binary_img1, binary_img2); //binary_img2가 마스크 결과 iverse 값
	
	bitwise_and(logo, logo, img1_fg, binary_img1);	//mask는 img1_fg 으로 logo와 logo를 and 연산 시키고 binary_img1 mask를 하여 
	// binary_img1에 저장하였다. 이러면 로고의 글자를 검정으로 하고 배경에는 final의 일부분이 저장된다.
	bitwise_and(src5, src5, img2_bg, binary_img2);	//mask는 img2_bg
	
	add (img1_fg, img2_bg, logo_final);

	// image에서 로고가 들어갈 부분에 logo_final를 넣어주기
	final_img(Rect(500, 550, src5.cols, src5.rows)) = logo_final + 1;
	imshow("hw3", final_img);
	waitKey(0);
	// 위 두개의 img1_fg와 img2_bg를 결합


	return 0;
}