#include <iostream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
using namespace cv;
using namespace std;

char e[30];


//이미지에 rgb 점 찍기
void SpreadSalts(Mat img, int num) {
	for (int n = 0; n < num; n++) {

		//img.cols는 이미지의 폭 정보를 저장
		int x = rand() % img.cols;	//x에 이미지의 폭에 해당하는 랜덤 원소값이 저장됨
		//img.rows는 이미지의 높이 정보를 저장
		int y = rand() % img.rows;	//y에 이미지의 높이에 해당하는 랜덤 원소값이 저장됨

		if (strcmp(e, "blue") == 0) { // blue에 점 찍기  // if 조건문에 문자열 작성 시 if( strcmp(변수이름, "특정 문자열")==0 ) {실행문;}
			img.at<Vec3b>(y, x)[0] = 255;	//Blue 채널 접근
			img.at<Vec3b>(y, x)[1] = 0;		//Green 채널 접근
			img.at<Vec3b>(y, x)[2] = 0;		//Red 채널 접근
		}
		else if (strcmp(e, "green") == 0) { // green에 점 찍기
			img.at<Vec3b>(y, x)[0] = 0;		//Blue 채널 접근
			img.at<Vec3b>(y, x)[1] = 255;	//Green 채널 접근
			img.at<Vec3b>(y, x)[2] = 0;		//Red 채널 접근
		}
		else if (strcmp(e, "red") == 0) { //red에 점 찍기
			img.at<Vec3b>(y, x)[0] = 0;		//Blue 채널 접근
			img.at<Vec3b>(y, x)[1] = 0;		//Green 채널 접근
			img.at<Vec3b>(y, x)[2] = 255;	//Red 채널 접근
		}
	}
}

void count_dots(Mat img) {
	int blue_dots = 0, green_dots = 0, red_dots = 0;

	for (int x = 0; x < img.cols; x++) {
		for (int y = 0; y < img.rows; y++) {
			//포인트에서 원소배열이 다음과 같으면 해당되는 R 또는 G 또는 B의 포인트개수를 증가
			if (img.at<Vec3b>(y, x) == Vec3b(255, 0, 0)) { blue_dots++; }
			else if (img.at<Vec3b>(y, x) == Vec3b(0, 255, 0)) { green_dots++; }
			else if (img.at<Vec3b>(y, x) == Vec3b(0, 0, 255)) { red_dots++; }
		}
	}

	cout << "Blue dots: " << blue_dots << endl
		<< "Green dots: " << green_dots << endl
		<< "Red dots: " << red_dots << endl;
}

int main() {
	Mat src_img = imread("img1.jpg", -1); // 이미지 영상을 그대로 읽음
	cout << "찍고 싶은 색의 점을 작성하시오(예시 : red, blue, green)";
	cin >> e;

	SpreadSalts(src_img, 5000);
	count_dots(src_img);

	imshow("Test window", src_img); //이미지 출력 **
	waitKey(0); //키 입력 대기(0: 키가 입력될 때까지 프로그램 멈춤 // 일정시간 대기를 원하면 0대신 다른 숫자 입력
	destroyWindow("Test window"); // 이미지 출력창 종료

}