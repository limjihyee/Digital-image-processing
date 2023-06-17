#include <iostream>
#include "opencv2/core/core.hpp" // Mat class�� ���� data structure �� ��� ��ƾ�� �����ϴ� ���
#include "opencv2/highgui/highgui.hpp" // GUI�� ���õ� ��Ҹ� �����ϴ� ���(imshow ��)
#include "opencv2/imgproc/imgproc.hpp" // ���� �̹��� ó�� �Լ��� �����ϴ� ���
using namespace cv;
using namespace std;

char e[30];


//�̹����� rgb �� ���
void SpreadSalts(Mat img, int num) {
	for (int n = 0; n < num; n++) {

		//img.cols�� �̹����� �� ������ ����
		int x = rand() % img.cols;	//x�� �̹����� ���� �ش��ϴ� ���� ���Ұ��� �����
		//img.rows�� �̹����� ���� ������ ����
		int y = rand() % img.rows;	//y�� �̹����� ���̿� �ش��ϴ� ���� ���Ұ��� �����

		if (strcmp(e, "blue") == 0) { // blue�� �� ���  // if ���ǹ��� ���ڿ� �ۼ� �� if( strcmp(�����̸�, "Ư�� ���ڿ�")==0 ) {���๮;}
			img.at<Vec3b>(y, x)[0] = 255;	//Blue ä�� ����
			img.at<Vec3b>(y, x)[1] = 0;		//Green ä�� ����
			img.at<Vec3b>(y, x)[2] = 0;		//Red ä�� ����
		}
		else if (strcmp(e, "green") == 0) { // green�� �� ���
			img.at<Vec3b>(y, x)[0] = 0;		//Blue ä�� ����
			img.at<Vec3b>(y, x)[1] = 255;	//Green ä�� ����
			img.at<Vec3b>(y, x)[2] = 0;		//Red ä�� ����
		}
		else if (strcmp(e, "red") == 0) { //red�� �� ���
			img.at<Vec3b>(y, x)[0] = 0;		//Blue ä�� ����
			img.at<Vec3b>(y, x)[1] = 0;		//Green ä�� ����
			img.at<Vec3b>(y, x)[2] = 255;	//Red ä�� ����
		}
	}
}

void count_dots(Mat img) {
	int blue_dots = 0, green_dots = 0, red_dots = 0;

	for (int x = 0; x < img.cols; x++) {
		for (int y = 0; y < img.rows; y++) {
			//����Ʈ���� ���ҹ迭�� ������ ������ �ش�Ǵ� R �Ǵ� G �Ǵ� B�� ����Ʈ������ ����
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
	Mat src_img = imread("img1.jpg", -1); // �̹��� ������ �״�� ����
	cout << "��� ���� ���� ���� �ۼ��Ͻÿ�(���� : red, blue, green)";
	cin >> e;

	SpreadSalts(src_img, 5000);
	count_dots(src_img);

	imshow("Test window", src_img); //�̹��� ��� **
	waitKey(0); //Ű �Է� ���(0: Ű�� �Էµ� ������ ���α׷� ���� // �����ð� ��⸦ ���ϸ� 0��� �ٸ� ���� �Է�
	destroyWindow("Test window"); // �̹��� ���â ����

}