# include "week11.h";

// ���� 1��
// coin.png�� ���� ������ �˾Ƴ��� ���α׷� ����
void ex1() {
	Mat img = imread("coin.png", IMREAD_COLOR);

	// <set params>
	SimpleBlobDetector::Params params;
	params.minThreshold = 10;
	params.maxThreshold = 300;
	params.filterByArea = true;
	params.minArea =300;
	params.maxArea = 9000;
	params.filterByCircularity = true;
	params.minCircularity = 0.6;
	params.filterByConvexity = false;
	params.minConvexity = 0.9;
	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;

	// <set blob detector>
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	// < detect blobs>
	std::vector<KeyPoint> keypoints;
	detector->detect(img, keypoints);

	// < draw blobs>
	Mat result;
	drawKeypoints(img, keypoints, result, Scalar(255, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	imshow("keypoints", result);
	cout << "���� ����" <<  keypoints.size(); // ���� ����
	waitKey(0);
	destroyAllWindows();
}

//  OpenCV�� 1. corner detection�� 2. circle detection�� �̿��� �ﰢ��, �簢
// ��, ������, �������� ������ ���������� �о�� ���� �� �������� �˾Ƴ�
// �� ���α׷��� ����(���� 4���� �׸���, PPT ���� �̿��� ���� ������ ��)
void ex2(Mat img)
{
	if (img.empty()) {
		cout << "Empty image!\n";
		exit(-1);
	}

	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	Mat harr;
	cornerHarris(gray, harr, 4, 5, 0.05, BORDER_DEFAULT);
	normalize(harr, harr, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

	Mat harr_abs;
	convertScaleAbs(harr, harr_abs);

	int thresh = 125;
	Mat result = img.clone();
	int cnt = 0;
	for (int y = 0; y < harr.rows; y++) {
		for (int x = 0; x < harr.cols; x++) {
			if ((int)harr.at<float>(y, x) > thresh) {
				circle(result, Point(x, y), 7, Scalar(0, 0, 255), 0, 4, 0);
			}
		}
	}

	SimpleBlobDetector::Params params;
	params.minThreshold = 10;
	params.maxThreshold = 1000;
	params.filterByArea = true;
	params.maxArea = 500;
	params.filterByCircularity = true;
	params.minCircularity = 0.01;
	params.filterByConvexity = true;
	params.minConvexity = 0.4;
	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;

	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	std::vector<KeyPoint> keypoints;
	detector->detect(result, keypoints);

	Mat blob_Result;
	drawKeypoints(img, keypoints, blob_Result, Scalar(255, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	if (keypoints.size() == 3) {
		cout << "triangle" << endl;
	}
	else if (keypoints.size() == 4) {
		cout << "square" << endl;
	}
	else if (keypoints.size() == 5) {
		cout << "pentagon" << endl;
	}
	else if (keypoints.size() == 6) {
		cout << "hexagon!" << endl;
	}

	imshow("Target image", result);
	imshow("Target blob Result", blob_Result);

	waitKey(0);
	destroyAllWindows();
}

void ex3()
{
	Mat img = imread("church.jpg", 1);
	cvFeatureSIFT(img);
	Mat dst = warpPers(img);
	cvFeatureSIFT(dst);
}

int main() {
	ex3();
}

 // �Ʒ��� ex2�������� main()
//int main() {
// 
//	Mat triangle = imread("triangle.png", IMREAD_COLOR);
//	Mat square = imread("square.png", IMREAD_COLOR);
//	Mat pentagon = imread("pentagon.png", IMREAD_COLOR);
//	Mat hexagon = imread("hexagon.png", IMREAD_COLOR);
//
//	ex2(hexagon); // ���ʴ�� �ٲ㼭 ����
//}