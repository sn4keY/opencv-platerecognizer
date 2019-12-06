#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

const string FOLDER = "D:\\Source\\OpenCVPlateRecognizer\\audi\\";

Mat GetGrayscaleImg(Mat defaultImg) {
	Mat hsv;
	cvtColor(defaultImg, hsv, COLOR_RGB2HSV);
	Mat hsvSplit[3];
	split(hsv, hsvSplit);
	return hsvSplit[2];
}

int main()
{
	Mat defaultImg = imread(FOLDER + "01default.jpg");

	Mat grayscaleImg = GetGrayscaleImg(defaultImg);
	imshow("Grayscale image", grayscaleImg);
	waitKey();
	imwrite(FOLDER + "02grayscale.jpg", grayscaleImg);

	return 0;
}