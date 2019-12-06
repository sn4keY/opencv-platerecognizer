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

Mat Morphology(Mat grayscaleImg) {
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

	Mat topHat;
	morphologyEx(grayscaleImg, topHat, MORPH_TOPHAT, kernel);
	imwrite(FOLDER + "03tophat.jpg", topHat);

	Mat blackHat;
	morphologyEx(grayscaleImg, blackHat, MORPH_BLACKHAT, kernel);
	imwrite(FOLDER + "04blackhat.jpg", blackHat);

	Mat afterMorph;
	add(grayscaleImg, topHat, afterMorph);
	subtract(afterMorph, blackHat, afterMorph);
	return afterMorph;
}

int main()
{
	Mat defaultImg = imread(FOLDER + "01default.jpg");

	Mat grayscaleImg = GetGrayscaleImg(defaultImg);
	imshow("Grayscale image", grayscaleImg);
	waitKey();
	imwrite(FOLDER + "02grayscale.jpg", grayscaleImg);

	Mat afterMorph = Morphology(grayscaleImg);
	imshow("Grayscale + topHat - blackHat", afterMorph);
	waitKey();
	imwrite(FOLDER + "05aftermorph.jpg", afterMorph);

	return 0;
}