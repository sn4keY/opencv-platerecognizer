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

Mat GetGaussianBlur(Mat morph) {
	Mat blur;
	GaussianBlur(morph, blur, Size(5, 5), 0);
	return blur;
}

Mat GetAdaptiveThreshold(Mat blur) {
	Mat thresh;
	adaptiveThreshold(blur, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 19, 9);
	return thresh;
}

Mat GetCannyImg(Mat grayscaleImg) {
	int thresh = 100;
	Mat canny;
	Canny(grayscaleImg, canny, thresh, thresh * 2);
	return canny;
}

int GetNumberOfChildren(vector<Vec4i> hierarchy, int idx) {
	idx = hierarchy[idx][2];
	if (idx < 0)
		return 0;

	int count = 1;
	while (hierarchy[idx][0] > 0)
	{
		count++;
		idx = hierarchy[idx][0];
	}
	return count;
}

vector<vector<Point>> GetChildren(int idx, vector<Vec4i> hierarchy, vector<vector<Point>> contours) {
	vector<vector<Point>> contoursWithChildren;
	for (; idx >= 0; idx = hierarchy[idx][0]) {
		int numberOfChildren = GetNumberOfChildren(hierarchy, idx);
		if (numberOfChildren <= 0)
		{
			continue;
		}
		if (numberOfChildren >= 2) {
			contoursWithChildren.push_back(contours[idx]);
			vector<vector<Point>> children = GetChildren(hierarchy[idx][2], hierarchy, contours);
			for each (auto var in children)
			{
				contoursWithChildren.push_back(var);
			}
		}
	}
	return contoursWithChildren;
}

vector<vector<Point>> GetContours(Mat thresh) {
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(thresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	vector<vector<Point>> contoursWithChildren = GetChildren(0, hierarchy, contours);

	return contoursWithChildren;
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

	Mat gauss = GetGaussianBlur(afterMorph);
	imshow("Gaussian blurred image", gauss);
	waitKey();
	imwrite(FOLDER + "06gauss.jpg", gauss);

	Mat thresh = GetAdaptiveThreshold(gauss);
	imshow("Adaptive threshold image", thresh);
	waitKey();
	imwrite(FOLDER + "07thresh.jpg", thresh);

	Mat canny = GetCannyImg(grayscaleImg);
	imshow("Canny edge detector", canny);
	waitKey();
	imwrite(FOLDER + "08canny.jpg", canny);

	Mat contoursImg = Mat::zeros(thresh.size(), CV_8UC3);
	vector<vector<Point>> contours = GetContours(thresh);
	for (size_t i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		drawContours(contoursImg, contours, (int)i, color); // thickness < 0 => kit�lt�s
		//rectangle(contoursImg, boundingRect(contours[i]).tl(), boundingRect(contours[i]).br(), color, 2);
	}
	imshow("Contours", contoursImg);
	waitKey();
	imwrite(FOLDER + "09contours.jpg", contoursImg);
	return 0;
}