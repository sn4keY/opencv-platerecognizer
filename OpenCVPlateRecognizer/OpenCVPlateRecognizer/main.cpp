#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include "asprise_ocr_api.h"
#include <iostream>

using namespace cv;
using namespace std;

const string FOLDER = "D:\\Source\\OpenCVPlateRecognizer\\dw\\";
RNG rng(12345);

bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) {
	double i = fabs(contourArea(cv::Mat(contour1)));
	double j = fabs(contourArea(cv::Mat(contour2)));
	return (i > j);
}

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
	//medianBlur(thresh, thresh, 3);
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
		if (numberOfChildren == 0)
		{
			continue;
		}
		if (numberOfChildren >= 3) {
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

vector<vector<Point>> GetRectangles(vector<vector<Point>> contours) {
	vector<vector<Point> > approx(contours.size());
	vector<vector<Point>> rectangle;
	int i = 0;
	for each (auto var in contours)
	{
		double peri = arcLength(var, true);
		approxPolyDP(var, approx[i], 0.04 * peri, true);
		if (approx[i].size() == 4)
		{
			rectangle.push_back(approx[i]);
		}
		i++;
	}
	return rectangle;
}

vector<Point> GetPlate(vector<vector<Point>> contours) {
	vector<Point> plate;
	for each (auto var in contours)
	{
		if ((var[1].y - var[0].y) < 5 && (var[2].y - var[3].y) < 5)
		{
			if ((var[1].x - var[2].x) < 10 && (var[3].x - var[0].x) < 10)
			{
				return var;
			}
		}
	}
	for (size_t i = 0; i < 4; i++)
	{
		plate.push_back(Point(0, 0));
	}
	return plate;
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

	Mat rectanglesImg = Mat::zeros(thresh.size(), CV_8UC3);
	vector<vector<Point>> rectangles = GetRectangles(contours);
	for (size_t i = 0; i < rectangles.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		drawContours(rectanglesImg, rectangles, (int)i, color); // thickness < 0 => kit�lt�s
		//rectangle(rectanglesImg, boundingRect(rectangles[i]).tl(), boundingRect(rectangles[i]).br(), color, 2);
	}
	imshow("Plate likes", rectanglesImg);
	waitKey();
	imwrite(FOLDER + "10plates.jpg", rectanglesImg);

	vector<Point> plate = GetPlate(rectangles);
	vector<vector<Point>> plateWrapper;
	plateWrapper.push_back(plate);
	Mat plateImg = Mat::zeros(thresh.size(), CV_8UC3);
	Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
	drawContours(plateImg, plateWrapper, 0, color);
	imshow("Plate contour", plateImg);
	waitKey();
	imwrite(FOLDER + "11plate.jpg", plateImg);

	Mat masked = defaultImg(boundingRect(plateWrapper[0]));
	imshow("Plate masked", masked);
	waitKey();
	imwrite(FOLDER + "12croppedplate.jpg", masked);

	const char * libFolder = "D:\\Source\\OpenCVPlateRecognizer\\";
	LIBRARY_HANDLE libHandle = dynamic_load_aocr_library(libFolder);
	int setup = c_com_asprise_ocr_setup(false);
	long long ptrToApi = c_com_asprise_ocr_start("eng", OCR_SPEED_FAST, NULL, NULL, NULL);
	char * s = c_com_asprise_ocr_recognize(ptrToApi, (FOLDER + "12croppedplate.jpg").c_str(), -1, -1, -1, -1, -1,OCR_RECOGNIZE_TYPE_TEXT, OCR_OUTPUT_FORMAT_PLAINTEXT,NULL, NULL, NULL);

	std::cout << "Returned: " << s << std::endl;
	c_com_asprise_ocr_stop(ptrToApi);

	dynamic_unload_aocr_library(libHandle);

	//tesseract::TessBaseAPI api;
	//if (api.Init("D:\\Source\\Tesseract\\tessdata", "eng")) {
	//	fprintf(stderr, "Could not initialize tesseract.\n");
	//	exit(1);
	//}
	//api->SetImage((uchar*)masked.data, masked.size().width, masked.size().height, masked.channels(), masked.step1());
	//api->Recognize(0);
	//const char* out = api->GetUTF8Text();
	//printf(out);

	return 0;
}