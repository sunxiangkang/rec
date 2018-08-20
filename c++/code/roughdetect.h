#pragma once
#include <string.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "roughdetect.h"

using namespace std;
using namespace cv;

void TestCodeR(vector<Mat>*);

vector<Mat>* RoughDetect(char* srcPath, char* modelPath)
{
	Mat src = imread(srcPath);
	Mat srcG;
	vector<Rect> res;
	vector<Mat>* out=new vector<Mat>;

	cvtColor(src, srcG, CV_RGB2GRAY);

	CascadeClassifier watchCascade;
	if (!watchCascade.load(modelPath))
	{
		cout << "Error loading watchCascade\n";
	}

	watchCascade.detectMultiScale(srcG, res, 1.1, 1);

	for (int i = 0; i < res.size(); i++)
	{
		Mat imgRoi = src(res[i]);
		(*out).push_back(imgRoi);
	}

	TestCodeR(out);

	return out;
}

void TestCodeR(vector<Mat>* out)
{
	for (int i = 0; i < (*out).size(); i++)
	{
		namedWindow("RoughDetect", 1);
		imshow("RoughDetect", (*out)[i]);
		waitKey();
	}
	destroyAllWindows();
}