#pragma once
#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "crop.h"
#include "error.h"

#define GAUSSSIZE 5

double condition1[12] = { 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.05, 0.6, 0.35, 0.55 };
double thres1[3] = { 75, 150, 5 };
double condition2[12] = { 0.1, 0.9, 0, 1, 0.1, 0.9, 0, 1, 0.01, 0.6, 0.7, 1 };
double thres2[3] = { 75, 150, 10 };
double groupCondition = 0.3;

using namespace std;
using namespace cv;

void TestCodeC(Mat& imgRGB,Rect&);

vector<vector<Rect>>* Crop(Mat& imgRGB,double* rectCondition,double* thres,double groupConditon)
{
	Mat imgGray;
	imgGray = imgRGB.clone();
	cvtColor(imgRGB, imgGray, COLOR_BGR2GRAY);
	GaussianBlur(imgGray, imgGray, Size(GAUSSSIZE, GAUSSSIZE),0,0);
	int imgW = (imgGray).size().width;
	int imgH = (imgGray).size().height;
	vector<vector<Rect>>* res = new vector<vector<Rect>>;
	for (double i = thres[0]; i < thres[1]; i += thres[2])
	{
		Mat testImg;
		testImg = imgRGB.clone();
		Mat thresImg;
		threshold(imgGray, thresImg, i, 255, THRESH_BINARY);
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(thresImg, contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);
		cout << "¼ì²âµ½ÂÖÀª¸öÊý:" << contours.size() << "\n" << endl;
		for (int j = 0; j < contours.size(); j++)
		{
			Rect rect=boundingRect(contours[j]);
			if (rectCondition[0] * imgW < rect.x && rect.x < rectCondition[1] * imgW &&
				rectCondition[2] * imgH < rect.y && rect.y < rectCondition[3] * imgH &&
				rectCondition[4] * imgW < (rect.x + rect.width) && (rect.x + rect.width) < rectCondition[5] * imgW &&
				rectCondition[6] * imgH < (rect.y + rect.height) && (rect.y + rect.height) < rectCondition[7] * imgH &&
				rectCondition[8] * imgW < rect.width && rect.width < rectCondition[9] * imgW &&
				rectCondition[10] * imgH < rect.height && rect.height < rectCondition[11] * imgH)
			{

				TestCodeC(testImg, rect);

				if ((*res).size() == 0)
				{
					vector<Rect> temp;
					temp.push_back(rect);
					(*res).push_back(temp);
				}
				else
				{
					for (int k = 0; k < (*res).size(); k++)
					{
						if (abs(rect.x - (*res)[k][0].x < groupConditon*(*res)[k][0].width) &&
							abs(rect.y - (*res)[k][0].y < groupConditon*(*res)[k][0].height))
						{
							(*res)[k].push_back(rect);
							break;
						}
						else if (k == (*res).size()-1)
						{
							vector<Rect> temp;
							temp.push_back(rect);
							(*res).push_back(temp);
						}
					}
				}
			}
		}
	}

	return res;
}

vector<Mat>* CropImg(Mat& imgRGB, double* rectCondition, double* thres, double groupConditon,int Y)
{
	int xU = 9999; int yU = 9999;
	int xL = -1; int yL = -1;
	vector<vector<Rect>>* crop = Crop(imgRGB, rectCondition, thres, groupConditon);
	vector<Mat>* cropedImg = new vector<Mat>;
	bool isZero = CropErrorCheck(crop);
	if (!isZero)
	{
		for (int i = 0; i < (*crop).size(); i++)
		{
			if ((*crop)[i].size() > Y)
			{
				for (int j = 0; j < (*crop)[i].size(); j++)
				{
					Rect tempRect = (*crop)[i][j];
					if (tempRect.x < xU) { xU = tempRect.x; }
					if (tempRect.x+tempRect.width > xL) { xL = tempRect.x+tempRect.width; }
					if (tempRect.y < yU) { yU = tempRect.y; }
					if (tempRect.y+tempRect.height > yL) { yL = tempRect.y+tempRect.height; }
					Mat tempImg = imgRGB(Range(xU, xL), Range(yU, yL));
					(*cropedImg).push_back(tempImg);
				}
			}
		}
	}

	delete(crop);
	return cropedImg;
}

void TestCodeC(Mat& imgRGB,Rect& rect)
{
	rectangle(imgRGB, rect, Scalar(0, 0, 255), 8);
	namedWindow("cropRect");
	imshow("cropRect", imgRGB);
	waitKey();
}