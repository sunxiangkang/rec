#pragma once
#include "crop.h"
#include "deskewV.h"
#include <opencv2/opencv.hpp>
#include <math.h>

#define M_PI 3.14159265358979323846
#define Y 1

using namespace std;
using namespace cv;

struct myShape
{
	int height;
	int width;
};

Mat DeskewV(Mat* imgRGB)
{
	Mat imgRGBBorder;
	int height = (*imgRGB).size().height;
	int width = (*imgRGB).size().width;
	copyMakeBorder(*imgRGB, imgRGBBorder, 
		int(height*0.1), int(height*0.1), 
		int(width*0.1), int(width*0.1), 
		BORDER_CONSTANT, Scalar(255, 255, 255));
	vector<Vec4i> res;
	vector<int> mArea;
	for (int theta = -25; theta < 25 + 1; theta++)
	{
		Mat tempImg;
		if (theta == 0) { imgRGBBorder.copyTo(tempImg); }
		else
		{
			tempImg = Rot(imgRGBBorder, theta);
		}
		vector<vector<Rect>>* rectl;
		rectl= Crop(tempImg, condition2, thres2, groupCondition);
		for (int i = 0; i < (*rectl).size(); i++)
		{
			if ((*rectl)[i].size() <= Y) { continue; }
			vector<Rect> tempRect = (*rectl)[i];
			int topMax = -1, leftMax = -1;
			int botMin = (*imgRGB).size().height;
			int rightMin = (*imgRGB).size().width;
			for (int j = 0; j < tempRect.size(); j++)
			{
				if (tempRect[j].x > leftMax) { leftMax = tempRect[j].x; }
				if (tempRect[j].y > topMax) { topMax = tempRect[j].y; }
				if (tempRect[j].x + tempRect[j].width < rightMin) 
				{ 
					rightMin = tempRect[j].x + tempRect[j].width; 
				}
				if (tempRect[j].y + tempRect[j].height < botMin)
				{
					botMin = tempRect[j].y + tempRect[j].height;
				}
			}
			Vec4i temp = { topMax,leftMax,botMin,rightMin };
			res.push_back(temp);
			mArea.push_back(rightMin - leftMax);
		}
	}
	int maxIndex = -1, maxArea = -1;
	for (int i = 0; i < mArea.size(); i++)
	{
		if (mArea[i] > maxArea)
		{
			maxIndex = i;
			maxArea = mArea[i];
		}
	}
	int adTheta = maxIndex - 25;
	cout << "adjust theta:" << adTheta << endl;
	Mat adImg = Rot(*imgRGB, adTheta);
	Vec4i rec = res[maxIndex + 25];
	if (rec[1] - int((*imgRGB).size().height*0.1) < 0) { rec[1] = 0; }
	if (rec[3] - (*imgRGB).size().height > 0) { rec[3] = (*imgRGB).size().height; }
	Mat dst = (*imgRGB)(Range(rec[0], rec[2]), Range(rec[1],rec[3]));
	return dst;
}

Mat& Rot(Mat& imgRGBBorder, float theta)
{
	int height = imgRGBBorder.size().height;
	int width = imgRGBBorder.size().width;
	int interval = abs(int(sin(theta / 180 * M_PI) + height));
	myShape shape;
	shape.height = height;
	shape.width = width + interval;
	vector<Point2f> subImg(4);
	vector<Point2f> stdImg(4);
	stdImg[0] = Point2f(0, 0);
	stdImg[1] = Point2f(0, height);
	stdImg[2] = Point2f(width, 0);
	stdImg[3] = Point2f(width, height);
	if (theta > 0)
	{
		subImg[0] = Point2f(interval, 0);
		subImg[1] = Point2f(0, height);
		subImg[2] = Point2f(shape.width, 0);
		subImg[3] = Point2f(width, height);
	}
	else
	{
		subImg[0] = Point2f(0, 0);
		subImg[1] = Point2f(interval, height);
		subImg[2] = Point2f(width, 0);
		subImg[3] = Point2f(shape.width, height);
	}
	Mat transform= getPerspectiveTransform(subImg, stdImg);
	Mat* outImg = new Mat;
	warpPerspective(imgRGBBorder, *outImg, transform,
		Size(shape.width,shape.height),
		INTER_CUBIC, 0, Scalar(255, 255, 255));
	return *outImg;
}