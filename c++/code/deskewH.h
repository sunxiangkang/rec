#pragma once
#include <opencv2/opencv.hpp>
#include "deskewH.h"

#define Y 1

using namespace std;
using namespace  cv;

int* FitlineRansac(vector<vector<Rect>>* rectV, int* imgSize)
{
	int counter;
	vector<Point>* ptsT;
	vector<Point>* ptsB;
	for (int i = 0; i < (*rectV).size(); i++)
	{
		if ((*rectV)[i].size() > Y)
		{
			counter++;
			for (int j = 0; j < (*rectV)[i].size(); j++)
			{
				Point tempPt;
				tempPt.x = (*rectV)[i][0].x;
				tempPt.y = (*rectV)[i][0].y;
				(*ptsT).push_back(tempPt);
				tempPt.x = (*rectV)[i][0].x + (*rectV)[i][0].width;
				tempPt.x = (*rectV)[i][0].y + (*rectV)[i][0].height;
				(*ptsB).push_back(tempPt);
			}
		}
		else if (i == ((*rectV).size()-1) && counter == 0)
		{
			//一般不会发生
			cout << "Warning:No rect match condition!" << endl;
		}
	}

	int* out = new int[4];
	if ((*ptsT).size() > 2)
	{
		if (counter > 1)
		{
			Vec4f lineT, lineB;
			fitLine(*ptsT, lineT, CV_DIST_HUBER, 0, 1e-2, 1e-2);
			fitLine(*ptsB, lineB, CV_DIST_HUBER, 0, 1e-2, 1e-2);
			int leftT = int((-lineT[2] * lineT[1] / lineT[0]) + lineT[3]);
			int rightT = int((imgSize[1]-lineT[2])*lineT[1]/lineT[0]+lineT[3]);
			int leftB = int((-lineB[2] * lineB[1] / lineB[0]) + lineB[3]);
			int rightB = int((imgSize[1] - lineB[2])*lineB[1] / lineB[0] + lineB[3]);
			out[0] = leftT;out[1] = rightT;
			out[2] = leftB; out[3] = rightB;
		}
		else
		{
			int rowNumT = 0;int rowNumB=0;
			for (int i = 0; i < (*ptsT).size(); i++)
			{
				rowNumT += (*ptsT)[i].y;
				rowNumB += (*ptsB)[i].y;
				if (i == ((*ptsT).size() - 1))
				{
					rowNumT /= i; rowNumB /= i;
					out[0] = out[1] = rowNumT;
					out[2] = out[3] = rowNumB;
				}
			}
		}
	}
	else
	{
		out[0] = out[1] = out[2] = out[3] = 0;
	}
	return out;
}

Mat* DeskewH(Mat* imgRGB, vector<vector<Rect>>* rectL)
{
	Mat imgGray;
	cvtColor(*imgRGB, imgGray, COLOR_RGB2GRAY);
	int* imgSize;
	imgSize[0] = imgGray.size().width;
	imgSize[1] = imgGray.size().height;
	int* imgContour;
	imgContour = FitlineRansac(rectL, imgSize);
	vector<Point2f> subImg(4);
	vector<Point2f> stdImg(4);
	subImg[0] = Point2f(0, imgContour[0]);
	subImg[1] = Point2f((*imgRGB).size().width - 1, imgContour[1]);
	subImg[2] = Point2f((*imgRGB).size().width - 1, imgContour[4]);
	subImg[3] = Point2f(0, imgContour[3]);
	stdImg[0] = Point2f(0, 0);
	stdImg[1] = Point2f((*imgRGB).size().width - 1, 0);
	stdImg[2] = Point2f((*imgRGB).size().width - 1, int((*imgRGB).size().height / 2));
	stdImg[3] = Point2f(0, int((*imgRGB).size().height / 2));
	Mat transform=getPerspectiveTransform(subImg, stdImg);
	Mat* outImg = new Mat;
	warpPerspective(*imgRGB, *outImg, transform, 
		Size((*imgRGB).size().width, 
			int((*imgRGB).size().height/2)),
		INTER_CUBIC,0, Scalar(255, 255, 255));

	delete(imgContour);
	return outImg;
}