#pragma once
#include "error.h"

void RoughDetectErrorCheck(vector<Mat>* roughRect)
{
	if ((*roughRect).size() < 1)
	{
		cerr << "Error:Undetected target!\n";
		getchar();
		exit(1);
	}
}

bool CropErrorCheck(vector<vector<Rect>>* crop)
{
	if ((*crop).size() == 0)
	{
		cout << "Warning:No rect detected!" << "\n" << endl;
		return false;
	}
	return true;
}