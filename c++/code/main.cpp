#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "roughdetect.h"
#include "crop.h"
#include "error.h"

using namespace std;
using namespace cv;


int main()
{
	char* srcPath = "c:\\users\\Sunxk\\desktop\\c++\\rotatetest\\test.jpg";
	char* modelPath = "c:\\users\\Sunxk\\desktop\\c++\\model\\cascade.xml";
	/*roughdetect test img*/
	Mat img = imread(srcPath);
	imshow("nima", img);
	waitKey();

	vector<Mat>* roughRect = RoughDetect(srcPath, modelPath);

	RoughDetectErrorCheck(roughRect);
	
	for (int i = 0; i < (*roughRect).size(); i++)
	{
		Mat tempImg = (*roughRect)[i];
		vector<vector<Rect>>* crop;
		crop = Crop(tempImg, condition1, thres1, groupCondition);
		bool isZero = CropErrorCheck(crop);
		if (!isZero) { continue; }
		else
		{
			;
		}

		delete(crop);
	}

	destroyAllWindows();
	delete(roughRect);
	system("pause");
	return 0;
}
