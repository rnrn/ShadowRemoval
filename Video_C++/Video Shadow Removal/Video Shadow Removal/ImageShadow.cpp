#include "stdafx.h"
#include "ImageShadow.h"

using namespace std;
using namespace cv;

ImageShadow::ImageShadow() {
}

ImageShadow::ImageShadow(string& imageName) {
	originImage = imread(imageName);
}

ImageShadow::ImageShadow(Mat imageName) {
	originImage = imageName;
}

ImageShadow::~ImageShadow()
{
}

void ImageShadow::readImage(string& imageName) {
	originImage = imread(imageName);
}

void ImageShadow::readImage(Mat& imageName) {
	originImage = imageName;
}

void ImageShadow::writeImage(string & fileName) {
	imwrite(fileName, shadowImage);
}

Mat ImageShadow::getImage() {
	return shadowImage;
}

Mat ImageShadow::createRatioMap(Mat& src) {
	int height = src.rows, width = src.cols;
	Mat img_h(src.size(), CV_64FC1);
	Mat img_i(src.size(), CV_64FC1);

	//compute the hue and intensity channel and normalize them to [0,1]
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			//the intensity channel
			img_i.at<double>(i, j) = (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2]) / 3.0;
			double v1 = sqrt(6.0)*(2.0 * src.at<Vec3b>(i, j)[0] - (double)src.at<Vec3b>(i, j)[1] - (double)src.at<Vec3b>(i, j)[2]) / 6.0;
			double v2 = (src.at<Vec3b>(i, j)[2] - 2 * src.at<Vec3b>(i, j)[1]) / sqrt(6.0);
			//the hue channel
			if (v1 != 0) {
				img_h.at<double>(i, j) = (atan2(v2, v1) + PI) * 0.5 / PI;
			}
			else {
				v1 = 0.0000001f;
				img_h.at<double>(i, j) = (atan2(v2, v1) + PI) * 0.5 / PI;
			}
		}
	}
	img_i.convertTo(img_i, CV_64FC1, 1.0 / 255);

	//create the raiomap and normalize to [0,255]
	Mat ratioMap(height, width, CV_64FC1);
	double max_ratio = -1.0f, min_ratio = 1.79769e+308;
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			ratioMap.at<double>(i, j) = (img_h.at<double>(i, j) + 0.1) / (img_i.at<double>(i, j) + 0.01);
			if (ratioMap.at<double>(i, j) > max_ratio)
				max_ratio = ratioMap.at<double>(i, j);
			if (ratioMap.at<double>(i, j) < min_ratio)
				min_ratio = ratioMap.at<double>(i, j);
		}
	}
	cout << "max_ratio: " << max_ratio << " min_ratio: " << min_ratio << endl;
	double ratioRange = max_ratio - min_ratio;
	Mat ratiomap(height, width, CV_32SC1);
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			ratiomap.at<int>(i, j) = (int)((ratioMap.at<double>(i, j) - min_ratio) * 255 / ratioRange);
		}
	}

	//compute the threshold Ts and sigma2
	double sigma2 = 0;
	int Ts = getThresholdTs(ratiomap, sigma2);
	cout << "Ts: " << Ts << "  sigma2： " << sigma2 << endl;

	//get the final ratiomap
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			if (ratiomap.at<int>(i, j) >= Ts)
				ratiomap.at<int>(i, j) = 255;
			else {
				double tmp = -(ratiomap.at<int>(i, j) - Ts)*(ratiomap.at<int>(i, j) - Ts) / (4 * sigma2);
				ratiomap.at<int>(i, j) = (int)(255 * exp(tmp));
			}
		}
	}

#ifdef TEST_createRatioMap
	imshow("intensity", img_i);
	imshow("hue", img_h);
	imshow("ratiomap_raw", ratioMap);
	ratiomap.convertTo(ratiomap, CV_8UC1);//for show
	imshow("ratiomap_final", ratiomap);
	waitKey(0);
#endif
	return ratiomap;
}

Mat ImageShadow::createRatioMap_simple(const Mat & src) {
	int height = src.rows, width = src.cols;
	Mat img_h(height, width, CV_32FC1);
	Mat img_s(height, width, CV_32FC1);
	Mat img_i(height, width, CV_32FC1);

	//convert the src to hsi color space
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			img_i.at<float>(i, j) = (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2]) / 3.0f;
			float v1 = (float)(sqrt(6.0)*(2.0 * src.at<Vec3b>(i, j)[0] - (float)src.at<Vec3b>(i, j)[1] - (float)src.at<Vec3b>(i, j)[2]) / 6.0);
			float v2 = (float)((src.at<Vec3b>(i, j)[2] - 2 * src.at<Vec3b>(i, j)[1]) / sqrt(6.0));
			img_s.at<float>(i, j) = sqrt(v1*v1 + v2*v2);
			if (v1 != 0) {
				img_h.at<float>(i, j) = (float)((atan2(v2, v1) + PI) * 0.5 / PI);
			}
			else {
				v1 = 0.0000001f;
				img_h.at<float>(i, j) = (float)((atan2(v2, v1) + PI) * 0.5 / PI);
			}
		}
	}
	//normalize the h,s,i to [0,1]
	img_i.convertTo(img_i, CV_32FC1, 1.0 / 255);
	img_s.convertTo(img_s, CV_32FC1, 1.0 / 255);
	Mat img_hsi(height, width, CV_32FC3);
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			img_hsi.at<Vec3f>(i, j)[0] = img_h.at<float>(i, j);
			img_hsi.at<Vec3f>(i, j)[1] = img_s.at<float>(i, j);
			img_hsi.at<Vec3f>(i, j)[2] = img_i.at<float>(i, j);
		}
	}

#ifdef TEST_createRatioMap_simple
	imshow("src", src);
	imshow("hue", img_h);
	imshow("saturation", img_s);
	imshow("intensity", img_i);
	imshow("hsi", img_hsi);
	waitKey(0);
#endif

	return Mat();
}

Mat ImageShadow::detectShadowByRatioMap(Mat& src) {
	/************************************************************************/
	/*  初始化该类的成员变量                                                   */
	/************************************************************************/
	shadowMask.create(src.size(), CV_32SC1);
	shadowMask = 0;
	shadowCandi.create(src.size(), CV_32SC1);
	shadowCandi = 0;
	shadowLabel.create(src.size(), CV_32SC1);
	shadowLabel = 0;
	shadowFinal.create(src.size(), CV_32SC1);
	shadowFinal = 0;
	markCS.create(src.size(), CV_32SC1);
	markCS = 0;
	regionNums = 0;

	//1.预处理- 得到改进后的比率图并对比率图进行滤波
	RatioMap = createRatioMap(src);
	RatioMap.convertTo(RatioMap, CV_8UC1);
	//imshow("ratiomap", RatioMap);

	Mat Ratiomap_filter;
	//就用OpenCV自带的实现，速度比较快
	bilateralFilter(RatioMap, Ratiomap_filter, 6, 30, 10);
	//imshow("ratiomap_bfiltered", Ratiomap_filter);


	//2.阴影检测- 取全局阈值，对候选阴影像素分区，取局部阈值 [1]的核心算法
	Ratiomap_filter.convertTo(Ratiomap_filter, CV_32SC1);
	int T = getThresholdT(Ratiomap_filter);
	cout << "T: " << T << endl;
	RatioMap = Ratiomap_filter.clone();
	//2.1 global thresholding to create the coarse shadow mask


	int shadowPixelNumc = 0;
	for (int i = 0; i < shadowMask.rows; ++i) {
		for (int j = 0; j < shadowMask.cols; ++j) {
			if (RatioMap.at<int>(i, j) > T)
			{
				shadowMask.at<int>(i, j) = 1;//candidate shadow pixels
				shadowPixelNumc++;
			}
			else
				shadowMask.at<int>(i, j) = 0;//nonshadow pixels
		}
	}
	cout << "candidate shadow pixels: " << shadowPixelNumc << endl;
	Mat shadowMaskCopy = shadowMask.clone();
	shadowMaskCopy.convertTo(shadowMaskCopy, CV_8UC1, 255);

#ifdef TEST_detectShadowByRatioMap
	imshow("shadowMaskCoarse", shadowMaskCopy);
	imwrite("D:\\Video Shadow Removal\\datasets\\test images\\coarseMask.jpg", shadowMaskCopy);
#endif // TEST_detectShadowByRatioMap

	//2.2 patition the candidate shadow pixels and processing the regions
	shadowProcessing(shadowMask);
	shadowMaskCopy = shadowMask.clone();
	shadowMaskCopy.convertTo(shadowMaskCopy, CV_8UC1, 255);

#ifdef IMGTEST
	imshow("shadowMaskBeforeSTS", shadowMaskCopy);
	imwrite("E:\\Video Shadow Removal\\datasets\\test images\\maskBeforeSTS.jpg", shadowMaskCopy);
#endif

	//2.3 取局部阈值
	//succesiveThresholding();

	//3 阴影检测结果的优化
	//fineshadow();

	//直接根据mask的结果在原图像上显示阴影检测的结果
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			if (shadowMask.at<int>(i, j) == 1) {
				src.at<Vec3b>(i, j)[0] = 255;
				src.at<Vec3b>(i, j)[1] = 255;
				src.at<Vec3b>(i, j)[2] = 255;
			}

		}
	}
	/*
	imshow("detection results", src);
	waitKey(41);
	*/
	shadowMask.convertTo(shadowMask, CV_8UC1, 255);
	return shadowMask;
}
