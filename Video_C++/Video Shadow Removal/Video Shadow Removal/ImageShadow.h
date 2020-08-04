#pragma once
#include <string>
#include <iostream>
#include <opencv2\opencv.hpp>

// #define TEST_createRatioMap
// #define TEST_createRatiomap
// #define TEST_createRatioMap_simple
// #define TEST_detectShadowByRatioMap

#define TS 40	//候选阴影连通分区的像素个数的阈值，<40个像素的分区去掉
#define TH 0	//非阴影像素的连通分析的像素个数的阈值，<30则该洞被补上即变为阴影像素

using std::string;
using cv::Mat;
using cv::Vec3d;

const double PI = 3.1415926535897932384626433832;

class ImageShadow
{
	public:
		/* constructor and deconstructor */
		ImageShadow();
		ImageShadow(string& imageName);
		ImageShadow(Mat imageName);
		~ImageShadow();

		/* read and write image */
		void readImage(string& imageName);
		void readImage(Mat& imageName);
		void writeImage(string& fileName);
		Mat getImage();

		/* Shadow Detection */
		Mat createRatioMap(Mat& src);
		Mat createRatioMap_simple(const Mat &src);
		Mat detectShadowByRatioMap(Mat &src);
/*
		int getThresholdTs(const Mat& src, double& sigma);
		int getThresholdT(const Mat& src);
		void BilteralFilter(Mat& src);

		int connectedComponentLabel(const Mat& rawShadowMask, Mat& shadowLabel, const int mode);
		int connectedComponentAnalysis(const Mat& rawShadowMask, Mat& shadowLabel, const int mode);
		void localThresholdProcess();
		void fineShadowDetermination();

		void shadowProcessing(Mat& shadowMap);
		void smallShadowRegionElimination(Mat& shadowmap);
		void smallShadowRegionFilling(Mat& shadowmap);

		/*这几个函数是文献[1]的主要实现方法STS-based，这部分的代码有问题，需要进一步调试
		void succesiveThresholding();
		void detectShadow(Point* pCS, int num, bool flag);
		double separabilityAnalysis(int tl, double nHistogram[]);
		void fineShadow();

		void videoShadowDetection(const string& _videoName, const string& resultVideo);
*/
	private:
		Mat originImage;	// 原始图像
		Mat shadowImage;	// 最终阴影图像
		int regionNums;		// 区域数量
		Mat RatioMap;		// 比率图
		Mat markCS;			// 候选阴影像素的标记
		Mat shadowMask;		// 初始阴影二值标记
		Mat shadowLabel;	// 候选阴影的连通区域标记 [1, 2, 3, ..., N]
		Mat shadowCandi;	// 候选阴影的二值标记
		Mat shadowFinal;	// 最终的阴影二值标记

		/* inline functions */
		double inline distance2D (int x1, int y1, int x2, int y2) {
			return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
		}

		double inline distance2D (Point p1, Point p2) {
			return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
		}

		double inline distance2D_square (int x1, int y1, int x2, int y2) {
			return (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2);
		}
		double inline distance2D_square (Point p1, Point p2) {
			return (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
		}

		double inline distance_gray (const Mat &src, int x1, int y1, int x2, int y2) {
			return abs(src.at<double>(y1, x1) - src.at<double>(y2, x2));
		}

		double inline distance_rgb (const Mat &src, int x1, int y1, int x2, int y2) {
			return sqrt((src.at<Vec3d>(y1, x1)[0] - src.at<Vec3d>(y2, x2)[0])*(src.at<Vec3d>(y1, x1)[0] - src.at<Vec3d>(y2, x2)[0]) +
				(src.at<Vec3d>(y1, x1)[1] - src.at<Vec3d>(y2, x2)[1])*(src.at<Vec3d>(y1, x1)[1] - src.at<Vec3d>(y2, x2)[1]) +
				(src.at<Vec3d>(y1, x1)[2] - src.at<Vec3d>(y2, x2)[2])*(src.at<Vec3d>(y1, x1)[2] - src.at<Vec3d>(y2, x2)[2]));
		}

		double inline distance_rgb_abs (const Mat &src, int x1, int y1, int x2, int y2) {
			return abs(src.at<Vec3d>(y1, x1)[0] - src.at<Vec3d>(y2, x2)[0]) +
				abs(src.at<Vec3d>(y1, x1)[1] - src.at<Vec3d>(y2, x2)[1]) + abs(src.at<Vec3d>(y1, x1)[2] - src.at<Vec3d>(y2, x2)[2]);
		}

		double gaussianKernel (double x, double sigma) {
			double result = 0;
			result = exp(-x*x / (2 * sigma*sigma));
			//张青师兄的实现加了e前面的1/(sqrt(2*PI)*sigma)
			//result = result / (sqrt(2 * PI)*sigma);
			return result;
		}
};

