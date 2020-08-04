#pragma once
#include <string>
#include <iostream>
#include <opencv2\opencv.hpp>

// #define TEST_createRatioMap
// #define TEST_createRatiomap
// #define TEST_createRatioMap_simple
// #define TEST_detectShadowByRatioMap

#define TS 40	//��ѡ��Ӱ��ͨ���������ظ�������ֵ��<40�����صķ���ȥ��
#define TH 0	//����Ӱ���ص���ͨ���������ظ�������ֵ��<30��ö������ϼ���Ϊ��Ӱ����

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

		/*�⼸������������[1]����Ҫʵ�ַ���STS-based���ⲿ�ֵĴ��������⣬��Ҫ��һ������
		void succesiveThresholding();
		void detectShadow(Point* pCS, int num, bool flag);
		double separabilityAnalysis(int tl, double nHistogram[]);
		void fineShadow();

		void videoShadowDetection(const string& _videoName, const string& resultVideo);
*/
	private:
		Mat originImage;	// ԭʼͼ��
		Mat shadowImage;	// ������Ӱͼ��
		int regionNums;		// ��������
		Mat RatioMap;		// ����ͼ
		Mat markCS;			// ��ѡ��Ӱ���صı��
		Mat shadowMask;		// ��ʼ��Ӱ��ֵ���
		Mat shadowLabel;	// ��ѡ��Ӱ����ͨ������ [1, 2, 3, ..., N]
		Mat shadowCandi;	// ��ѡ��Ӱ�Ķ�ֵ���
		Mat shadowFinal;	// ���յ���Ӱ��ֵ���

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
			//����ʦ�ֵ�ʵ�ּ���eǰ���1/(sqrt(2*PI)*sigma)
			//result = result / (sqrt(2 * PI)*sigma);
			return result;
		}
};

