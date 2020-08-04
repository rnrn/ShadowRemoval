#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <opencv2\opencv.hpp>

using std::string;
using std::vector;
using cv::Mat;
using cv::VideoCapture;

class Video
{
public:
	Video();
	Video(string& fileName, int ImageOrVideo);
	~Video();

	/* mem to mem*/
	Mat getImage(int frameIdx) const;
	VideoCapture getVideo(void) const;

	/* mem to file*/
	void saveImage(string& fileName);
	void saveVideo(string& fileName);

	/* file to mem */
	void readVideo(string& fileName);
	void readImage(string& fileName);

	/* get information */
	int getHeight(void) const;
	int getWidght(void) const;
	int getFrames(void) const;
	double getFrameRate(void) const;

private:
	int height;
	int width;
	int frames;
	double frameRate;
	VideoCapture originVideo;
	vector<Mat> originImage;
	void initVideo(string& fileName);
	void initImage(string& fileName);
};

