#include "stdafx.h"
#include "Video.h"
#include <iostream>
#include <stdio.h>
#include <time.h>

using namespace std;
using namespace cv;

Video::Video() : height(0), width(0), frames(0), frameRate(0) {
}

Video::Video(string & fileName, int ImageOrVideo) {
	// initial Video using image sequence or video
	// input full name
	if (fileName.empty()) {
		cout << "File Name is Null" << endl;
		return;
	}
	if (ImageOrVideo == 0)
		initImage(fileName);
	else if (ImageOrVideo == 1)
		initVideo(fileName);
}


Video::~Video()
{
}

Mat Video::getImage(int frameIdx) const {
	return originImage[frameIdx];;
}

VideoCapture Video::getVideo(void) const {
	return originVideo;
}

void Video::saveImage(string & fileName) {
	char fullFileName[255];
	for (int i = 0; i < Video::frames; i++) {
		strcpy_s(fullFileName, "");
		sprintf_s(fullFileName, "%s%03d%s", fileName.c_str(), i, ".bmp");
		imwrite(fullFileName, originImage[i]);
	}
}

void Video::saveVideo(string & fileName) {
	int fourcc = CV_FOURCC('X', 'V', 'I', 'D');
	Size size(width, height);
	VideoWriter videoWriter(fileName, fourcc, frameRate, size);
	for (int i = 0; i < frames; i++)
		videoWriter.write(originImage[i]);
}

void Video::readVideo(string & fileName) {
	initVideo(fileName);
}

void Video::readImage(string & fileName) {
	initImage(fileName);
}

int Video::getHeight(void) const {
	return height;
}

int Video::getWidght(void) const {
	return width;
}

int Video::getFrames(void) const {
	return frames;
}

double Video::getFrameRate(void) const {
	return frameRate;
}

void Video::initVideo(string & fileName) {
	// read video through videoName with full path and file type
	if (fileName.empty()) {
		cout << "the input video name is null" << endl;
		return;
	}

	// init Video
	originVideo.open(fileName);
	if (!originVideo.isOpened()) {
		cout << "Open " << fileName << " failed." << endl;
		return;
	}
	// set info
	width = (int)originVideo.get(CV_CAP_PROP_FRAME_WIDTH);
	height = (int)originVideo.get(CV_CAP_PROP_FRAME_HEIGHT);
	frames = (int)originVideo.get(CV_CAP_PROP_FRAME_COUNT);
	frameRate = (int)originVideo.get(CV_CAP_PROP_FPS);
	// set the start frame
	originVideo.set(CV_CAP_PROP_POS_FRAMES, 0);

	// init Images
	Mat currentFrame;
	while (originVideo.read(currentFrame)) {
		Mat tempFrame = currentFrame.clone();
		originImage.push_back(tempFrame);
	}
}

void Video::initImage(string & fileName) {
	// read image through imageName with full path and file type
	if (fileName.empty()) {
		cout << "the image name is null" << endl;
		return;
	}

	// frame info set
	char frameName[255];
	Mat currentFrame;
	strcpy_s(frameName, "");
	int frameIdx = 0;

	// read first frame and set info
	sprintf_s(frameName, "%s%03d%s", fileName.c_str(), frameIdx, ".bmp");
	currentFrame = imread(string(frameName));
	if (currentFrame.empty()) {
		cout << "read image failed, please check the directory again \n";
		return;
	}
	width = currentFrame.cols;
	height = currentFrame.rows;
	frameRate = 24.0;

	// read in image
	while (!currentFrame.empty()) {
		Mat tempFrame = currentFrame.clone();
		originImage.push_back(tempFrame);
		// set next frame
		strcpy_s(frameName, "");
		sprintf_s(frameName, "%s%03d%s", fileName.c_str(), frameIdx, ".bmp");
		currentFrame = imread(string(frameName));
		++frameIdx;
	}
	frames = originImage.size();

	// write temp video
	string tempVideo = "temp.avi";
	saveVideo(tempVideo);
	originVideo.open(tempVideo);
	originVideo.set(CV_CAP_PROP_POS_FRAMES, 0);

	// delete temp Video
	remove(tempVideo.c_str());
}
