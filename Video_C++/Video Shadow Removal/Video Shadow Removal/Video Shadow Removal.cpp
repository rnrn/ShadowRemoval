// Video Shadow Removal.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include "Video.h"
#include "imageShadow.h"
#include <cstdio>
#include <iostream>
#include <cstring>
#include <string>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

using namespace std;
using namespace cv;

int main() {

	// read video file
	String originVideoName = "D:\\input.avi";
	String saveToImage = "D:\\images\\";
	Video originVideo(originVideoName, 1);
	originVideo.saveImage(saveToImage);
	//Video originVideo(saveToImage, 0);
/*
	// get image shadow
	int frames = originVideo.getFrames();
	vector<Mat> shadowImages;

	for (int i = 0; i < frames; i++) {
		ImageShadow getImageShadow(originVideo.getImage[i]);
		shadowImages.push_back = getImageShadow.getImage();
	}
*/
	/*
	ViedoMatting vm;

	int frames = originVideo.frames;
	Mat shadowImg[frames];
	for (int i = 0; i < frames; i++)
	shadowImg[i] = is.getShadow(originVideo.getImage[i]);
	shadowVideo.img2Video(&shadowImg, frames, 24);
	shadowVideo.
	*/
/*
	//originVideo.saveImage(saveToImage);
	originVideo.saveVideo(originVideoName);
	system("PAUSE");
	return 0;
*/


}

