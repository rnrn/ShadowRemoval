/************************************************************************
project: image shadow removal
function: image shadow removal using the zhang et al algorithm.
************************************************************************/
#pragma once
#include <opencv2\opencv.hpp>
#include <vector>
#include <map>
#include <utility>
#include <string>

using std::vector;
using std::string;
using std::map;
using std::pair;
using cv::Mat;
using cv::Vec3b;

//6ά��������������
struct pixelFeature
{
	uchar intensity;
	uchar chromaticity;
	float firstDevX, firstDevY;
	float secondDevX, secondDevY;
};

struct patchFeature
{
	float corvarianceVector;
};

typedef struct pixel2D
{
	uchar b, g, r;//uchar, short,int,float,double ...
	uchar lbpB, lbpG, lbpR;
	uchar alpha;
	int x, y;
	pixelFeature pf;
}pixel2D;


class ShadowRemovalImg
{
public:
	ShadowRemovalImg();
	~ShadowRemovalImg();

	//memory operations
	pixel2D *** memAlloc(int w, int h, int patchNum);
	void memFree(pixel2D **** p);
	void clear();

	//patch decomposition
	void imageOverlapDecompose();
	void imageDecompose();

	void patchDecomposeTest();

	//patchMatch
	void patchDivision2LitAndShadow();
	double getDisOfPos(int pIndex1, int pIndex2) const;
	double getDisOfRGB(int pIndex1, int pIndex2) const;
	void logOperator();
	void lbpDescriptor();
	void imageNormalize(Mat * input);
	Vec3b getLbpValue(int x, int y) const;
	Vec3b getImprovedLbpValue(int x, int y) const;

	void computePatchFeatureDescriptor();

	double getDisOfTexture_lbp(int pIndex1, int pIndex2) const;
	double getDisOfTexture_covM(int pIndex1, int pIndex2) const;
	double getDisOfTexture_fea(int pIndex1, int pIndex2) const;
	double getDisOfChromaticity(int pIndex1, int pIndex2) const;

	double getPatchSimilarity(int pIndex1, int pIndex2) const;
	void coherentPatchMatch2D();

	//shadow removal process
	cv::Vec3d getMeanValueofPatch(int p);
	double getMeanValueofPatchIntensity(int p);
	double getStandardDevofPatch(int  p);
	void illuminationTransferOperator_Naive(int _shadowPatchIndex);
	void illuminationTransferOperator_shor(int _shadowPatchIndex);
	void illuminationTransferOperator_Zhang(unsigned int _shadowPatchIndex);
	void patchPixelNormalization(int p);
	void shadowRemovalLocal(const string & _inputImg, const string _shadowImg, const string _shadowFreeImg);
	void shadowRemovalGlobal(const string & _inputImg, const string & _shadowImg, const string & _shadowfreeImg);

	//boundary post process
	void shadowBoundaryProcessing();
	//write the final results
	double getSpatialDistance(int x1, int x2, int y1, int y2) const;
	inline double maxDistance(double dis1, double dis2, double dis3, double dis4);
	void imagePatchesRecompose();
	void imageRecompose();

	double maxValued(double data[], int n)
	{
		double maxv = data[0];
		for (int i = 0; i < n; ++i)
		{
			if (maxv < data[i])
				maxv = data[i];
		}
		return maxv;
	}

	pixel2D *** patchImg;


private:
	Mat inputImg;//original image
	Mat chromaticity;
	Mat logFilteredImg;
	Mat shadowImg;//detected shadow
	Mat shadowlessImg;//shadow free image before shadow boundary processing
	Mat shadowFreeFinalImg;//final shadow free image after shadow boundary processing

	int imHeight, imWidth, imChannels;
	int imSize;

	int patchWidth, patchHeight;//patch size
	int patchSize;
	int gapx, gapy;
	int patchNumber;//the num of overlapped patches

					//the lit patch index and shadow patch index
	std::vector<int> litPatchIndices;
	std::vector<int> shadowPatchIndices;
	//vector<pair<int, int>> match; //the matched lit and shadow patch pair index
	map<int, int> match;

	std::vector<std::vector<double>> patchCovVector;//�洢ÿ��patch���������patch������Э������������

													//different kinds of image reconstruction strategies
													/************************************************************************
													0.ֱ������ƽ��
													1.����ļ�Ȩƽ��
													3.�����Զ����ļ�Ȩƽ��
													************************************************************************/
	int compositeWays = 0;
};

