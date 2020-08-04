/************************************************************************
project: image shadow removal
function: image shadow removal
************************************************************************/
#include "ImageShadowRemoval.h"
#include <algorithm>
#include <time.h>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

ShadowRemovalImg::ShadowRemovalImg()
{
	//��ʼ����ĳ�Ա����
	shadowlessImg = NULL;
	patchImg = NULL;

	imHeight = 0;
	imWidth = 0;
	imChannels = 0;
	imSize = 0;

	//patch information initialize
	patchWidth = 8;
	patchHeight = 8;
	patchSize = patchWidth*patchHeight;
	gapx = static_cast<int>(0.5*patchWidth);
	gapy = static_cast<int>(0.5*patchHeight);
}

ShadowRemovalImg::~ShadowRemovalImg()
{

}

pixel2D *** ShadowRemovalImg::memAlloc(int w, int h, int patchNum)
{
	pixel2D *** p = new pixel2D **[patchNum];
	for (int i = 0; i < patchNum; ++i)
	{
		p[i] = new pixel2D *[h];
		for (int j = 0; j < h; ++j)
		{
			p[i][j] = new pixel2D[w];
		}
	}
	return p;
}

void ShadowRemovalImg::memFree(pixel2D **** p)
{
	if (!p || !*p)
		return;
	for (int i = 0; i < patchNumber; ++i)
	{
		for (int j = 0; j < patchHeight; ++j)
		{
			delete[](*p)[i][j];
			(*p)[i][j] = NULL;
		}
		delete[](*p)[i];
		(*p)[i] = NULL;
	}
	delete[](*p);
	(*p) = NULL;
}

void ShadowRemovalImg::clear()
{

}

void ShadowRemovalImg::imageDecompose()
{
	if (inputImg.empty())
	{
		cout << "the input image is null!" << endl;
		return;
	}

	if (shadowImg.empty())
	{
		cout << "the shadow mask is null!" << endl;
		return;
	}
	patchImg = memAlloc(patchWidth, patchHeight, patchNumber);
	int patchIdx = 0;
	logOperator();

	for (int i = 0; i <= imHeight - patchHeight; i += patchHeight)
	{
		for (int j = 0; j <= imWidth - patchWidth; j += patchWidth)
		{
			for (int ii = i; ii < i + patchHeight; ++ii)
			{
				for (int jj = j; jj < j + patchWidth; ++jj)
				{
					patchImg[patchIdx][ii - i][jj - j].b = inputImg.at<Vec3b>(ii, jj)[0];
					patchImg[patchIdx][ii - i][jj - j].g = inputImg.at<Vec3b>(ii, jj)[1];
					patchImg[patchIdx][ii - i][jj - j].r = inputImg.at<Vec3b>(ii, jj)[2];

					/*patchImg[patchIdx][ii - i][jj - j].lbpB = getImprovedLbpValue(ii, jj)[0];
					patchImg[patchIdx][ii - i][jj - j].lbpB = getImprovedLbpValue(ii, jj)[1];
					patchImg[patchIdx][ii - i][jj - j].lbpB = getImprovedLbpValue(ii, jj)[2];*/

					patchImg[patchIdx][ii - i][jj - j].lbpB = getLbpValue(ii, jj)[0];
					patchImg[patchIdx][ii - i][jj - j].lbpB = getLbpValue(ii, jj)[1];
					patchImg[patchIdx][ii - i][jj - j].lbpB = getLbpValue(ii, jj)[2];

					patchImg[patchIdx][ii - i][jj - j].alpha = shadowImg.at<uchar>(ii, jj);

					patchImg[patchIdx][ii - i][jj - j].x = jj;
					patchImg[patchIdx][ii - i][jj - j].y = ii;
					//patchImg[patchIdx][ii - i][jj - j].pf.chromaticity = chromaticity.at<uchar>(ii, jj);

					//�洢��������,ii�������꣬jj�Ǻ�����
					patchImg[patchIdx][ii - i][jj - j].pf.intensity = (inputImg.at<Vec3b>(ii, jj)[2] * 299 + inputImg.at<Vec3b>(ii, jj)[1] * 587 + inputImg.at<Vec3b>(ii, jj)[0] * 114 + 500) / 1000;//intensity
					patchImg[patchIdx][ii - i][jj - j].pf.chromaticity = chromaticity.at<uchar>(ii, jj);//chromaticity

					if (jj == imWidth - 1) patchImg[patchIdx][ii - i][jj - j].pf.firstDevX = (inputImg.at<Vec3b>(ii, jj)[0] + inputImg.at<Vec3b>(ii, jj)[1] + inputImg.at<Vec3b>(ii, jj)[2]) / 3.0f;
					else patchImg[patchIdx][ii - i][jj - j].pf.firstDevX = (inputImg.at<Vec3b>(ii, jj + 1)[0] - inputImg.at<Vec3b>(ii, jj)[0] + inputImg.at<Vec3b>(ii, jj + 1)[1] - inputImg.at<Vec3b>(ii, jj)[1] + inputImg.at<Vec3b>(ii, jj + 1)[2] - inputImg.at<Vec3b>(ii, jj)[2]) / 3.0f;//x�����ϵ�һ�׵���

					if (ii == imHeight - 1) patchImg[patchIdx][ii - i][jj - j].pf.firstDevY = (inputImg.at<Vec3b>(ii, jj)[0] + inputImg.at<Vec3b>(ii, jj)[1] + inputImg.at<Vec3b>(ii, jj)[2]) / 3.0f;
					else patchImg[patchIdx][ii - i][jj - j].pf.firstDevY = (inputImg.at<Vec3b>(ii + 1, jj)[0] - inputImg.at<Vec3b>(ii, jj)[0] + inputImg.at<Vec3b>(ii + 1, jj)[1] - inputImg.at<Vec3b>(ii, jj)[1] + inputImg.at<Vec3b>(ii + 1, jj)[2] - inputImg.at<Vec3b>(ii, jj)[2]) / 3.0f;//y���������һ�׵���

					if (jj == 0 || jj == imWidth - 1) patchImg[patchIdx][ii - i][jj - j].pf.secondDevX = (inputImg.at<Vec3b>(ii, jj)[0] + inputImg.at<Vec3b>(ii, jj)[1] + inputImg.at<Vec3b>(ii, jj)[2]) / 3.0f;
					else patchImg[patchIdx][ii - i][jj - j].pf.secondDevX = (inputImg.at<Vec3b>(ii, jj + 1)[0] + inputImg.at<Vec3b>(ii, jj - 1)[0] - 2 * inputImg.at<Vec3b>(ii, jj)[0] + inputImg.at<Vec3b>(ii, jj + 1)[1] + inputImg.at<Vec3b>(ii, jj - 1)[1] - 2 * inputImg.at<Vec3b>(ii, jj)[1] + inputImg.at<Vec3b>(ii, jj + 1)[2] + inputImg.at<Vec3b>(ii, jj - 1)[2] - 2 * inputImg.at<Vec3b>(ii, jj)[2]) / 3.0f;//x�����ϵĶ��׵���

					if (ii == 0 || ii == imHeight - 1) patchImg[patchIdx][ii - i][jj - j].pf.secondDevY = (inputImg.at<Vec3b>(ii, jj)[0] + inputImg.at<Vec3b>(ii, jj)[1] + inputImg.at<Vec3b>(ii, jj)[2]) / 3.0f;
					else patchImg[patchIdx][ii - i][jj - j].pf.secondDevY = (inputImg.at<Vec3b>(ii + 1, jj)[0] + inputImg.at<Vec3b>(ii - 1, jj)[0] - 2 * inputImg.at<Vec3b>(ii, jj)[0] + inputImg.at<Vec3b>(ii + 1, jj)[1] + inputImg.at<Vec3b>(ii - 1, jj)[1] - 2 * inputImg.at<Vec3b>(ii, jj)[1] + inputImg.at<Vec3b>(ii + 1, jj)[2] + inputImg.at<Vec3b>(ii - 1, jj)[2] - 2 * inputImg.at<Vec3b>(ii, jj)[2]) / 3.0f;//y�����ϵĶ��׵���

				}
			}
			++patchIdx;
		}
	}
	cout << "patch numbers: " << patchIdx << endl;
}

void ShadowRemovalImg::imageOverlapDecompose()
{
	if (inputImg.empty())
	{
		cout << "the input image is null!" << endl;
		return;
	}
	if (shadowImg.empty())
	{
		cout << "the detected shadow image is null!" << endl;
		return;
	}
	patchImg = memAlloc(patchWidth, patchHeight, patchNumber);
	int patchIdx = 0;
	logOperator();
	for (int i = 0; i <= imHeight - patchHeight; i += gapy)
	{
		for (int j = 0; j <= imWidth - patchWidth; j += gapx)
		{
			for (int ii = i; ii < i + patchHeight; ++ii)
			{
				for (int jj = j; jj < j + patchWidth; ++jj)
				{
					//�洢rgbֵ
					patchImg[patchIdx][ii - i][jj - j].b = inputImg.at<Vec3b>(ii, jj)[0];
					patchImg[patchIdx][ii - i][jj - j].g = inputImg.at<Vec3b>(ii, jj)[1];
					patchImg[patchIdx][ii - i][jj - j].r = inputImg.at<Vec3b>(ii, jj)[2];
					/*patchImg[patchIdx][ii - i][jj - j].lbpB = getLbpValue(ii,jj)[0];
					patchImg[patchIdx][ii - i][jj - j].lbpB = getLbpValue(ii, jj)[1];
					patchImg[patchIdx][ii - i][jj - j].lbpB = getLbpValue(ii, jj)[2];*/

					//�洢lbpֵ
					patchImg[patchIdx][ii - i][jj - j].lbpB = getImprovedLbpValue(ii, jj)[0];
					patchImg[patchIdx][ii - i][jj - j].lbpB = getImprovedLbpValue(ii, jj)[1];
					patchImg[patchIdx][ii - i][jj - j].lbpB = getImprovedLbpValue(ii, jj)[2];
					//�洢��Ӱmatteֵ
					patchImg[patchIdx][ii - i][jj - j].alpha = shadowImg.at<uchar>(ii, jj);
					//�洢��ά����ֵ
					patchImg[patchIdx][ii - i][jj - j].x = jj;
					patchImg[patchIdx][ii - i][jj - j].y = ii;

					//�洢��������,ii�������꣬jj�Ǻ�����
					patchImg[patchIdx][ii - i][jj - j].pf.intensity = (inputImg.at<Vec3b>(ii, jj)[2] * 299 + inputImg.at<Vec3b>(ii, jj)[1] * 587 + inputImg.at<Vec3b>(ii, jj)[0] * 114 + 500) / 1000;//intensity
					patchImg[patchIdx][ii - i][jj - j].pf.chromaticity = chromaticity.at<uchar>(ii, jj);//chromaticity

					if (jj == imWidth - 1) patchImg[patchIdx][ii - i][jj - j].pf.firstDevX = (inputImg.at<Vec3b>(ii, jj)[0] + inputImg.at<Vec3b>(ii, jj)[1] + inputImg.at<Vec3b>(ii, jj)[2]) / 3.0f;
					else patchImg[patchIdx][ii - i][jj - j].pf.firstDevX = (inputImg.at<Vec3b>(ii, jj + 1)[0] - inputImg.at<Vec3b>(ii, jj)[0] + inputImg.at<Vec3b>(ii, jj + 1)[1] - inputImg.at<Vec3b>(ii, jj)[1] + inputImg.at<Vec3b>(ii, jj + 1)[2] - inputImg.at<Vec3b>(ii, jj)[2]) / 3.0f;//x�����ϵ�һ�׵���

					if (ii == imHeight - 1) patchImg[patchIdx][ii - i][jj - j].pf.firstDevY = (inputImg.at<Vec3b>(ii, jj)[0] + inputImg.at<Vec3b>(ii, jj)[1] + inputImg.at<Vec3b>(ii, jj)[2]) / 3.0f;
					else patchImg[patchIdx][ii - i][jj - j].pf.firstDevY = (inputImg.at<Vec3b>(ii + 1, jj)[0] - inputImg.at<Vec3b>(ii, jj)[0] + inputImg.at<Vec3b>(ii + 1, jj)[1] - inputImg.at<Vec3b>(ii, jj)[1] + inputImg.at<Vec3b>(ii + 1, jj)[2] - inputImg.at<Vec3b>(ii, jj)[2]) / 3.0f;//y���������һ�׵���

					if (jj == 0 || jj == imWidth - 1) patchImg[patchIdx][ii - i][jj - j].pf.secondDevX = (inputImg.at<Vec3b>(ii, jj)[0] + inputImg.at<Vec3b>(ii, jj)[1] + inputImg.at<Vec3b>(ii, jj)[2]) / 3.0f;
					else patchImg[patchIdx][ii - i][jj - j].pf.secondDevX = (inputImg.at<Vec3b>(ii, jj + 1)[0] + inputImg.at<Vec3b>(ii, jj - 1)[0] - 2 * inputImg.at<Vec3b>(ii, jj)[0] + inputImg.at<Vec3b>(ii, jj + 1)[1] + inputImg.at<Vec3b>(ii, jj - 1)[1] - 2 * inputImg.at<Vec3b>(ii, jj)[1] + inputImg.at<Vec3b>(ii, jj + 1)[2] + inputImg.at<Vec3b>(ii, jj - 1)[2] - 2 * inputImg.at<Vec3b>(ii, jj)[2]) / 3.0f;//x�����ϵĶ��׵���

					if (ii == 0 || ii == imHeight - 1) patchImg[patchIdx][ii - i][jj - j].pf.secondDevY = (inputImg.at<Vec3b>(ii, jj)[0] + inputImg.at<Vec3b>(ii, jj)[1] + inputImg.at<Vec3b>(ii, jj)[2]) / 3.0f;
					else patchImg[patchIdx][ii - i][jj - j].pf.secondDevY = (inputImg.at<Vec3b>(ii + 1, jj)[0] + inputImg.at<Vec3b>(ii - 1, jj)[0] - 2 * inputImg.at<Vec3b>(ii, jj)[0] + inputImg.at<Vec3b>(ii + 1, jj)[1] + inputImg.at<Vec3b>(ii - 1, jj)[1] - 2 * inputImg.at<Vec3b>(ii, jj)[1] + inputImg.at<Vec3b>(ii + 1, jj)[2] + inputImg.at<Vec3b>(ii - 1, jj)[2] - 2 * inputImg.at<Vec3b>(ii, jj)[2]) / 3.0f;//y�����ϵĶ��׵���
				}
			}
			++patchIdx;
		}
	}
	cout << "patch numbers: " << patchIdx << endl;
}

//�����������굽patch���ĵľ���ֵ
double ShadowRemovalImg::getSpatialDistance(int x1, int y1, int x2, int y2) const
{
	if (x1 < 0 || x2 <0 || y1 <0 || y2 <0 || x1 >= imWidth || x2 >= imWidth || y1 >= imHeight || y2 >= imHeight)
		return 0;
	return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

double ShadowRemovalImg::maxDistance(double dis1, double dis2, double dis3, double dis4)
{
	double maxDis = dis1;
	maxDis = maxDis < dis2 ? dis2 : maxDis;
	maxDis = maxDis < dis3 ? dis3 : maxDis;
	maxDis = maxDis < dis4 ? dis4 : maxDis;
	return maxDis;
}

/************************************************************************
�ؼ�����֮�ģ� �ֲ�������ÿ��patch(cuboids)��������ϣ������·����ɹ����ǡ�
1. ֱ����ĳһ��patch(cuboid)��ֵ������ȡ�Ḳ��ԭ���������������
2. �򵥵�����ƽ��ֵ��Ҳ����
3. �������ص�patch(cuboid)���ľ�������ļ�Ȩƽ����Ŀǰʵ������̫����
4. ��Ҫ����������Ч�ķ�����
************************************************************************/
void  ShadowRemovalImg::imagePatchesRecompose()
{
	if (patchImg == NULL)
	{
		cout << "the patch is null, make sure you overlapped decopose the input image!" << endl;
		return;
	}
	//ÿһ���ж��ٸ�patch��ÿһ���ж��ٸ�patch��Ϊ�˼��㷽�㣬patch�ĸ���ÿ��ÿ��ǡ�ö�������
	int patchesPerRow = (imWidth - gapx) / gapx;
	int patchesPerCol = (imHeight - gapy) / gapy;

	//��ʹ�þ����Ȩƽ����ʱ�򣬻��õ���Щ����ֵ���趨����
	double w[4], wMax, wTotal = 0.0;
	int index1, index2, index3, index4;

	//ֻ��Ҫ������Ӱ���������ֵ������
	for (int i = 0; i < imHeight; ++i)
	{
		for (int j = 0; j < imWidth; ++j)
		{
			if (i >= gapy && i < imHeight - gapy && j >= gapx && j < imWidth - gapx)
			{
				if (shadowImg.at<uchar>(i, j) != 0)
				{
					//1. ����ƽ��ֵ
					if (compositeWays == 0)
					{
						shadowlessImg.at<Vec3b>(i, j)[0] = (patchImg[(i / gapy - 1)*patchesPerRow + j / gapx - 1][i - (i / gapy - 1)*gapy][j - (j / gapx - 1)*gapx].b + patchImg[(i / gapy - 1)*patchesPerRow + j / gapx][i - (i / gapy - 1)*gapy][j - j / gapx*gapx].b
							+ patchImg[i / gapy*patchesPerRow + j / gapx - 1][i - i / gapy*gapy][j - (j / gapx - 1)*gapx].b + patchImg[i / gapy*patchesPerRow + j / gapx][i - i / gapy*gapy][j - j / gapx*gapx].b) / 4;
						shadowlessImg.at<Vec3b>(i, j)[1] = (patchImg[(i / gapy - 1)*patchesPerRow + j / gapx - 1][i - (i / gapy - 1)*gapy][j - (j / gapx - 1)*gapx].g + patchImg[(i / gapy - 1)*patchesPerRow + j / gapx][i - (i / gapy - 1)*gapy][j - j / gapx*gapx].g
							+ patchImg[i / gapy*patchesPerRow + j / gapx - 1][i - i / gapy*gapy][j - (j / gapx - 1)*gapx].g + patchImg[i / gapy*patchesPerRow + j / gapx][i - i / gapy*gapy][j - j / gapx*gapx].g) / 4;
						shadowlessImg.at<Vec3b>(i, j)[2] = (patchImg[(i / gapy - 1)*patchesPerRow + j / gapx - 1][i - (i / gapy - 1)*gapy][j - (j / gapx - 1)*gapx].r + patchImg[(i / gapy - 1)*patchesPerRow + j / gapx][i - (i / gapy - 1)*gapy][j - j / gapx*gapx].r
							+ patchImg[i / gapy*patchesPerRow + j / gapx - 1][i - i / gapy*gapy][j - (j / gapx - 1)*gapx].r + patchImg[i / gapy*patchesPerRow + j / gapx][i - i / gapy*gapy][j - j / gapx*gapx].r) / 4;
					}

					//2.�����Ȩƽ��
					if (compositeWays == 1)
					{
						index1 = (i / gapy - 1)*patchesPerRow + j / gapx - 1;
						index2 = (i / gapy - 1)*patchesPerRow + j / gapx;
						index3 = i / gapy*patchesPerRow + j / gapx - 1;
						index4 = i / gapy*patchesPerRow + j / gapx;

						w[0] = getSpatialDistance(j, i, patchImg[index1][patchHeight / 2][patchWidth / 2].x, patchImg[index1][patchHeight / 2][patchWidth / 2].y);
						w[1] = getSpatialDistance(j, i, patchImg[index2][patchHeight / 2][patchWidth / 2].x, patchImg[index2][patchHeight / 2][patchWidth / 2].y);
						w[2] = getSpatialDistance(j, i, patchImg[index3][patchHeight / 2][patchWidth / 2].x, patchImg[index3][patchHeight / 2][patchWidth / 2].y);
						w[3] = getSpatialDistance(j, i, patchImg[index4][patchHeight / 2][patchWidth / 2].x, patchImg[index4][patchHeight / 2][patchWidth / 2].y);

						for (int k = 0; k < 4; ++k)
						{
							wTotal += w[k];
						}

						for (int k = 0; k < 4; ++k)
						{
							w[k] = 1 - w[k] / wTotal;
						}

						shadowlessImg.at<Vec3b>(i, j)[0] = static_cast<uchar>((w[0] * patchImg[(i / gapy - 1)*patchesPerRow + j / gapx - 1][i - (i / gapy - 1)*gapy][j - (j / gapx - 1)*gapx].b + w[1] * patchImg[(i / gapy - 1)*patchesPerRow + j / gapx][i - (i / gapy - 1)*gapy][j - j / gapx*gapx].b
							+ w[2] * patchImg[i / gapy*patchesPerRow + j / gapx - 1][i - i / gapy*gapy][j - (j / gapx - 1)*gapx].b + w[3] * patchImg[i / gapy*patchesPerRow + j / gapx][i - i / gapy*gapy][j - j / gapx*gapx].b));
						shadowlessImg.at<Vec3b>(i, j)[1] = static_cast<uchar>((w[0] * patchImg[(i / gapy - 1)*patchesPerRow + j / gapx - 1][i - (i / gapy - 1)*gapy][j - (j / gapx - 1)*gapx].g + w[1] * patchImg[(i / gapy - 1)*patchesPerRow + j / gapx][i - (i / gapy - 1)*gapy][j - j / gapx*gapx].g
							+ w[2] * patchImg[i / gapy*patchesPerRow + j / gapx - 1][i - i / gapy*gapy][j - (j / gapx - 1)*gapx].g + w[3] * patchImg[i / gapy*patchesPerRow + j / gapx][i - i / gapy*gapy][j - j / gapx*gapx].g));
						shadowlessImg.at<Vec3b>(i, j)[2] = static_cast<uchar>((w[0] * patchImg[(i / gapy - 1)*patchesPerRow + j / gapx - 1][i - (i / gapy - 1)*gapy][j - (j / gapx - 1)*gapx].r + w[1] * patchImg[(i / gapy - 1)*patchesPerRow + j / gapx][i - (i / gapy - 1)*gapy][j - j / gapx*gapx].r
							+ w[2] * patchImg[i / gapy*patchesPerRow + j / gapx - 1][i - i / gapy*gapy][j - (j / gapx - 1)*gapx].r + w[3] * patchImg[i / gapy*patchesPerRow + j / gapx][i - i / gapy*gapy][j - j / gapx*gapx].r));
					}
				}
				continue;
			}


			//�ϱ���  done!
			if (i >= 0 && i < gapy && j >= gapx && j < imWidth - gapx)
			{
				if (shadowImg.at<uchar>(i, j) != 0)
				{
					//1.����ƽ��
					if (compositeWays == 0)
					{
						shadowlessImg.at<Vec3b>(i, j)[0] = (patchImg[i / gapy * patchesPerRow + j / gapx - 1][i][j - (j / gapx - 1)*gapx].b + patchImg[i / gapy * patchesPerRow + j / gapx][i][j - (j / gapx)*gapx].b) / 2;
						shadowlessImg.at<Vec3b>(i, j)[1] = (patchImg[i / gapy * patchesPerRow + j / gapx - 1][i][j - (j / gapx - 1)*gapx].g + patchImg[i / gapy * patchesPerRow + j / gapx][i][j - (j / gapx)*gapx].g) / 2;
						shadowlessImg.at<Vec3b>(i, j)[2] = (patchImg[i / gapy * patchesPerRow + j / gapx - 1][i][j - (j / gapx - 1)*gapx].r + patchImg[i / gapy * patchesPerRow + j / gapx][i][j - (j / gapx)*gapx].r) / 2;
					}

					//2.�����Ȩƽ��
					if (compositeWays == 1)
					{
						index1 = i / gapy * patchesPerRow + j / gapx - 1;
						index2 = i / gapy * patchesPerRow + j / gapx;

						w[0] = getSpatialDistance(j, i, patchImg[index1][patchHeight / 2][patchWidth / 2].x, patchImg[index1][patchHeight / 2][patchWidth / 2].y);
						w[1] = getSpatialDistance(j, i, patchImg[index2][patchHeight / 2][patchWidth / 2].x, patchImg[index2][patchHeight / 2][patchWidth / 2].y);

						wMax = maxValued(w, 2);

						for (int k = 0; k < 2; ++k)
						{
							w[k] = 1 - w[k] / wMax;
							wTotal += w[k];
						}
						shadowlessImg.at<Vec3b>(i, j)[0] = static_cast<uchar>((w[0] * patchImg[i / gapy * patchesPerRow + j / gapx - 1][i][j - (j / gapx - 1)*gapx].b + w[1] * patchImg[i / gapy * patchesPerRow + j / gapx][i][j - (j / gapx)*gapx].b) / wTotal);
						shadowlessImg.at<Vec3b>(i, j)[1] = static_cast<uchar>((w[0] * patchImg[i / gapy * patchesPerRow + j / gapx - 1][i][j - (j / gapx - 1)*gapx].g + w[1] * patchImg[i / gapy * patchesPerRow + j / gapx][i][j - (j / gapx)*gapx].g) / wTotal);
						shadowlessImg.at<Vec3b>(i, j)[2] = static_cast<uchar>((w[0] * patchImg[i / gapy * patchesPerRow + j / gapx - 1][i][j - (j / gapx - 1)*gapx].r + w[1] * patchImg[i / gapy * patchesPerRow + j / gapx][i][j - (j / gapx)*gapx].r) / wTotal);
					}
				}
				continue;

			}

			//����� done!
			if (i >= gapy && i < imHeight - gapy && j >= 0 && j < gapx)
			{
				if (shadowImg.at<uchar>(i, j) != 0)
				{
					//1.����ƽ��
					if (compositeWays == 0)
					{
						shadowlessImg.at<Vec3b>(i, j)[0] = (patchImg[(i / gapy - 1) * patchesPerRow + j / gapx][i - (i / gapy - 1)*gapy][j].b + patchImg[i / gapy * patchesPerRow + j / gapx][i - i / gapy * gapy][j].b) / 2;
						shadowlessImg.at<Vec3b>(i, j)[1] = (patchImg[(i / gapy - 1) * patchesPerRow + j / gapx][i - (i / gapy - 1)*gapy][j].g + patchImg[i / gapy * patchesPerRow + j / gapx][i - i / gapy * gapy][j].g) / 2;
						shadowlessImg.at<Vec3b>(i, j)[2] = (patchImg[(i / gapy - 1) * patchesPerRow + j / gapx][i - (i / gapy - 1)*gapy][j].r + patchImg[i / gapy * patchesPerRow + j / gapx][i - i / gapy * gapy][j].r) / 2;
					}

					//2.�����Ȩƽ��
					if (compositeWays == 1)
					{
						index1 = (i / gapy - 1) * patchesPerRow + j / gapx;
						index2 = i / gapy * patchesPerRow + j / gapx;

						w[0] = getSpatialDistance(j, i, patchImg[index1][patchHeight / 2][patchWidth / 2].x, patchImg[index1][patchHeight / 2][patchWidth / 2].y);
						w[1] = getSpatialDistance(j, i, patchImg[index2][patchHeight / 2][patchWidth / 2].x, patchImg[index2][patchHeight / 2][patchWidth / 2].y);

						wMax = maxValued(w, 2);

						for (int k = 0; k < 2; ++k)
						{
							w[k] = 1 - w[k] / wMax;
							wTotal += w[k];
						}

						shadowlessImg.at<Vec3b>(i, j)[0] = static_cast<uchar>((w[0] * patchImg[(i / gapy - 1) * patchesPerRow + j / gapx][i - (i / gapy - 1)*gapy][j].b + w[1] * patchImg[i / gapy * patchesPerRow + j / gapx][i - i / gapy * gapy][j].b) / wTotal);
						shadowlessImg.at<Vec3b>(i, j)[1] = static_cast<uchar>((w[0] * patchImg[(i / gapy - 1) * patchesPerRow + j / gapx][i - (i / gapy - 1)*gapy][j].g + w[1] * patchImg[i / gapy * patchesPerRow + j / gapx][i - i / gapy * gapy][j].g) / wTotal);
						shadowlessImg.at<Vec3b>(i, j)[2] = static_cast<uchar>((w[0] * patchImg[(i / gapy - 1) * patchesPerRow + j / gapx][i - (i / gapy - 1)*gapy][j].r + w[1] * patchImg[i / gapy * patchesPerRow + j / gapx][i - i / gapy * gapy][j].r) / wTotal);
					}
				}
				continue;
			}

			//�±��� done!
			if (i >= imHeight - gapy && i < imHeight && j >= gapx && j < imWidth - gapx)
			{
				if (shadowImg.at<uchar>(i, j) != 0)
				{
					//1.����ƽ��
					if (compositeWays == 0)
					{
						shadowlessImg.at<Vec3b>(i, j)[0] = (patchImg[(i / gapy - 1) * patchesPerRow + j / gapx - 1][i - (i / gapy - 1)*gapy][j - (j / gapx - 1) * gapx].b + patchImg[(i / gapy - 1) * patchesPerRow + j / gapx][i - (i / gapy - 1) * gapy][j - j / gapx * gapx].b) / 2;
						shadowlessImg.at<Vec3b>(i, j)[1] = (patchImg[(i / gapy - 1) * patchesPerRow + j / gapx - 1][i - (i / gapy - 1)*gapy][j - (j / gapx - 1) * gapx].g + patchImg[(i / gapy - 1) * patchesPerRow + j / gapx][i - (i / gapy - 1) * gapy][j - j / gapx * gapx].g) / 2;
						shadowlessImg.at<Vec3b>(i, j)[2] = (patchImg[(i / gapy - 1) * patchesPerRow + j / gapx - 1][i - (i / gapy - 1)*gapy][j - (j / gapx - 1) * gapx].r + patchImg[(i / gapy - 1) * patchesPerRow + j / gapx][i - (i / gapy - 1) * gapy][j - j / gapx * gapx].r) / 2;
					}

					//2.�����Ȩƽ��
					if (compositeWays == 1)
					{
						index1 = (i / gapy - 1) * patchesPerRow + j / gapx - 1;
						index2 = (i / gapy - 1) * patchesPerRow + j / gapx;

						w[0] = getSpatialDistance(j, i, patchImg[index1][patchHeight / 2][patchWidth / 2].x, patchImg[index1][patchHeight / 2][patchWidth / 2].y);
						w[1] = getSpatialDistance(j, i, patchImg[index2][patchHeight / 2][patchWidth / 2].x, patchImg[index2][patchHeight / 2][patchWidth / 2].y);

						wMax = maxValued(w, 2);

						for (int k = 0; k < 2; ++k)
						{
							w[k] = 1 - w[k] / wMax;
							wTotal += w[k];
						}

						shadowlessImg.at<Vec3b>(i, j)[0] = static_cast<uchar>((w[0] * patchImg[(i / gapy - 1) * patchesPerRow + j / gapx - 1][i - (i / gapy - 1)*gapy][j - (j / gapx - 1) * gapx].b + w[1] * patchImg[(i / gapy - 1) * patchesPerRow + j / gapx][i - (i / gapy - 1) * gapy][j - j / gapx * gapx].b) / wTotal);
						shadowlessImg.at<Vec3b>(i, j)[1] = static_cast<uchar>((w[0] * patchImg[(i / gapy - 1) * patchesPerRow + j / gapx - 1][i - (i / gapy - 1)*gapy][j - (j / gapx - 1) * gapx].g + w[1] * patchImg[(i / gapy - 1) * patchesPerRow + j / gapx][i - (i / gapy - 1) * gapy][j - j / gapx * gapx].g) / wTotal);
						shadowlessImg.at<Vec3b>(i, j)[2] = static_cast<uchar>((w[0] * patchImg[(i / gapy - 1) * patchesPerRow + j / gapx - 1][i - (i / gapy - 1)*gapy][j - (j / gapx - 1) * gapx].r + w[1] * patchImg[(i / gapy - 1) * patchesPerRow + j / gapx][i - (i / gapy - 1) * gapy][j - j / gapx * gapx].r) / wTotal);
					}
				}
				continue;
			}

			//�ұ���(k done!
			if (i >= gapy && i < imHeight - gapy && j >= imWidth - gapx && j < imWidth)
			{
				if (shadowImg.at<uchar>(i, j) != 0)
				{
					//1.����ƽ��
					if (compositeWays == 0)
					{
						shadowlessImg.at<Vec3b>(i, j)[0] = (patchImg[(i / gapy - 1) * patchesPerRow + j / gapx - 1][i - (i / gapy - 1)*gapy][j - (j / gapx - 1) * gapx].b + patchImg[i / gapy * patchesPerRow + j / gapx - 1][i - i / gapy * gapy][j - (j / gapx - 1) * gapx].b) / 2;
						shadowlessImg.at<Vec3b>(i, j)[1] = (patchImg[(i / gapy - 1) * patchesPerRow + j / gapx - 1][i - (i / gapy - 1)*gapy][j - (j / gapx - 1) * gapx].g + patchImg[i / gapy * patchesPerRow + j / gapx - 1][i - i / gapy * gapy][j - (j / gapx - 1) * gapx].g) / 2;
						shadowlessImg.at<Vec3b>(i, j)[2] = (patchImg[(i / gapy - 1) * patchesPerRow + j / gapx - 1][i - (i / gapy - 1)*gapy][j - (j / gapx - 1) * gapx].r + patchImg[i / gapy * patchesPerRow + j / gapx - 1][i - i / gapy * gapy][j - (j / gapx - 1) * gapx].r) / 2;
					}

					//2.�����Ȩƽ��
					if (compositeWays == 1)
					{
						index1 = (i / gapy - 1) * patchesPerRow + j / gapx - 1;
						index2 = i / gapy * patchesPerRow + j / gapx - 1;

						w[0] = getSpatialDistance(j, i, patchImg[index1][patchHeight / 2][patchWidth / 2].x, patchImg[index1][patchHeight / 2][patchWidth / 2].y);
						w[1] = getSpatialDistance(j, i, patchImg[index2][patchHeight / 2][patchWidth / 2].x, patchImg[index2][patchHeight / 2][patchWidth / 2].y);

						wMax = maxValued(w, 2);

						for (int k = 0; k < 2; ++k)
						{
							w[k] = 1 - w[k] / wMax;
							wTotal += w[k];
						}
						shadowlessImg.at<Vec3b>(i, j)[0] = static_cast<uchar>((w[0] * patchImg[(i / gapy - 1) * patchesPerRow + j / gapx - 1][i - (i / gapy - 1)*gapy][j - (j / gapx - 1) * gapx].b + w[1] * patchImg[i / gapy * patchesPerRow + j / gapx - 1][i - i / gapy * gapy][j - (j / gapx - 1) * gapx].b) / wTotal);
						shadowlessImg.at<Vec3b>(i, j)[1] = static_cast<uchar>((w[0] * patchImg[(i / gapy - 1) * patchesPerRow + j / gapx - 1][i - (i / gapy - 1)*gapy][j - (j / gapx - 1) * gapx].g + w[1] * patchImg[i / gapy * patchesPerRow + j / gapx - 1][i - i / gapy * gapy][j - (j / gapx - 1) * gapx].g) / wTotal);
						shadowlessImg.at<Vec3b>(i, j)[2] = static_cast<uchar>((w[0] * patchImg[(i / gapy - 1) * patchesPerRow + j / gapx - 1][i - (i / gapy - 1)*gapy][j - (j / gapx - 1) * gapx].r + w[1] * patchImg[i / gapy * patchesPerRow + j / gapx - 1][i - i / gapy * gapy][j - (j / gapx - 1) * gapx].r) / wTotal);
					}
				}
				continue;
			}

			/*�����Χ������ֵֻ����һ��patch,����Ҫ��Ȩƽ��*/
			//���Ͻ�
			if (i >= 0 && i < gapy && j >= 0 && j < gapx)
			{
				if (shadowImg.at<uchar>(i, j) != 0)
				{
					//the first patch
					shadowlessImg.at<Vec3b>(i, j)[0] = patchImg[i / gapy * patchesPerRow + j / gapx][i][j].b;
					shadowlessImg.at<Vec3b>(i, j)[1] = patchImg[i / gapy * patchesPerRow + j / gapx][i][j].g;
					shadowlessImg.at<Vec3b>(i, j)[2] = patchImg[i / gapy * patchesPerRow + j / gapx][i][j].r;
				}
				continue;
			}
			//���Ͻ� done!
			if (i >= 0 && i < gapy && j < imWidth && j >= imWidth - gapx)
			{
				if (shadowImg.at<uchar>(i, j) != 0)
				{
					shadowlessImg.at<Vec3b>(i, j)[0] = patchImg[i / gapy * patchesPerRow + j / gapx - 1][i][j - (imWidth - patchWidth)].b;
					shadowlessImg.at<Vec3b>(i, j)[1] = patchImg[i / gapy * patchesPerRow + j / gapx - 1][i][j - (imWidth - patchWidth)].g;
					shadowlessImg.at<Vec3b>(i, j)[2] = patchImg[i / gapy * patchesPerRow + j / gapx - 1][i][j - (imWidth - patchWidth)].r;
				}
				continue;
			}
			//���½� done!
			if (i < imHeight && i >= imHeight - gapy && j >= 0 && j < gapx)
			{
				if (shadowImg.at<uchar>(i, j) != 0)
				{
					shadowlessImg.at<Vec3b>(i, j)[0] = patchImg[(i / gapy - 1) * patchesPerRow + j / gapx][i - (imHeight - patchHeight)][j].b;
					shadowlessImg.at<Vec3b>(i, j)[1] = patchImg[(i / gapy - 1) * patchesPerRow + j / gapx][i - (imHeight - patchHeight)][j].g;
					shadowlessImg.at<Vec3b>(i, j)[2] = patchImg[(i / gapy - 1) * patchesPerRow + j / gapx][i - (imHeight - patchHeight)][j].r;
				}
				continue;
			}
			//���½� done!
			if (i < imHeight && i >= imHeight - gapy && j < imWidth && j >= imWidth - gapx)
			{
				if (shadowImg.at<uchar>(i, j) != 0)
				{
					shadowlessImg.at<Vec3b>(i, j)[0] = patchImg[patchNumber - 1][i - (imHeight - patchHeight)][j - (imWidth - patchWidth)].b;
					shadowlessImg.at<Vec3b>(i, j)[1] = patchImg[patchNumber - 1][i - (imHeight - patchHeight)][j - (imWidth - patchWidth)].g;
					shadowlessImg.at<Vec3b>(i, j)[2] = patchImg[patchNumber - 1][i - (imHeight - patchHeight)][j - (imWidth - patchWidth)].r;
				}
			}
		}
	}
}


void ShadowRemovalImg::imageRecompose()
{
	if (patchImg == NULL)
	{
		cout << "the patch is null, make sure you overlapped decopose the input image!" << endl;
		return;
	}

	int patchesPerRow = imWidth / patchWidth;
	int patchesPerCol = imHeight / patchHeight;

	for (int i = 0; i < imHeight; ++i)
	{
		for (int j = 0; j < imWidth; ++j)
		{
			if (shadowImg.at<uchar>(i, j) != 0)
			{
				shadowlessImg.at<Vec3b>(i, j)[0] = patchImg[i / patchHeight*patchesPerRow + j / patchWidth][i - i / patchHeight*patchHeight][j - j / patchWidth*patchWidth].b;
				shadowlessImg.at<Vec3b>(i, j)[1] = patchImg[i / patchHeight*patchesPerRow + j / patchWidth][i - i / patchHeight*patchHeight][j - j / patchWidth*patchWidth].g;
				shadowlessImg.at<Vec3b>(i, j)[2] = patchImg[i / patchHeight*patchesPerRow + j / patchWidth][i - i / patchHeight*patchHeight][j - j / patchWidth*patchWidth].r;
			}
		}
	}
}

void ShadowRemovalImg::patchDecomposeTest()
{
	if (patchImg == NULL)
	{
		imageOverlapDecompose();
	}

	Mat temp(patchHeight, patchWidth, CV_8UC3, Scalar(0));
	vector<Mat> patches(100, temp);
	for (int i = 0; i < 100; ++i)
	{
		for (int j = 0; j < patchHeight; ++j)
		{
			for (int k = 0; k < patchWidth; ++k)
			{
				patches[i].at<Vec3b>(j, k)[0] = patchImg[i + 50][j][k].b;
				patches[i].at<Vec3b>(j, k)[1] = patchImg[i + 50][j][k].g;
				patches[i].at<Vec3b>(j, k)[2] = patchImg[i + 50][j][k].r;
			}
		}
	}

	char win[255];
	for (int i = 0; i < 5; ++i)
	{
		sprintf_s(win, "%03d", ++i);
		imshow(win, patches[i]);

	}
	waitKey(0);
}


//��������patchΪ��λ�ģ�������Ӱ�ͷ���Ӱpatch��ͬʱ����ÿ��patch������Э��������������
void ShadowRemovalImg::patchDivision2LitAndShadow()
{
	if (patchImg == NULL)
	{
		cout << "please overlap decompose the input image first!" << endl;
		return;
	}

	//��Ҫ��ǰ����һ��6*6��float���͵ľ���
	double ** C_R = new double *[6];
	for (int i = 0; i < 6; ++i)
	{
		C_R[i] = new double[6];
	}
	pixelFeature mean_u;

	for (int i = 0; i < patchNumber; ++i)
	{
		int shadowPixelsNum = 0;

		//��ֵ��Э�������ĳ�ʼ��
		mean_u.intensity = 0;
		mean_u.chromaticity = 0;
		mean_u.firstDevX = 0;
		mean_u.firstDevY = 0;
		mean_u.secondDevX = 0;
		mean_u.secondDevY = 0;

		for (int j = 0; j < 6; ++j)
		{
			for (int k = 0; k < 6; ++k)
			{
				C_R[j][k] = 0.0;
			}
		}

		for (int j = 0; j < patchHeight; ++j)
		{
			for (int k = 0; k < patchWidth; ++k)
			{
				if (patchImg[i][j][k].alpha != 0)
					++shadowPixelsNum;

				//�����patch�ľ�ֵ����������ƽ��ֵ
				mean_u.intensity += patchImg[i][j][k].pf.intensity;
				mean_u.chromaticity += patchImg[i][j][k].pf.chromaticity;
				mean_u.firstDevX += patchImg[i][j][k].pf.firstDevX;
				mean_u.firstDevY += patchImg[i][j][k].pf.firstDevY;
				mean_u.secondDevX += patchImg[i][j][k].pf.secondDevX;
				mean_u.secondDevY += patchImg[i][j][k].pf.secondDevY;
			}
		}
		//��ƽ��ֵ
		mean_u.intensity /= patchSize;
		mean_u.chromaticity /= patchSize;
		mean_u.firstDevX /= patchSize;
		mean_u.firstDevY /= patchSize;
		mean_u.secondDevX /= patchSize;
		mean_u.secondDevY /= patchSize;

		if ((float)shadowPixelsNum / patchSize >= 0.1)
			shadowPatchIndices.push_back(i);
		else
			litPatchIndices.push_back(i);

		//�����patch������Э�������
		for (int j = 0; j < patchHeight; ++j)
		{
			for (int k = 0; k < patchWidth; ++k)
			{
				//�����Ԫ��
				C_R[0][0] += (patchImg[i][j][k].pf.intensity - mean_u.intensity)*(patchImg[i][j][k].pf.intensity - mean_u.intensity);
				C_R[0][1] += (patchImg[i][j][k].pf.intensity - mean_u.intensity)*(patchImg[i][j][k].pf.chromaticity - mean_u.chromaticity);
				C_R[0][2] += (patchImg[i][j][k].pf.intensity - mean_u.intensity)*(patchImg[i][j][k].pf.firstDevX - mean_u.firstDevX);
				C_R[0][3] += (patchImg[i][j][k].pf.intensity - mean_u.intensity)*(patchImg[i][j][k].pf.firstDevY - mean_u.firstDevY);
				C_R[0][4] += (patchImg[i][j][k].pf.intensity - mean_u.intensity)*(patchImg[i][j][k].pf.secondDevX - mean_u.secondDevX);
				C_R[0][5] += (patchImg[i][j][k].pf.intensity - mean_u.intensity)*(patchImg[i][j][k].pf.secondDevY - mean_u.secondDevY);

				C_R[1][0] += (patchImg[i][j][k].pf.chromaticity - mean_u.chromaticity)*(patchImg[i][j][k].pf.intensity - mean_u.intensity);
				C_R[1][1] += (patchImg[i][j][k].pf.chromaticity - mean_u.chromaticity)*(patchImg[i][j][k].pf.chromaticity - mean_u.chromaticity);
				C_R[1][2] += (patchImg[i][j][k].pf.chromaticity - mean_u.chromaticity)*(patchImg[i][j][k].pf.firstDevX - mean_u.firstDevX);
				C_R[1][3] += (patchImg[i][j][k].pf.chromaticity - mean_u.chromaticity)*(patchImg[i][j][k].pf.firstDevY - mean_u.firstDevY);
				C_R[1][4] += (patchImg[i][j][k].pf.chromaticity - mean_u.chromaticity)*(patchImg[i][j][k].pf.secondDevX - mean_u.secondDevX);
				C_R[1][5] += (patchImg[i][j][k].pf.chromaticity - mean_u.chromaticity)*(patchImg[i][j][k].pf.secondDevY - mean_u.secondDevY);

				C_R[2][0] += (patchImg[i][j][k].pf.firstDevX - mean_u.firstDevX)*(patchImg[i][j][k].pf.intensity - mean_u.intensity);
				C_R[2][1] += (patchImg[i][j][k].pf.firstDevX - mean_u.firstDevX)*(patchImg[i][j][k].pf.chromaticity - mean_u.chromaticity);
				C_R[2][2] += (patchImg[i][j][k].pf.firstDevX - mean_u.firstDevX)*(patchImg[i][j][k].pf.firstDevX - mean_u.firstDevX);
				C_R[2][3] += (patchImg[i][j][k].pf.firstDevX - mean_u.firstDevX)*(patchImg[i][j][k].pf.firstDevY - mean_u.firstDevY);
				C_R[2][4] += (patchImg[i][j][k].pf.firstDevX - mean_u.firstDevX)*(patchImg[i][j][k].pf.secondDevX - mean_u.secondDevX);
				C_R[2][5] += (patchImg[i][j][k].pf.firstDevX - mean_u.firstDevX)*(patchImg[i][j][k].pf.secondDevY - mean_u.secondDevY);

				C_R[3][0] += (patchImg[i][j][k].pf.firstDevY - mean_u.firstDevY)*(patchImg[i][j][k].pf.intensity - mean_u.intensity);
				C_R[3][1] += (patchImg[i][j][k].pf.firstDevY - mean_u.firstDevY)*(patchImg[i][j][k].pf.chromaticity - mean_u.chromaticity);
				C_R[3][2] += (patchImg[i][j][k].pf.firstDevY - mean_u.firstDevY)*(patchImg[i][j][k].pf.firstDevX - mean_u.firstDevX);
				C_R[3][3] += (patchImg[i][j][k].pf.firstDevY - mean_u.firstDevY)*(patchImg[i][j][k].pf.firstDevY - mean_u.firstDevY);
				C_R[3][4] += (patchImg[i][j][k].pf.firstDevY - mean_u.firstDevY)*(patchImg[i][j][k].pf.secondDevX - mean_u.secondDevX);
				C_R[3][5] += (patchImg[i][j][k].pf.firstDevY - mean_u.firstDevY)*(patchImg[i][j][k].pf.secondDevY - mean_u.secondDevY);

				C_R[4][0] += (patchImg[i][j][k].pf.secondDevX - mean_u.secondDevX)*(patchImg[i][j][k].pf.intensity - mean_u.intensity);
				C_R[4][1] += (patchImg[i][j][k].pf.secondDevX - mean_u.secondDevX)*(patchImg[i][j][k].pf.chromaticity - mean_u.chromaticity);
				C_R[4][2] += (patchImg[i][j][k].pf.secondDevX - mean_u.secondDevX)*(patchImg[i][j][k].pf.firstDevX - mean_u.firstDevX);
				C_R[4][3] += (patchImg[i][j][k].pf.secondDevX - mean_u.secondDevX)*(patchImg[i][j][k].pf.firstDevY - mean_u.firstDevY);
				C_R[4][4] += (patchImg[i][j][k].pf.secondDevX - mean_u.secondDevX)*(patchImg[i][j][k].pf.secondDevX - mean_u.secondDevX);
				C_R[4][5] += (patchImg[i][j][k].pf.secondDevX - mean_u.secondDevX)*(patchImg[i][j][k].pf.secondDevY - mean_u.secondDevY);

				C_R[5][0] += (patchImg[i][j][k].pf.secondDevY - mean_u.secondDevY)*(patchImg[i][j][k].pf.intensity - mean_u.intensity);
				C_R[5][1] += (patchImg[i][j][k].pf.secondDevY - mean_u.secondDevY)*(patchImg[i][j][k].pf.chromaticity - mean_u.chromaticity);
				C_R[5][2] += (patchImg[i][j][k].pf.secondDevY - mean_u.secondDevY)*(patchImg[i][j][k].pf.firstDevX - mean_u.firstDevX);
				C_R[5][3] += (patchImg[i][j][k].pf.secondDevY - mean_u.secondDevY)*(patchImg[i][j][k].pf.firstDevY - mean_u.firstDevY);
				C_R[5][4] += (patchImg[i][j][k].pf.secondDevY - mean_u.secondDevY)*(patchImg[i][j][k].pf.secondDevX - mean_u.secondDevX);
				C_R[5][5] += (patchImg[i][j][k].pf.secondDevY - mean_u.secondDevY)*(patchImg[i][j][k].pf.secondDevY - mean_u.secondDevY);
			}
		}

		//Ԫ�س���patch�ߴ��С
		for (int j = 0; j < 6; ++j)
		{
			for (int k = 0; k < 6; ++k)
			{
				C_R[j][k] /= (patchSize - 1);
			}
		}

		//�������˸�patch��Ӧ��Э���������������ķ˹���ֽ⽫����C_R�ֽ�������Ǿ���L�����Ӧ�Ĺ���ת�þ���LT
		Eigen::MatrixXd cov(6, 6);
		for (int j = 0; j < 6; ++j)
		{
			for (int k = 0; k < 6; ++k)
			{
				cov(j, k) = C_R[j][k];
			}
		}
		//ִ����ķ˹���ֽ�õ�L����,L��ά�Ⱥ�cov��һ����
		Eigen::MatrixXd L(cov.llt().matrixL());

		//���õ���L�ľ���Ԫ�ش洢������������ȥ
		std::vector<double> covector;
		//�Ƚ�mean_u��ֵѹ��ȥ

		covector.push_back(mean_u.intensity);
		covector.push_back(mean_u.chromaticity);
		covector.push_back(mean_u.firstDevX);
		covector.push_back(mean_u.firstDevY);
		covector.push_back(mean_u.secondDevX);
		covector.push_back(mean_u.secondDevY);

		//��L��Ԫ��ѹ��ȥ���Ը���6
		for (int j = 0; j < 6; ++j)
		{
			for (int k = 0; k < 6; ++k)
			{
				covector.push_back(sqrt(6.0)*L(j, k));
			}
		}

		//��L��Ԫ��ѹ��ȥ���Ը��ĸ���6
		for (int j = 0; j < 6; ++j)
		{
			for (int k = 0; k < 6; ++k)
			{
				covector.push_back(-sqrt(6.0)*L(j, k));
			}
		}

		//��patch�����Ͷ�Ӧ��patch������������map����
		patchCovVector.push_back(covector);
	}
	cout << "shadow patch numbers: " << shadowPatchIndices.size() << endl;
	cout << "lit patch numbers: " << litPatchIndices.size() << endl;

	cout << patchCovVector.size() << " " << patchCovVector[0].size() << endl;
}

double ShadowRemovalImg::getDisOfRGB(int pIndex1, int pIndex2) const
{
	double disRgb = 0.0;
	//the patch size is patch_w * patch_h
	for (int i = 0; i < patchHeight; ++i)
	{
		for (int j = 0; j < patchWidth; ++j)
		{
			double dis = sqrt((patchImg[pIndex1][i][j].b / 255.0 - patchImg[pIndex2][i][j].b / 255.0)*(patchImg[pIndex1][i][j].b / 255.0 - patchImg[pIndex2][i][j].b / 255.0)
				+ (patchImg[pIndex1][i][j].g / 255.0 - patchImg[pIndex2][i][j].g / 255.0)*(patchImg[pIndex1][i][j].g / 255.0 - patchImg[pIndex2][i][j].g / 255.0)
				+ (patchImg[pIndex1][i][j].r / 255.0 - patchImg[pIndex1][i][j].r / 255.0)*(patchImg[pIndex1][i][j].r / 255.0 - patchImg[pIndex2][i][j].r / 255.0));
			disRgb += dis;
		}
	}
	return disRgb / patchSize;
}

double ShadowRemovalImg::getDisOfChromaticity(int pIndex1, int pIndex2) const
{
	double disChrom = 0.0;

	double meanChrom1 = 0.0, meanChrom2 = 0.0;
	for (int i = 0; i < patchHeight; ++i)
	{
		for (int j = 0; j < patchWidth; ++j)
		{
			meanChrom1 += patchImg[pIndex1][i][j].pf.chromaticity;
			meanChrom2 += patchImg[pIndex2][i][j].pf.chromaticity;
			/*double dis = sqrt((patchImg[pIndex1][i][j].pf.chromaticity / 255.0 - patchImg[pIndex2][i][j].pf.chromaticity / 255.0)*(patchImg[pIndex1][i][j].pf.chromaticity / 255.0 - patchImg[pIndex2][i][j].pf.chromaticity / 255.0));
			disChrom += dis;*/
		}
	}

	meanChrom1 /= patchSize;
	meanChrom2 /= patchSize;

	disChrom = sqrt((meanChrom1 / 255.0 - meanChrom2 / 255.0)*(meanChrom1 / 255.0 - meanChrom2 / 255.0));
	return disChrom;
}

//distance of centroid points or sum of all the point
double ShadowRemovalImg::getDisOfPos(int pIndex1, int pIndex2) const
{
	//1. sum of all the point pos dis
	double disPos = 0.0;
	for (int i = 0; i < patchHeight; ++i)
	{
		for (int j = 0; j < patchWidth; ++j)
		{
			double dis = sqrt((patchImg[pIndex1][i][j].x / (double)imWidth - patchImg[pIndex2][i][j].x / (double)imWidth)  * (patchImg[pIndex1][i][j].x / (double)imWidth - patchImg[pIndex2][i][j].x / (double)imWidth)
				+ (patchImg[pIndex1][i][j].y / (double)imHeight - patchImg[pIndex2][i][j].y / (double)imHeight) * (patchImg[pIndex1][i][j].y / (double)imHeight - patchImg[pIndex2][i][j].y / (double)imHeight));
			disPos += dis;
		}
	}

	//2. dis of the patch centroid
	//disPos = sqrt((p1[patch_h / 2][patch_w / 2].x - p2[patch_h / 2][patch_w / 2].x) / (double)imWidth * (p1[patch_h / 2][patch_w / 2].x - p2[patch_h / 2][patch_w / 2].x) / (double)imWidth + (p1[patch_h / 2][patch_w / 2].y - p2[patch_h / 2][patch_w / 2].y) / (double)imHeight * (p1[patch_h / 2][patch_w / 2].y - p2[patch_h / 2][patch_w / 2].y) / (double)imHeight);
	disPos /= patchSize;
	return disPos;
}



//Preprocess the image using LOG filter
void ShadowRemovalImg::logOperator()
{
	if (inputImg.empty())
	{
		cout << "the input image is null!" << endl;
		return;
	}
	//�ȶ�ԭͼ����и�˹�˲�
	Mat gaussianFilterImg = inputImg.clone();
	GaussianBlur(inputImg, gaussianFilterImg, Size(5, 5), 0, 0);

	logFilteredImg.create(inputImg.size(), inputImg.type());
	double sigma = 1.0;//the variance

					   //���˲������ͼ������lapalacian����

					   //����laplacian����
					   /*0  1  0      1  1  1
					   1 -4  1      1 -8  1
					   0  1  0      1  1  1*/

	for (int i = 0; i < imHeight; ++i)
	{
		for (int j = 0; j < imWidth; ++j)
		{
			if (i == 0 || i == imHeight - 1 || j == 0 || j == imWidth - 1)
				logFilteredImg.at<Vec3b>(i, j) = gaussianFilterImg.at<Vec3b>(i, j);
			else
			{
				for (int c = 0; c < 3; ++c)
				{
					logFilteredImg.at<Vec3b>(i, j)[c] = gaussianFilterImg.at<Vec3b>(i - 1, j - 1)[c] + gaussianFilterImg.at<Vec3b>(i - 1, j)[c] + gaussianFilterImg.at<Vec3b>(i - 1, j + 1)[c]
						+ gaussianFilterImg.at<Vec3b>(i, j - 1)[c] + gaussianFilterImg.at<Vec3b>(i, j + 1)[c] + gaussianFilterImg.at<Vec3b>(i + 1, j - 1)[c] + gaussianFilterImg.at<Vec3b>(i + 1, j)[c]
						+ gaussianFilterImg.at<Vec3b>(i + 1, j + 1)[c] - 8 * gaussianFilterImg.at<Vec3b>(i, j)[c];
				}
			}
		}
	}

	//imageNormalize(&logFilteredImg);
	//imshow("log image", logFilteredImg);
	//imwrite("E:\\Video Shadow Removal\\results\\demo\\logFilter.bmp", logFilteredImg);
	//waitKey(0);
}

//��ƪ���������ķ������Ծ���log���Ӵ������ͼ�����lbp
void ShadowRemovalImg::lbpDescriptor()
{
	Mat inputLbp(imHeight, imWidth, CV_8UC3, Scalar(0));
	for (int i = 0; i < imHeight; i++)
	{
		for (int j = 0; j < imWidth; j++)
		{
			if (i == 0 || i == imHeight - 1 || j == 0 || j == imWidth - 1)
			{
				inputLbp.at<Vec3b>(i, j) = logFilteredImg.at<Vec3b>(i, j);
			}
			else
			{
				Vec3b tt = 0;

				//b channel
				int ttb = 0;
				uchar cb = logFilteredImg.at<Vec3b>(i, j)[0];
				if (logFilteredImg.at<Vec3b>(i - 1, j - 1)[0] > cb) { tt[0] += 1 << ttb; }
				ttb++;
				if (logFilteredImg.at<Vec3b>(i - 1, j)[0] > cb) { tt[0] += 1 << ttb; }
				ttb++;
				if (logFilteredImg.at<Vec3b>(i - 1, j + 1)[0] > cb) { tt[0] += 1 << ttb; }
				ttb++;
				if (logFilteredImg.at<Vec3b>(i, j + 1)[0] > cb) { tt[0] += 1 << ttb; }
				ttb++;
				if (logFilteredImg.at<Vec3b>(i + 1, j + 1)[0] > cb) { tt[0] += 1 << ttb; }
				ttb++;
				if (logFilteredImg.at<Vec3b>(i + 1, j)[0] > cb) { tt[0] += 1 << ttb; }
				ttb++;
				if (logFilteredImg.at<Vec3b>(i + 1, j - 1)[0] > cb) { tt[0] += 1 << ttb; }
				ttb++;
				if (logFilteredImg.at<Vec3b>(i, j - 1)[0] > cb) { tt[0] += 1 << ttb; }
				ttb++;

				//g channel
				int ttg = 0;
				uchar cg = logFilteredImg.at<Vec3b>(i, j)[1];
				if (logFilteredImg.at<Vec3b>(i - 1, j - 1)[1] > cg) { tt[1] += 1 << ttg; }
				ttg++;
				if (logFilteredImg.at<Vec3b>(i - 1, j)[1] > cg) { tt[1] += 1 << ttg; }
				ttg++;
				if (logFilteredImg.at<Vec3b>(i - 1, j + 1)[1] > cg) { tt[1] += 1 << ttg; }
				ttg++;
				if (logFilteredImg.at<Vec3b>(i, j + 1)[1] > cg) { tt[1] += 1 << ttg; }
				ttg++;
				if (logFilteredImg.at<Vec3b>(i + 1, j + 1)[1] > cg) { tt[1] += 1 << ttg; }
				ttg++;
				if (logFilteredImg.at<Vec3b>(i + 1, j)[1] > cg) { tt[1] += 1 << ttg; }
				ttg++;
				if (logFilteredImg.at<Vec3b>(i + 1, j - 1)[1] > cg) { tt[1] += 1 << ttg; }
				ttg++;
				if (logFilteredImg.at<Vec3b>(i, j - 1)[1] > cg) { tt[1] += 1 << ttg; }
				ttg++;

				//r channel
				int ttr = 0;
				uchar cr = logFilteredImg.at<Vec3b>(i, j)[2];
				if (logFilteredImg.at<Vec3b>(i - 1, j - 1)[2] > cr) { tt[2] += 1 << ttr; }
				ttr++;
				if (logFilteredImg.at<Vec3b>(i - 1, j)[2] > cr) { tt[2] += 1 << ttr; }
				ttr++;
				if (logFilteredImg.at<Vec3b>(i - 1, j + 1)[2] > cr) { tt[2] += 1 << ttr; }
				ttr++;
				if (logFilteredImg.at<Vec3b>(i, j + 1)[2] > cr) { tt[2] += 1 << ttr; }
				ttr++;
				if (logFilteredImg.at<Vec3b>(i + 1, j + 1)[2] > cr) { tt[2] += 1 << ttr; }
				ttr++;
				if (logFilteredImg.at<Vec3b>(i + 1, j)[2] > cr) { tt[2] += 1 << ttr; }
				ttr++;
				if (logFilteredImg.at<Vec3b>(i + 1, j - 1)[2] > cr) { tt[2] += 1 << ttr; }
				ttr++;
				if (logFilteredImg.at<Vec3b>(i, j - 1)[2] > cr) { tt[2] += 1 << ttr; }
				ttr++;

				inputLbp.at<Vec3b>(i, j) = tt;
			}

		}
	}

	imshow("orginal image", logFilteredImg);
	imageNormalize(&inputLbp);
	imshow("lbp image", inputLbp);
	waitKey(0);
}

void ShadowRemovalImg::imageNormalize(Mat * input)
{
	for (int i = 0; i < input->rows; ++i)
	{
		for (int j = 0; j < input->cols; ++j)
		{
			for (int c = 0; c < 3; ++c)
			{
				if (input->at<Vec3b>(i, j)[c] < 0)
					input->at<Vec3b>(i, j)[c] = 0;
				else if (input->at<Vec3b>(i, j)[c] > 255)
					input->at<Vec3b>(i, j)[c] = 255;
				else
					continue;
			}
		}
	}
}

void ShadowRemovalImg::patchPixelNormalization(int p)
{
	if (p < 0 || p >= patchNumber)
		return;

	for (int i = 0; i < patchHeight; ++i)
	{
		for (int j = 0; j < patchWidth; ++j)
		{
			if (patchImg[p][i][j].b < 0) patchImg[p][i][j].b = 0;
			if (patchImg[p][i][j].b > 255) patchImg[p][i][j].b = 255;

			if (patchImg[p][i][j].g < 0) patchImg[p][i][j].g = 0;
			if (patchImg[p][i][j].g > 255) patchImg[p][i][j].g = 255;

			if (patchImg[p][i][j].r < 0) patchImg[p][i][j].r = 0;
			if (patchImg[p][i][j].r > 255) patchImg[p][i][j].r = 255;
		}
	}
}

/************************************************************************
�ؼ�����֮һ�������޹ص�ͼ�������������ӵ���ƣ��ɹ�ѡ���̽�ֵķ������¡�
1. �򵥵�RGBֵ���������Ƕ����ڹ��յ��������������뵽������ȡ��
2. ��������ĸĽ���lbp�����ӣ�ĿǰЧ�����������룻
3. ����ʦ���Э������󣬿��Գ��ԣ�Ŀǰ��û�������
4. ����Gabor�����ӵ�С����ص������ӣ�
5. ����sift,Gist,Hog,������ص�һϵ�������ӿ��Գ��ԡ�
************************************************************************/

Vec3b ShadowRemovalImg::getLbpValue(int i, int j) const
{
	if (i < 0 || i >= imHeight || j < 0 || j >= imWidth)
		return 0;

	Vec3b value = 0;
	if (i == 0 || i == imHeight - 1 || j == 0 || j == imWidth - 1)
	{
		value[0] = inputImg.at<Vec3b>(i, j)[0];
		value[1] = inputImg.at<Vec3b>(i, j)[1];
		value[2] = inputImg.at<Vec3b>(i, j)[2];
	}
	else
	{
		int ttb = 0;
		uchar cb = inputImg.at<Vec3b>(i, j)[0];
		if (inputImg.at<Vec3b>(i - 1, j - 1)[0] > cb) { value[0] += 1 << ttb; }
		ttb++;
		if (inputImg.at<Vec3b>(i - 1, j)[0] > cb) { value[0] += 1 << ttb; }
		ttb++;
		if (inputImg.at<Vec3b>(i - 1, j + 1)[0] > cb) { value[0] += 1 << ttb; }
		ttb++;
		if (inputImg.at<Vec3b>(i, j + 1)[0] > cb) { value[0] += 1 << ttb; }
		ttb++;
		if (inputImg.at<Vec3b>(i + 1, j + 1)[0] > cb) { value[0] += 1 << ttb; }
		ttb++;
		if (inputImg.at<Vec3b>(i + 1, j)[0] > cb) { value[0] += 1 << ttb; }
		ttb++;
		if (inputImg.at<Vec3b>(i + 1, j - 1)[0] > cb) { value[0] += 1 << ttb; }
		ttb++;
		if (inputImg.at<Vec3b>(i, j - 1)[0] > cb) { value[0] += 1 << ttb; }
		ttb++;

		//g channel
		int ttg = 0;
		uchar cg = inputImg.at<Vec3b>(i, j)[1];
		if (inputImg.at<Vec3b>(i - 1, j - 1)[1] > cg) { value[1] += 1 << ttg; }
		ttg++;
		if (inputImg.at<Vec3b>(i - 1, j)[1] > cg) { value[1] += 1 << ttg; }
		ttg++;
		if (inputImg.at<Vec3b>(i - 1, j + 1)[1] > cg) { value[1] += 1 << ttg; }
		ttg++;
		if (inputImg.at<Vec3b>(i, j + 1)[1] > cg) { value[1] += 1 << ttg; }
		ttg++;
		if (inputImg.at<Vec3b>(i + 1, j + 1)[1] > cg) { value[1] += 1 << ttg; }
		ttg++;
		if (inputImg.at<Vec3b>(i + 1, j)[1] > cg) { value[1] += 1 << ttg; }
		ttg++;
		if (inputImg.at<Vec3b>(i + 1, j - 1)[1] > cg) { value[1] += 1 << ttg; }
		ttg++;
		if (inputImg.at<Vec3b>(i, j - 1)[1] > cg) { value[1] += 1 << ttg; }
		ttg++;

		//r channel
		int ttr = 0;
		uchar cr = inputImg.at<Vec3b>(i, j)[2];
		if (inputImg.at<Vec3b>(i - 1, j - 1)[2] > cr) { value[2] += 1 << ttr; }
		ttr++;
		if (inputImg.at<Vec3b>(i - 1, j)[2] > cr) { value[2] += 1 << ttr; }
		ttr++;
		if (inputImg.at<Vec3b>(i - 1, j + 1)[2] > cr) { value[2] += 1 << ttr; }
		ttr++;
		if (inputImg.at<Vec3b>(i, j + 1)[2] > cr) { value[2] += 1 << ttr; }
		ttr++;
		if (inputImg.at<Vec3b>(i + 1, j + 1)[2] > cr) { value[2] += 1 << ttr; }
		ttr++;
		if (inputImg.at<Vec3b>(i + 1, j)[2] > cr) { value[2] += 1 << ttr; }
		ttr++;
		if (inputImg.at<Vec3b>(i + 1, j - 1)[2] > cr) { value[2] += 1 << ttr; }
		ttr++;
		if (inputImg.at<Vec3b>(i, j - 1)[2] > cr) { value[2] += 1 << ttr; }
	}
	return value;
}

Vec3b ShadowRemovalImg::getImprovedLbpValue(int i, int j) const
{
	if (logFilteredImg.empty())
	{
		cout << "compute LOG filtered image first" << endl;
		return (0, 0, 0);
	}
	if (i <0 || i >= imHeight || j <0 || j >= imWidth)
		return 0;

	Vec3b value = 0;
	if (i == 0 || i == imHeight - 1 || j == 0 || j == imWidth - 1)
	{
		value[0] = logFilteredImg.at<Vec3b>(i, j)[0];
		value[1] = logFilteredImg.at<Vec3b>(i, j)[1];
		value[2] = logFilteredImg.at<Vec3b>(i, j)[2];
	}
	else
	{
		int ttb = 0;
		uchar cb = logFilteredImg.at<Vec3b>(i, j)[0];
		if (logFilteredImg.at<Vec3b>(i - 1, j - 1)[0] > cb) { value[0] += 1 << ttb; }
		ttb++;
		if (logFilteredImg.at<Vec3b>(i - 1, j)[0] > cb) { value[0] += 1 << ttb; }
		ttb++;
		if (logFilteredImg.at<Vec3b>(i - 1, j + 1)[0] > cb) { value[0] += 1 << ttb; }
		ttb++;
		if (logFilteredImg.at<Vec3b>(i, j + 1)[0] > cb) { value[0] += 1 << ttb; }
		ttb++;
		if (logFilteredImg.at<Vec3b>(i + 1, j + 1)[0] > cb) { value[0] += 1 << ttb; }
		ttb++;
		if (logFilteredImg.at<Vec3b>(i + 1, j)[0] > cb) { value[0] += 1 << ttb; }
		ttb++;
		if (logFilteredImg.at<Vec3b>(i + 1, j - 1)[0] > cb) { value[0] += 1 << ttb; }
		ttb++;
		if (logFilteredImg.at<Vec3b>(i, j - 1)[0] > cb) { value[0] += 1 << ttb; }
		ttb++;

		//g channel
		int ttg = 0;
		uchar cg = logFilteredImg.at<Vec3b>(i, j)[1];
		if (logFilteredImg.at<Vec3b>(i - 1, j - 1)[1] > cg) { value[1] += 1 << ttg; }
		ttg++;
		if (logFilteredImg.at<Vec3b>(i - 1, j)[1] > cg) { value[1] += 1 << ttg; }
		ttg++;
		if (logFilteredImg.at<Vec3b>(i - 1, j + 1)[1] > cg) { value[1] += 1 << ttg; }
		ttg++;
		if (logFilteredImg.at<Vec3b>(i, j + 1)[1] > cg) { value[1] += 1 << ttg; }
		ttg++;
		if (logFilteredImg.at<Vec3b>(i + 1, j + 1)[1] > cg) { value[1] += 1 << ttg; }
		ttg++;
		if (logFilteredImg.at<Vec3b>(i + 1, j)[1] > cg) { value[1] += 1 << ttg; }
		ttg++;
		if (logFilteredImg.at<Vec3b>(i + 1, j - 1)[1] > cg) { value[1] += 1 << ttg; }
		ttg++;
		if (logFilteredImg.at<Vec3b>(i, j - 1)[1] > cg) { value[1] += 1 << ttg; }
		ttg++;

		//r channel
		int ttr = 0;
		uchar cr = logFilteredImg.at<Vec3b>(i, j)[2];
		if (logFilteredImg.at<Vec3b>(i - 1, j - 1)[2] > cr) { value[2] += 1 << ttr; }
		ttr++;
		if (logFilteredImg.at<Vec3b>(i - 1, j)[2] > cr) { value[2] += 1 << ttr; }
		ttr++;
		if (logFilteredImg.at<Vec3b>(i - 1, j + 1)[2] > cr) { value[2] += 1 << ttr; }
		ttr++;
		if (logFilteredImg.at<Vec3b>(i, j + 1)[2] > cr) { value[2] += 1 << ttr; }
		ttr++;
		if (logFilteredImg.at<Vec3b>(i + 1, j + 1)[2] > cr) { value[2] += 1 << ttr; }
		ttr++;
		if (logFilteredImg.at<Vec3b>(i + 1, j)[2] > cr) { value[2] += 1 << ttr; }
		ttr++;
		if (logFilteredImg.at<Vec3b>(i + 1, j - 1)[2] > cr) { value[2] += 1 << ttr; }
		ttr++;
		if (logFilteredImg.at<Vec3b>(i, j - 1)[2] > cr) { value[2] += 1 << ttr; }
	}
	return value;
}


void ShadowRemovalImg::computePatchFeatureDescriptor()
{

}

//��һ�֣���lbp����
double ShadowRemovalImg::getDisOfTexture_lbp(int pIndex1, int pIndex2) const
{
	double disTexture = 0.0;
	for (int i = 0; i < patchHeight; ++i)
	{
		for (int j = 0; j < patchWidth; ++j)
		{
			double dis = sqrt((patchImg[pIndex1][i][j].lbpB / 255.0 - patchImg[pIndex2][i][j].lbpB / 255.0)*(patchImg[pIndex1][i][j].lbpB / 255.0 - patchImg[pIndex2][i][j].lbpB / 255.0)
				+ (patchImg[pIndex1][i][j].lbpG / 255.0 - patchImg[pIndex2][i][j].lbpG / 255.0)*(patchImg[pIndex1][i][j].lbpG / 255.0 - patchImg[pIndex2][i][j].lbpG / 255.0)
				+ (patchImg[pIndex1][i][j].lbpR / 255.0 - patchImg[pIndex2][i][j].lbpR / 255.0)*(patchImg[pIndex1][i][j].lbpR / 255.0 - patchImg[pIndex2][i][j].lbpR / 255.0));
			disTexture += dis;
		}
	}
	disTexture /= patchSize;
	return disTexture;
}

//�ڶ��֣����ص�������������(6ά�ģ�
double ShadowRemovalImg::getDisOfTexture_fea(int pIndex1, int pIndex2) const
{
	double disTexture = 0.0;
	double dis;
	for (int i = 0; i < patchHeight; ++i)
	{
		for (int j = 0; j < patchWidth; ++j)
		{
			dis = sqrt((patchImg[pIndex1][i][j].pf.intensity / 255.0 - patchImg[pIndex2][i][j].pf.intensity / 255.0)*(patchImg[pIndex1][i][j].pf.intensity / 255.0 - patchImg[pIndex2][i][j].pf.intensity / 255.0)
				+ (patchImg[pIndex1][i][j].pf.chromaticity / 255.0 - patchImg[pIndex2][i][j].pf.chromaticity / 255.0)*(patchImg[pIndex1][i][j].pf.chromaticity / 255.0 - patchImg[pIndex2][i][j].pf.chromaticity / 255.0)
				+ (patchImg[pIndex1][i][j].pf.firstDevX / 255.0 - patchImg[pIndex2][i][j].pf.firstDevX / 255.0)*(patchImg[pIndex1][i][j].pf.firstDevX / 255.0 - patchImg[pIndex2][i][j].pf.firstDevX / 255.0)
				+ (patchImg[pIndex1][i][j].pf.firstDevY / 255.0 - patchImg[pIndex2][i][j].pf.firstDevY / 255.0)*(patchImg[pIndex1][i][j].pf.firstDevY / 255.0 - patchImg[pIndex2][i][j].pf.firstDevY / 255.0)
				+ (patchImg[pIndex1][i][j].pf.secondDevX / 255.0 - patchImg[pIndex2][i][j].pf.secondDevX / 255.0)*(patchImg[pIndex1][i][j].pf.secondDevX / 255.0 - patchImg[pIndex2][i][j].pf.secondDevX / 255.0)
				+ (patchImg[pIndex1][i][j].pf.secondDevY / 255.0 - patchImg[pIndex2][i][j].pf.secondDevY / 255.0)*(patchImg[pIndex1][i][j].pf.secondDevY / 255.0 - patchImg[pIndex2][i][j].pf.secondDevY / 255.0));
			disTexture += dis;
		}
	}
	disTexture /= patchSize;
	return disTexture;
}

//�����֣�����Э�������
double ShadowRemovalImg::getDisOfTexture_covM(int pIndex1, int pIndex2) const
{
	//Ϊ����߼���Ч�ʣ��ÿռ任ʱ�䣬֮ǰ�Ѿ�������ÿ��patch��Э���������������洢,ÿ��patch������������ά�ȶ���78
	double disTexture = 0.0;

	for (int i = 0; i < 78; ++i)
	{
		double dis = sqrt((patchCovVector[pIndex1][i] - patchCovVector[pIndex2][i])*(patchCovVector[pIndex1][i] - patchCovVector[pIndex2][i]));
		disTexture += dis;
	}

	return disTexture;
}

double ShadowRemovalImg::getPatchSimilarity(int pIndex1, int pIndex2) const
{
	double disPatch = 0.0;

	//1.ֻ��RGB
	//disPatch = getDisOfRGB(p1, p2);
	//2.ֻ��lbp
	//disPatch = getDisOfTexture_lbp(pIndex1, pIndex2);
	//3.RGB+Pos
	//disPatch = getDisOfRGB(p1, p2) + getDisOfPos(p1, p2);
	//disPatch = getDisOfPos(pIndex1, pIndex2);
	//4.lbp+Pos ���ԽСԽ����
	disPatch = 1.5*exp(-1 / getDisOfTexture_lbp(pIndex1, pIndex2)) + 0.5*exp(-1 / getDisOfPos(pIndex1, pIndex2)) + 2.0*exp(-1 / getDisOfChromaticity(pIndex1, pIndex2));

	return disPatch;
}

/**********************************************************************************
�ؼ�����֮���� ��ƥ��patch match�����������Ժ�����Ч�����������֡�
�������ԣ���һ��һ���������ƥ�䣬���ǿ�������patch����Ϣ����һ���Ե������������д����ƣ�
����Ч�ʣ� ��Ҫ�Ǽ����������ٶȣ������������ǰk-d tree�����
************************************************************************/
void ShadowRemovalImg::coherentPatchMatch2D()
{
	if (patchImg == NULL)
	{
		cout << "decompose the input image first!" << endl;
		return;
	}
	if (shadowPatchIndices.empty() || litPatchIndices.empty())
	{
		cout << "please divide the patches into lit and shadow!" << endl;
		return;
	}
	/* A����(Ĭ�ϣ�,�ǵ���ʽƥ��: ÿ�β���һ����Ч��ƥ��֮�󣬲�ɾ�����Ѿ���Ե�lit patch������ʹ�����match patch���ӹ�ƽ�����ǻ�����ͬ��lit patch����������patch������ʱ�俪�����ܻ�ܴ�*/


	for (unsigned int i = 0; i < shadowPatchIndices.size(); ++i)
	{
		double leastPatchDiff = 0.0;
		double patchDiff = 0.0;
		int matchedLitPatchIndex = 0;
		for (unsigned int j = 0; j < litPatchIndices.size(); ++j)
		{
			patchDiff = getPatchSimilarity(shadowPatchIndices[i], litPatchIndices[j]);
			if (j == 0)
				leastPatchDiff = patchDiff;
			else if (leastPatchDiff > patchDiff)
			{
				leastPatchDiff = patchDiff;
				matchedLitPatchIndex = j;
			}
		}
		match.insert(make_pair(shadowPatchIndices[i], litPatchIndices[matchedLitPatchIndex]));
	}


	/* B����������ʽƥ��: ÿ��ƥ��һ��lit patch ֮�� ��litPatchIndex��ɾ������lit patch��������Ч�Ľ���patch match��ʱ�俪�������ǿ��ܻ�ɾ������Ч��patch���������ܻ��½�*/
	//vector<int>  litPatchIndexCopy = litPatchIndices;
	//for (size_t i = 0; i < shadowPatchIndices.size(); ++i)
	//{
	//	double leastPatchDiff = 0.0;
	//	int matchLitIndex = 0;
	//	for (unsigned int j = 0; j < litPatchIndexCopy.size(); ++j)
	//	{
	//		double patchDiff = getPatchSimilarity(shadowPatchIndices[i], litPatchIndexCopy[j]);
	//		if (j == 0)
	//			leastPatchDiff = patchDiff;
	//		else if (leastPatchDiff > patchDiff)
	//		{
	//			leastPatchDiff = patchDiff;
	//			matchLitIndex = j;
	//		}
	//	}
	//	match.insert(make_pair(shadowPatchIndices[i], litPatchIndexCopy[matchLitIndex]));
	//	vector<int>::iterator it = litPatchIndexCopy.begin() + matchLitIndex;
	//	litPatchIndexCopy.erase(it);
	//	if (litPatchIndexCopy.empty())//������е�lit patch��ƥ������,���¿�ʼ
	//		litPatchIndexCopy = litPatchIndices;
	//}


	/*C����: patch match���Լ��٣�1. �ȶ�ͼ���lit patches���о��ࣨGaussian K-D ������������������
	2. ʹ������ķ�������patch������������˳�������������ҵ���Ч��ƥ���𣿣�
	3.�ο�patchMatchϵ�����µļ��ٷ�����*/

	/*D����: һ���Ե�patch match, ��Ҫ�ܵ�NRDC�����������ĵĹ۵����ʹ����Ӱ�������ڵ�patch�ڷ���Ӱ����ƥ��õ���lit patchesҲ���������ڵģ��������Ա��������Ӱ�������
	���������ϳ���һ���ԣ��������ģ����������ʧ��artifacts*/

	cout << "the matched patch pairs number: " << match.size() << endl;
}

Vec3d ShadowRemovalImg::getMeanValueofPatch(int p)
{
	if (p <0 || p >= patchNumber)
	{
		return (0, 0, 0);
	}
	Vec3d mean(0, 0, 0);
	for (int i = 0; i < patchHeight; ++i)
	{
		for (int j = 0; j<patchWidth; ++j)
		{
			mean[0] += patchImg[p][i][j].b;
			mean[1] += patchImg[p][i][j].g;
			mean[2] += patchImg[p][i][j].r;
		}
	}
	mean[0] /= patchSize;
	mean[1] /= patchSize;
	mean[2] /= patchSize;
	return mean;
}

double ShadowRemovalImg::getMeanValueofPatchIntensity(int p)
{
	if (p <0 || p >= patchNumber)
	{
		return (0, 0, 0);
	}

	double mean = 0.0;
	for (int i = 0; i < patchHeight; ++i)
	{
		for (int j = 0; j < patchWidth; ++j)
		{
			double gray = patchImg[p][i][j].b*0.114 + patchImg[p][i][j].g*0.587 + patchImg[p][i][j].r*0.299;
			mean += gray;
		}
	}

	mean /= patchSize;
	return mean;
}

double ShadowRemovalImg::getStandardDevofPatch(int p)
{
	if (p<0 || p >= patchNumber)
	{
		return (0, 0, 0);
	}
	double sigma = 0.0;
	double mean = getMeanValueofPatchIntensity(p);
	for (int i = 0; i < patchHeight; ++i)
	{
		for (int j = 0; j < patchWidth; ++j)
		{
			double gray = patchImg[p][i][j].b*0.114 + patchImg[p][i][j].g*0.587 + patchImg[p][i][j].r*0.299;
			sigma += (gray - mean)*(gray - mean);
		}
	}

	sigma /= patchSize;
	sigma = sqrt(sigma);
	return sigma;
}

/************************************************************************
�ؼ�����֮���� �ֲ���Ӱ�������ӣ�local illumination transfer operator
��ͬ�ľֲ���Ӱ�������Ӷ����յĽ����Ӱ��ϴ󣬼��滻 or Shor����or Zhang et al.
************************************************************************/

//1����򵥵ķ�����ֱ����ƥ���lit patches������ֵ�����滻
void ShadowRemovalImg::illuminationTransferOperator_Naive(int _shadowPatchIndex)
{
	for (int i = 0; i < patchHeight; ++i)
	{
		for (int j = 0; j < patchWidth; ++j)
		{
			patchImg[_shadowPatchIndex][i][j].b = patchImg[match[_shadowPatchIndex]][i][j].b;
			patchImg[_shadowPatchIndex][i][j].g = patchImg[match[_shadowPatchIndex]][i][j].g;
			patchImg[_shadowPatchIndex][i][j].r = patchImg[match[_shadowPatchIndex]][i][j].r;
		}
	}
}

//2: Shor���˵Ĺ���Ǩ�Ʒ�����������ɫǨ�ƣ�
void ShadowRemovalImg::illuminationTransferOperator_shor(int _shadowPatchIndex)
{
	Vec3d mean_S = getMeanValueofPatch(_shadowPatchIndex);
	Vec3d mean_L = getMeanValueofPatch(match[_shadowPatchIndex]);

	double sigma_S = getStandardDevofPatch(_shadowPatchIndex);
	double sigma_L = getStandardDevofPatch(match[_shadowPatchIndex]);

	Vec3b shadowFreePixel;

	for (int i = 0; i < patchHeight; ++i)
	{
		for (int j = 0; j < patchWidth; ++j)
		{
			shadowFreePixel[0] = (uchar)((sigma_L / sigma_S)*(patchImg[_shadowPatchIndex][i][j].b - mean_S[0]) + mean_L[0]);
			shadowFreePixel[1] = (uchar)((sigma_L / sigma_S)*(patchImg[_shadowPatchIndex][i][j].g - mean_S[1]) + mean_L[1]);
			shadowFreePixel[2] = (uchar)((sigma_L / sigma_S)*(patchImg[_shadowPatchIndex][i][j].r - mean_S[2]) + mean_L[2]);

			patchImg[_shadowPatchIndex][i][j].b = shadowFreePixel[0];
			patchImg[_shadowPatchIndex][i][j].g = shadowFreePixel[1];
			patchImg[_shadowPatchIndex][i][j].r = shadowFreePixel[2];
		}
	}

	patchPixelNormalization(_shadowPatchIndex);
}

//3: ����ʦ��TIP2015��cvpr2011�ȵľֲ�����Ǩ�����ӣ���ʵ��һ���ģ�
void ShadowRemovalImg::illuminationTransferOperator_Zhang(unsigned int _shadowPatchIndex)
{
	if (_shadowPatchIndex < 0 || _shadowPatchIndex >= (unsigned int)patchNumber)
		return;
	Vec3d mean_S = getMeanValueofPatch(_shadowPatchIndex);
	Vec3b mean_L = getMeanValueofPatch(match[_shadowPatchIndex]);
	double sigma_S = getStandardDevofPatch(_shadowPatchIndex);
	double sigma_L = getStandardDevofPatch(match[_shadowPatchIndex]);

	Vec3d initialValue;
	Vec3d t;

	for (int i = 0; i < patchHeight; ++i)
	{
		for (int j = 0; j < patchWidth; ++j)
		{
			initialValue[0] = (sigma_L / sigma_S)*(patchImg[_shadowPatchIndex][i][j].b - mean_S[0]) + mean_L[0];
			initialValue[1] = (sigma_L / sigma_S)*(patchImg[_shadowPatchIndex][i][j].g - mean_S[1]) + mean_L[1];
			initialValue[2] = (sigma_L / sigma_S)*(patchImg[_shadowPatchIndex][i][j].r - mean_S[2]) + mean_L[2];

			t[0] = (patchImg[_shadowPatchIndex][i][j].b - initialValue[0]) / (patchImg[_shadowPatchIndex][i][j].alpha*initialValue[0] - patchImg[_shadowPatchIndex][i][j].b);
			t[1] = (patchImg[_shadowPatchIndex][i][j].g - initialValue[1]) / (patchImg[_shadowPatchIndex][i][j].alpha*initialValue[1] - patchImg[_shadowPatchIndex][i][j].g);
			t[2] = (patchImg[_shadowPatchIndex][i][j].r - initialValue[2]) / (patchImg[_shadowPatchIndex][i][j].alpha*initialValue[2] - patchImg[_shadowPatchIndex][i][j].r);

			patchImg[_shadowPatchIndex][i][j].b = (uchar)((t[0] + 1)*patchImg[_shadowPatchIndex][i][j].b / (patchImg[_shadowPatchIndex][i][j].alpha*t[0] + 1));
			patchImg[_shadowPatchIndex][i][j].g = (uchar)((t[1] + 1)*patchImg[_shadowPatchIndex][i][j].g / (patchImg[_shadowPatchIndex][i][j].alpha*t[1] + 1));
			patchImg[_shadowPatchIndex][i][j].r = (uchar)((t[2] + 1)*patchImg[_shadowPatchIndex][i][j].r / (patchImg[_shadowPatchIndex][i][j].alpha*t[2] + 1));
		}
	}
	patchPixelNormalization(_shadowPatchIndex);
}

//

/************************************************************************
��������3����Ҫ��һ�������Ĺؼ����⣺
1.�������ƶȵ���ơ�����һ���ؼ����⣬�����õ�patch�����Զ�����׼������lbp�����ӵ��������ƶȺͶ�ά��������ƶ��Ƿ�ɿ���������Ϊ��Ӱ����ʧ�˺ܶ��������Ϣ
�������»������������Զ�����patch matchƥ�䲻׼ȷ���������յĽ����

2.��ƥ�䡣coherentPatchmatch2D�У��費��Ҫ���Ѿ���Ե�lit patchɾ��������ܶ�shadow patches��ƥ�䵽һ��lit patch,ͬʱ���Լ���ʱ�俪��,
���������Ǽ򵥵�patchMatch��û�п���һ���Ե�patchƥ�䣬��NRDC�еķ�����

3.��Ӱ�������ӡ��ھֲ���Ӱ�����Ĺ����У���ֱ����ƥ���lit patches��������Ϣ�滻shadow patches��������Ϣ��Ҳ��һ�ֹ���Ǩ��˼�룩��������shor������
������Ӱ�������������Ӱ��������֮������Թ�ϵ����ͳ����Ϣ���ڹ�ϵʽǨ�ƹ�����Ϣ������һ��ֵ���о������⣬Ҳ���Ǿֲ�����Ӱ�������ӵ���ƣ�
��Ҳ��һ�������д��µĵط���

4.һ���Իָ��Ĳ��ԡ�����Ч��ƥ����patch֮����ô�ѹ���Ǩ�ƺ��shadow patches(illumination transfered) ��ֵ���ϵ�һ��Ҳ����һ���ԵĹ���Ǩ�ƣ�һ��
���ر����patches���ǣ�������ص���Ӱ�������ֵ����ô�㣿����
************************************************************************/
void ShadowRemovalImg::shadowRemovalLocal(const string & _inputImg, const string _shadowImg, const string _shadowFreeImg)
{
	inputImg = imread(_inputImg);
	shadowImg = imread(_shadowImg, 0);
	shadowlessImg = inputImg.clone();

	Mat img_hsv;
	vector<Mat> hsv;
	cvtColor(inputImg, img_hsv, CV_RGB2HSV);
	split(img_hsv, hsv);
	chromaticity = hsv[0].clone();

	imHeight = inputImg.rows;
	imWidth = inputImg.cols;
	imChannels = inputImg.channels();
	imSize = imHeight*imWidth;

	patchNumber = imWidth / patchWidth * imHeight / patchHeight;

	imageDecompose();

	patchDivision2LitAndShadow();

	coherentPatchMatch2D();

	for (unsigned int i = 0; i < shadowPatchIndices.size(); ++i)
	{
		illuminationTransferOperator_Zhang(shadowPatchIndices[i]);
	}

	imageRecompose();
	imwrite(_shadowFreeImg, shadowlessImg);

	imshow("input", inputImg);
	imshow("remove", shadowlessImg);
	waitKey(0);
}


//global shadow removal, the main image shadow removal process
void ShadowRemovalImg::shadowRemovalGlobal(const string & _inputImg, const string & _shadowImg, const string & _shadowfreeImg)
{
	//1 - read the input and shadow image assign image-related values
	cout << "1 ****** reading images ******" << endl;
	inputImg = imread(_inputImg);
	shadowlessImg = inputImg.clone();

	//�õ�ɫ��ֵ
	Mat img_hsv;
	vector<Mat> hsv;
	cvtColor(inputImg, img_hsv, CV_RGB2HSV);
	split(img_hsv, hsv);
	chromaticity = hsv[0].clone();

	shadowImg = imread(_shadowImg, 0);
	imHeight = inputImg.rows;
	imWidth = inputImg.cols;
	imChannels = inputImg.channels();
	imSize = imHeight*imWidth;
	patchNumber = (imWidth - gapx) / gapx * (imHeight - gapy) / gapy;

	clock_t start, end;
	double totalTime;

	//2 - image decompose to overlapped patches
	cout << endl << "2 ****** image overlapped decomposition ******" << endl;
	start = clock();
	imageOverlapDecompose();
	end = clock();
	totalTime = (double)(end - start) / 1000;
	cout << "total time of image overlapped decomposition is: " << totalTime << " s" << endl;

	//3 - divide patches to lit and shadow based on the alpha values
	cout << endl << "3 ****** divide patches to lit and shadow ******" << endl;
	start = clock();
	patchDivision2LitAndShadow();
	end = clock();
	totalTime = (double)(end - start) / 1000;
	cout << "total time of patches division is: " << totalTime << " s" << endl;

	//4 - patch match
	cout << endl << "4 ****** patch match ******" << endl;
	start = clock();
	coherentPatchMatch2D();
	end = clock();
	totalTime = (double)(end - start) / 1000;
	cout << "total time of patch match is: " << totalTime << " s" << endl;

	//5-shadow removal process
	cout << endl << "5 ****** shadow removal ******" << endl;
	start = clock();
	for (unsigned int i = 0; i < shadowPatchIndices.size(); ++i)
	{
		//illuminationTransferOperator_shor(shadowPatchIndices[i]);
		illuminationTransferOperator_Zhang(shadowPatchIndices[i]);
	}
	end = clock();
	totalTime = (double)(end - start) / 1000;
	cout << "total time of global shadow removal is: " << totalTime << " s" << endl;

	//6 - shadow boundary processing
	cout << endl << "6 ****** shadow boundary processing (to be continue) ******" << endl;

	//7 - write the result back
	cout << endl << "7 ****** save the shadowfree image ******" << endl;
	imagePatchesRecompose();
	imwrite(_shadowfreeImg, shadowlessImg);

	/*imshow("input", inputImg);
	imshow("remove", shadowlessImg);*/
	//waitKey(0);

}

//the last step of shadow removal
void ShadowRemovalImg::shadowBoundaryProcessing()
{

}