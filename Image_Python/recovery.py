import cv2
import numpy as np

def getBGR(inputImage):
    transMat = np.transpose(inputImage, (2, 0, 1))
    rtnB = transMat[0]
    rtnG = transMat[1]
    rtnR = transMat[2]

    return rtnB, rtnG, rtnR

def getParameter(inputImage, inputShadow):
    imageH, imageW = inputImage.shape

    sumL, sumS = 0.0, 0.0
    numL, numS = 1, 1
    for i in range(imageH):
        for j in range(imageW):
            if inputShadow[i, j] == 0:
                sumL += inputImage[i, j]
                numL = numL + 1
            else:
                sumS += inputImage[i, j]
                numS = numS + 1
    meanL = sumL / numL
    meanS = sumS / numS
    sDL, sDS = 0.0, 0.0
    for i in range(imageH):
        for j in range(imageW):
            if inputShadow[i, j] == 0:
                sDL += np.square(inputImage[i, j] - meanL)
            else:
                sDS += np.square(inputImage[i, j] - meanS)
    sDL = np.sqrt(sDL)
    sDS = np.sqrt(sDS)

    return np.array([meanL, meanS]), np.array([sDL, sDS])



inputImage = cv2.imread('13.png')
inputShadow = cv2.imread('shadow.png')

inputShadow = cv2.cvtColor(inputShadow, cv2.COLOR_BGR2GRAY)

inputB, inputG, inputR = getBGR(inputImage)

imageH, imageW = inputB.shape

mean, standardDeviation = getParameter(inputB, inputShadow)
gamaB = standardDeviation[0] / standardDeviation[1]
alphaB = mean[0] - gamaB * mean[1]

mean, standardDeviation = getParameter(inputG, inputShadow)
gamaG = standardDeviation[0] / standardDeviation[1]
alphaG= mean[0] - gamaG * mean[1]

mean, standardDeviation = getParameter(inputR, inputShadow)
gamaR = standardDeviation[0] / standardDeviation[1]
alphaR = mean[0] - gamaR * mean[1]

outputImage = inputImage
for i in range(imageH):
    for j in range(imageW):
        if inputShadow[i, j] == 0:
            continue

        outputImage[i, j, 0] = alphaB + gamaB * outputImage[i, j, 0]
        outputImage[i, j, 1] = alphaG + gamaG * outputImage[i, j, 1]
        outputImage[i, j, 2] = alphaR + gamaR * outputImage[i, j, 2]

cv2.imshow("a", outputImage)
cv2.imwrite("remove.png", outputImage)
cv2.waitKey(0)