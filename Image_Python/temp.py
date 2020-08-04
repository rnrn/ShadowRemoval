import cv2
import numpy as np
from collections import deque


def BFS(tempImg, X, Y):
    retImg = tempImg[X - 6 : X + 6, Y - 6 : Y + 6, 0 : 3]

    return retImg

def getDistance(a, b):
    c = a - b
    rtn = np.sqrt(c.dot(c))

    return rtn

def floodFill(ret, s_x, s_y, H, W):

    rtnRet = np.zeros((H, W))
    flag = np.zeros((H, W))
    rtnRet[s_x, s_y] = 1

    dx = [1, -1, 0, 0]
    dy = [0, 0, 1, -1]

    listX = [s_x]
    listY = [s_y]

    #print H, W;

    while len(listX) != 0 :
        for i in range(4) :
            tx = listX[0] + dx[i]
            ty = listY[0] + dy[i]

            if tx >= 0 and tx < H and ty >= 0 and ty < W and flag[tx, ty] == 0 :
                if ret[tx, ty] == 1 :
                    flag[tx, ty] = 1
                    listX.append(tx)
                    listY.append(ty)
                    rtnRet[tx, ty] = 1

        del listX[0]
        del listY[0]

    return rtnRet

def countS(sIn, flag, s_x, s_y, H, W) :
    #print s_x, s_y

    dx = [1, -1, 0, 0]
    dy = [0, 0, 1, -1]

    queueX = deque([s_x])
    queueY = deque([s_y])

    sOut = sIn

    areaS = 0
    while  len(queueX) != 0 :
        lx = queueX.popleft()
        ly = queueY.popleft()
        for i in range(4) :
            tx = lx + dx[i]
            ty = ly + dy[i]

            #print tx, ty

            if tx >= 0 and tx < H and ty >= 0 and ty < W and flag[tx, ty] == 0 and sOut[tx, ty] == 0 :
                areaS = areaS + 1
                sOut[tx, ty] = -1

                queueX.append(tx)
                queueY.append(ty)

    queueX = deque([s_x])
    queueY = deque([s_y])
    sOut[s_x, s_y] = areaS
    while  len(queueX) != 0 :
        lx = queueX.popleft()
        ly = queueY.popleft()
        for i in range(4) :
            tx = lx + dx[i]
            ty = ly + dy[i]

            #print tx, ty

            if tx >= 0 and tx < H and ty >= 0 and ty < W and flag[tx, ty] == 0 and sOut[tx, ty] == -1 :
                sOut[tx, ty] = areaS

                queueX.append(tx)
                queueY.append(ty)

    return sOut



textImage = cv2.imread('13.png')

textImage = np.array(textImage, dtype=np.double)

h, w, c = textImage.shape

print c, h, w

midX = h / 2
midY = w / 2

shadowSeed = BFS(textImage, midX, midY)

transMat = np.transpose(shadowSeed, (2, 0, 1))
#print transMat.shape
mean2d = shadowSeed.mean(axis=0)
seedBGR = mean2d.mean(axis=0)


eps = 30
retMask = np.zeros((h, w))
for i in range(h) :
    for j in range(w) :
        tempVector = textImage[i, j, ]

        if tempVector.dot(tempVector) == 0 :
            continue

        tempNum = getDistance(tempVector, seedBGR)
        if tempNum <= eps :
            retMask[i, j] = 1

retShadow = floodFill(retMask, midX, midY, h, w)

retIn = np.zeros((h, w))
#retIn = countS(retIn, retShadow, 0, 0, h, w)
#retIn = countS(retIn, retShadow, h - 1, 0, h, w)
#retIn = countS(retIn, retShadow, h - 1, w - 1, h, w)

#print retIn[0,0], retIn[h-1, 0], retIn[h-1, w-1]

for i in range(h) :
    for j in range(w) :
        if retShadow[i, j] == 0 and retIn[i, j] == 0 :
            retIn = countS(retIn, retShadow, i, j, h, w)

#cv2.imshow("a", retShadow)
#cv2.waitKey(0)

for i in range(h) :
    for j in range(w) :
        if retIn[i, j] < 1000:
            retShadow[i, j] = 255


#cv2.imshow("b", retMask)
cv2.imshow("a", retShadow)
cv2.waitKey(0)

#cv2.imwrite('shadow.png', retShadow)
