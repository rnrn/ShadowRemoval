import cv2
import numpy as np
from matplotlib import pyplot as plt

test_img1 = cv2.imread('test1.img')
#test_img1.height
#test_img1.weidth

#generate Gaussian pyramid
G = test_img1.copy()
gp1 = [G]
for i in xrange(6):
    G = cv2.pyrDown(G)
    gp1.append(G)

#generate Laplacian pyramid
lp1 = [gp1[5]]
for i in xrange(5, 0, -1):
    GE = cv2.pyrUp(gp1[i])
    L = cv2.subtract(gp1[i - 1], GE)
    lp1.append(L)

def RGB_Distance:
    dotProduct = pixel_temp1.R*pixel_temp2.R + pixel_temp1.G*pixel_temp2.G + pixel_temp1.B+pixel_temp2.B + pixel_temp1.A*pixel_temp2.A
    dotSqrt = 
    acos = dotProduct/dotSqrt

def Ms_Detect

def Ml_Detect

def Mshadow_Detect

