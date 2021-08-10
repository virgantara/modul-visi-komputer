import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('../dataset/cameraman.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)

dst = cv.dilate(dst,None)

img[dst>0.01*dst.max()]=[255,0,0]

plt.imshow(img)
plt.show()