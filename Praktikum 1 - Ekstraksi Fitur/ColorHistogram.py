import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

img = cv.imread('../dataset/cameraman.png',0)

hist = np.histogram(img.ravel(), bins=256,range=[0,256])
plt.bar(x=np.arange(256), height= hist[0],align="center", width=0.1)

plt.show()