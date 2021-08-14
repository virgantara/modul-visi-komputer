from imutils.video import FPS
import imutils

import cv2 as cv
import numpy as np

from myutils import dilatation, detect

H_low = 6  # 0
H_high = 35  # 179
S_low = 98  # 0
S_high = 162  # 255
V_low = 221  # 0
V_high = 255

def callback(x):
    global H_low, H_high, S_low, S_high, V_low, V_high

    H_low = cv.getTrackbarPos('low H', 'controls')
    H_high = cv.getTrackbarPos('high H', 'controls')
    S_low = cv.getTrackbarPos('low S', 'controls')
    S_high = cv.getTrackbarPos('high S', 'controls')
    V_low = cv.getTrackbarPos('low V', 'controls')
    V_high = cv.getTrackbarPos('high V', 'controls')


def track_realtime():
    cv.namedWindow('controls', 2)
    cv.resizeWindow("controls", 550, 10);

    # create trackbars for high,low H,S,V
    cv.createTrackbar('low H', 'controls', 0, 179, callback)
    cv.createTrackbar('high H', 'controls', 179, 179, callback)

    cv.createTrackbar('low S', 'controls', 0, 255, callback)
    cv.createTrackbar('high S', 'controls', 255, 255, callback)

    cv.createTrackbar('low V', 'controls', 0, 255, callback)
    cv.createTrackbar('high V', 'controls', 255, 255, callback)

    cv.createTrackbar('Element shape', 'controls', 0, 2, dilatation)
    cv.createTrackbar('Kernel size', 'controls', 0, 21, dilatation)

    cap = cv.VideoCapture(0)

    while 1:

        _, frame = cap.read()
        lower_limit = np.array([H_low, S_low, V_low])
        upper_limit = np.array([H_high, S_high, V_high])
        res, undilated, dilated = detect(frame, lower_limit, upper_limit)
        cv.imshow('Tanpa Dilatasi', undilated)
        cv.imshow('Dengan Dilatasi', dilated)
        cv.imshow('res', res)

        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break
    cv.destroyAllWindows()

if __name__ == '__main__':
    track_realtime()
