import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import time
import glob
import cv2
import sys
import os

import imutils
import numba

from matplotlib.style import use
from itertools import cycle
import collections


def shadow_mask(frame_hsv, bg_frame_hsv):
    alfa = 0.4  # val min ratio
    beta = 0.8  # val max ratio
    ts = 125  # t saturation max
    th = 125  # t hue max

    # DIFF = np.abs(DIFF)
    rat1 = frame_hsv[:, :, 2] / bg_frame_hsv[:, :, 2]
    # plt.imshow(rat1)
    # plt.colorbar()
    # plt.show()
    DIFF = frame_hsv.astype(float) - bg_frame_hsv.astype(float)

    cond1 = (alfa <= rat1) & (beta >= rat1)
    cond2 = DIFF[:, :, 1] <= ts # Saturation condition
    cond3 = DIFF[:, :, 0] <= th # hue condition

    mask = cond1 & cond2 & cond3
    # plt.imshow(mask)
    # plt.colorbar()
    # plt.show()

    return mask


# FILE_DIR = os.path.join(os.path.expanduser("~"), "Local") + os.path.sep
FILE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(FILE_DIR)

cap = cv2.VideoCapture(
        FILE_DIR + os.path.sep +
        'A3 A-road Traffic UK HD - rush hour - British Highway traffic May 2017.mp4'
)
W = 500

fgbgAdaptiveGaussain = cv2.createBackgroundSubtractorMOG2(10, )
_, frame = cap.read()
pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
cap.set(cv2.CAP_PROP_POS_FRAMES, pos + 200)
cap.set(cv2.CAP_PROP_POS_FRAMES, 1600)  # sunshine
cap.set(cv2.CAP_PROP_POS_FRAMES, 1750)  # sunshine
cap.set(cv2.CAP_PROP_POS_FRAMES, 1801)  # sunshine

first_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
first_gray = imutils.resize(first_gray, width=W)
first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)

frame_history = collections.deque(maxlen=10)
# mask_history = collections.deque(maxlen=5)
PAUSE = False

while True:
    if not PAUSE:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=W)
        curr_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_history.append(frame)
        mean_frame = np.mean(frame_history, axis=0).astype(np.uint8)
        mean_frame_gray = cv2.cvtColor(mean_frame, cv2.COLOR_BGR2GRAY)

        mean_frame_hsv = cv2.cvtColor(mean_frame, cv2.COLOR_BGR2HSV)
        cur_frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # value_dif = cv2.absdiff(mean_frame_hsv, cur_frame_hsv)
        # value_mask = value_dif > 30
        # value_mask = (value_mask * 255).astype(np.uint8)
        # value_shadow_mask = value_mask[:, :, 2]
        mask_shadow = shadow_mask(cur_frame_hsv, mean_frame_hsv)
        mask_shadow = np.round(mask_shadow * 255).astype(np.uint8)
        # value_dif = value_dif

        imdif = cv2.absdiff(mean_frame_gray, curr_gray_frame)
        _, fgmask = cv2.threshold(imdif, 25, 255, cv2.THRESH_BINARY)

        # noisefld=np.random.randn(frame.shape[0],frame.shape[1])
        # frame[:,:,0]=(frame[:,:,0]+10*noisefld).astype('int')
        # frame[:,:,1]=(frame[:,:,1]+10*noisefld).astype('int')
        # frame[:,:,2]=(frame[:,:,2]+10*noisefld).astype('int')

        fgbgAdaptiveGaussainmask = fgbgAdaptiveGaussain.apply(curr_gray_frame)
        # mask_history.append(fgbgAdaptiveGaussainmask)
        # mask = np.mean(mask_history, axis=0)
        mog_mask = fgbgAdaptiveGaussainmask
    else:
        time.sleep(0.1)

    # cv2.namedWindow('Background Subtraction', 0)
    # cv2.namedWindow('Background Subtraction Adaptive Gaussian', 0)
    # cv2.namedWindow('Original', 0)

    # cv2.imshow('Input', mean_frame)
    cv2.imshow('Original', frame)
    cv2.imshow('imdif', imdif)
    cv2.imshow('Mean mask', fgmask)
    cv2.imshow("MOG mask", mog_mask)
    cv2.imshow("Shadow mask", mask_shadow)
    # cv2.imshow("HSV,V Diff", cv2.cvtColor(value_dif, cv2.COLOR_HSV2BGR))
    # cv2.imshow("HSV,V Diff", cv2.cvtColor(value_mask, cv2.COLOR_HSV2BGR))

    k = cv2.waitKey(1) & 0xff

    if k == ord('q'):
        break
    elif k == 32:
        PAUSE = not PAUSE

cap.release()
cv2.destroyAllWindows()
print('Program Closed')
