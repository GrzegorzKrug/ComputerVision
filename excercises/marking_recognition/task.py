import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import imutils
import time
import os

from imutils.perspective import four_point_transform

"""
OPTICAL MARK RECOGNITION
"""

image = cv.imread("omr_test_01.png", cv.IMREAD_COLOR)
image = cv.imread("omr_test_02.png", cv.IMREAD_COLOR)
image = cv.imread("omr_test_03.png", cv.IMREAD_COLOR)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def mask_kartka():
    mask = cv.adaptiveThreshold(
            gray, 255, cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY, 1001, 0,
    )
    k = 5
    kernel = np.ones((k, k))
    mask = cv.erode(mask, kernel)
    mask = cv.dilate(mask, kernel)
    return mask


def grab_ctrs():
    cnts = cv.findContours(mask.copy(),
                           cv.RETR_EXTERNAL,
                           cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)
    c = cnts[0]
    peri = cv.arcLength(c, True)
    assert peri > 500, "Obwód jest duży"

    approx = cv.approxPolyDP(c, 0.03 * peri, True)
    # print(approx)

    better = four_point_transform(image, approx.reshape(-1, 2))
    return better
    # print(cnts)


def binaryze_final():
    im = morphed - morphed.min()
    mask = im < 127
    out = im.copy()
    out[mask] = 255
    out[~mask] = 0
    return out


def draw_grid():
    image = morphed
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h, w, c = image.shape

    X_padding = 65, 43
    Y_padding = 50, 100
    rows = 5
    columns = 5

    col_lines = np.linspace(X_padding[0], w - X_padding[1], columns + 1, dtype=int)
    row_lines = np.linspace(Y_padding[0], h - Y_padding[1], rows + 1, dtype=int)

    COL_START = X_padding[0]
    ROW_START = Y_padding[0]
    COL_END = w - X_padding[1]
    ROW_END = h - Y_padding[1]

    p1 = X_padding[0], Y_padding[0]
    p2 = COL_END, ROW_END
    cv.rectangle(image, p1, p2, (10, 10, 25), 2)
    for x in col_lines:
        pt1 = x, Y_padding[0]
        pt2 = x, h - Y_padding[1]
        cv.line(image, pt1, pt2, (70, 0, 0), 1)

    for y in row_lines:
        pt1 = X_padding[0], y
        pt2 = w - X_padding[1], y
        cv.line(image, pt1, pt2, (0, 0, 80), 2)

    for r_start, r_end in zip(row_lines, row_lines[1:]):
        # roi = image[COL_START:COL_END + 1, r_start:r_end]
        answers = []
        for c_start, c_end in zip(col_lines, col_lines[1:]):
            # print(r_start, r_end, c_start, c_end)
            roi = fn_mask[r_start:r_end, c_start: c_end]
            # print(roi.sum())
            answers.append(roi.sum())

        arg = np.argmax(answers)
        letter = chr(arg + 65)
        print(f"Odpowiedź: {letter}")

        roi = image[r_start:r_end, col_lines[arg]:col_lines[arg + 1], :]
        blank = (roi * 0 + (5, 80, 5,)).astype(np.uint8)
        roi = cv.add(roi, blank)
        image[r_start:r_end, col_lines[arg]:col_lines[arg + 1], :] = roi

    return image


mask = mask_kartka()
plt.imsave("pic-1-kontur.png", mask)

morphed = grab_ctrs()
plt.imsave("pic-2-morf.png", morphed)

fn_mask = binaryze_final()
plt.imsave("pic-3-maska.png", fn_mask)

final = draw_grid()
plt.imsave("pic-4-odpowiedzi.png", final)
