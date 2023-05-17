import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import sys
import os

import numba

from matplotlib.style import use
import matplotlib as mpl

# use('seaborn-dark')
mpl.rcParams['axes.grid'] = True
mpl.rcParams['lines.linewidth'] = 2


# mpl.rcParams['axes.facecolor'] = (0.7, 0.8, 0.7)


# import sklearn.image as simage

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


# def true_move_avg(x, w):
#     out = []
#     for i in range(len(x)):
#         st = i - w
#         if st < 0:
#             st = 0
#         sp = i + 1 + w
#
#         sl = x[st:sp]
#         v = np.mean(sl)
#         out.append(v)
#     return out


@numba.njit()
def calc_hist(mat, merge_channels=False):
    out = np.zeros((3, 256), dtype=np.int64)

    assert len(mat.shape) == 3, "Image must be in 3d shape- Y,X,C"
    hei, wid, cha = mat.shape
    # print("ycbr shape", mat.shape)
    for c in range(cha):
        vals = mat[:, :, c].ravel()
        for v in vals:
            out[c, v] += 1
        # out[c, vals] += 1

    return out


george_paths = glob.glob(
        f"lfw{os.path.sep}George*{os.path.sep}*.jpg",
        recursive=True)

arnold_paths = glob.glob(
        f"lfw{os.path.sep}Arnold*{os.path.sep}*.jpg",
        recursive=True)

orl_paths = glob.glob(
        f"orl{os.path.sep}*.jpg",
        recursive=True)


def subtask1():
    size = 40  # Center face radius
    hist = np.zeros((3, 256), dtype=int)

    for g in george_paths:
        print(g)
        mat = cv2.imread(g, cv2.IMREAD_COLOR)
        cx, cy, _ = map(lambda x: x // 2, mat.shape)
        mat = mat[cy - size:cy + size, cx - size:cx + size, :]
        # cv2.imshow("face",mat)
        # cv2.waitKey()

        mat_ycrcb = cv2.cvtColor(mat, cv2.COLOR_BGR2YCrCb)

        h = calc_hist(mat_ycrcb)
        hist += h
        k = cv2.waitKey() & 0xFF

        if k == ord("q"):
            break

        # for v in mat_ycrbc:
        # pass
        # break

    # plt.plot(hist[0, :], label='Y')
    plt.plot(hist[1, :], label='Cr')
    plt.plot(hist[2, :], label='Cb')

    win = 10
    perc = 10

    crmean = moving_average(hist[1, :], win)
    cr_p = (perc / 100) * np.max(crmean)
    cbmean = moving_average(hist[2, :], win)
    cb_p = (perc / 100) * np.max(cbmean)

    plt.plot(crmean, dashes=[5, 3], c='r', label="Cr Smooth")
    plt.plot(cbmean, dashes=[5, 3], c='b', label="Cb Smooth")

    plt.plot([0, 255], [cr_p] * 2, c='r', label=f"Cr Smooth, {perc}% of peak", linewidth=1)
    plt.plot([0, 255], [cb_p] * 2, c='b', label=f"Cb Smooth, {perc}% of peak", linewidth=1)

    plt.legend(loc='best')
    plt.xlim([80, 180])
    plt.title("George Bush histogram")
    plt.show()


def stack_arnolds(stack=5, stack_2d=False):
    images = None
    row = []
    end = stack ** 2 if stack_2d else stack
    for i, g in enumerate(arnold_paths):
        if i >= end:
            break

        mat = cv2.imread(g, cv2.IMREAD_COLOR)
        # cx, cy, _ = map(lambda x: x // 2, mat.shape)
        row.append(mat)

        if i > 0 and not ((i + 1) % stack):
            row_stack = np.hstack(row, )
            # print("row", row_stack.shape)

            if images is not None:
                # print("stacking")
                # print("row", row_stack.shape)
                # print("all", images.shape)
                images = np.concatenate([images, row_stack], axis=0)
                # print("after stack", images.shape)
            else:
                # print(f"new row: {row_stack.shape}")
                images = row_stack

            row = []

    # print("final:", images.shape)
    images = images.astype(np.uint8)
    return images


def subtask2():
    """Progowanie Arnolda"""

    blue = [93, 127]
    red = [133, 172]
    # blue = [85, 130]
    # red = [125, 180]

    # images = stack_arnolds()
    images = stack_arnolds(3, stack_2d=True)
    print(images.shape)
    cv2.imshow("Arnold", images)

    arnd_ycrcb = cv2.cvtColor(images, cv2.COLOR_BGR2YCrCb)
    mask_r = (red[0] < arnd_ycrcb[:, :, 1]) & (arnd_ycrcb[:, :, 1] < red[1])
    mask_b = (blue[0] < arnd_ycrcb[:, :, 2]) & (arnd_ycrcb[:, :, 2] < blue[1])
    mask = mask_r & mask_b

    prog = images.copy()
    print("masj", mask.shape)
    prog[~mask, :] = 0
    cv2.imshow("Progowanie", prog)

    key = cv2.waitKey() & 0xff


def subtask3():
    faces = []
    arnolds = stack_arnolds(stack_2d=True)

    for g in orl_paths:
        # for g in george_paths:
        mat_gray = cv2.imread(g, cv2.IMREAD_GRAYSCALE)
        faces.append(mat_gray)

    mean_face = np.mean(faces, axis=0).astype(np.uint8)
    h, w = mean_face.shape
    # mean_face = cv2.medianBlur(mean_face, 5)
    cv2.imwrite("face_template.png", mean_face)

    arnolds_gray = cv2.cvtColor(arnolds, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(arnolds_gray, mean_face, cv2.TM_SQDIFF_NORMED)
    print(arnolds_gray.shape)
    print(result.shape)
    print(h, w)
    # print(arnolds.shape)
    # print(o.shape)
    threshold = 0.08
    good = np.argwhere(result > threshold)

    output = arnolds.copy()
    # output = output.reshape(*output.shape, 1) * [1, 1, 1]
    # output = np.array(output)

    mask = result < threshold
    pos = np.where(mask)
    # for x, y in pos:
    # print(v)
    # output = cv2.rectangle(output, [y, x], [y + h, x + w], (0, int(255), 0), 1)

    for pt in zip(*pos[::-1]):
        cv2.rectangle(output, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    plt.imshow(output)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # subtask1()
    # subtask2()
    subtask3()
