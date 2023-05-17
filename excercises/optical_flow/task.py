import numpy as np
import cv2 as cv
import imutils
import time
import os

from itertools import product

# parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
#                                               The example file can be downloaded from: \
#                                               https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
# parser.add_argument('image', type=str, help='path to image file')
# print("parsed")
# args = parser.parse_args()
cv2 = cv

# print(args)
FILE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(FILE_DIR)

highway_path = FILE_DIR + os.path.sep + \
               'A3 A-road Traffic UK HD - rush hour - British Highway traffic May 2017.mp4'

video_path = "racinggames.avi"
cap = cv.VideoCapture(video_path)

"params for ShiTomasi corner detection"
CORNERS = 80
"GRID"
PIXEL_GAP = 20
"Display"
HMAX = 600


def lk_algo():
    cap = cv.VideoCapture(highway_path)
    # pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, pos + 200)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 1600)  # sunshine
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 800)  # car behind sign
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1800)  # sunshine
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 151)  # car behind sign
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 2422)  # truck
    feature_params = dict(maxCorners=CORNERS,
                          qualityLevel=0.3,
                          minDistance=5,
                          blockSize=20)

    "Parameters for lucas kanade optical flow"
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (CORNERS, 3))

    "Take first frame and find corners in it"
    ret, old_frame = cap.read()
    old_frame = imutils.resize(old_frame, height=HMAX)
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    h, w, _ = old_frame.shape

    x0 = list(range(0, w, PIXEL_GAP))
    y0 = list(range(0, h, PIXEL_GAP))

    x = x0[1:-1]
    y = y0[1:-1]

    # points = list(product(x, y))
    # p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    "Create a mask image for drawing purposes"
    mask = np.zeros_like(old_frame, dtype=np.uint8)
    BLEND = 1
    pause = False
    while True:
        k = cv.waitKey(15) & 0xff
        if k == ord('q'):
            break
        elif k == 32:
            "space"
            pause = not pause
        if pause:
            time.sleep(0.2)
            continue

        ret, frame = cap.read()
        frame = imutils.resize(frame, height=HMAX)
        # print(frame.shape, frame.dtype)

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        "calculate optical flow"
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # p1 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        # print(st.shape)
        "Select good points"
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            # print(good_new.shape)

        "draw the tracks"
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        mask = (mask * BLEND).astype(np.uint8)
        img = cv.add(frame, mask)
        cv.imshow('frame', img)

        "Now update the previous frame and previous points"
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        missing = CORNERS - len(p0)

        # print(f"missing: {missing}")
        if missing > 1 and False:
            ftrs = feature_params.copy()
            ftrs['maxCorners'] = missing
            p_miss = cv2.goodFeaturesToTrack(frame_gray, **ftrs)
            p0 = np.concatenate([p0, p_miss], axis=0)

        print(len(p0))


def guner_farn():
    HMAX = 300
    cap = cv.VideoCapture(highway_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1600)
    ret, frame1 = cap.read()
    frame1 = imutils.resize(frame1, height=HMAX)
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    pause = False

    while True:
        k = cv.waitKey(15) & 0xff
        if k == ord('q'):
            break
        elif k == 32:
            "space"
            pause = not pause
        if pause:
            time.sleep(0.2)
            continue

        ret, frame2 = cap.read()
        frame2 = imutils.resize(frame2, height=HMAX)
        if not ret:
            print('No frames grabbed!')
            break
        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        mean = cv2.add(frame2, bgr)
        cv.imshow('Guner-Farn', mean)
        prvs = next


def lk_grid():
    LINE_THICK = 1
    BLEND = 0.999
    PIXEL_GAP = 25
    # BLEND = 1

    cap = cv.VideoCapture(highway_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 2422)  # truck
    cap.set(cv2.CAP_PROP_POS_FRAMES, 422)  #
    # feature_params = dict(maxCorners=CORNERS,
    #                       qualityLevel=0.3,
    #                       minDistance=5,
    #                       blockSize=20)

    "Parameters for lucas kanade optical flow"
    lk_params = dict(winSize=(10, 10),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 5, 0.03))

    "Take first frame and find corners in it"
    ret, old_frame = cap.read()
    old_frame = imutils.resize(old_frame, height=HMAX)
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

    h, w, _ = old_frame.shape
    x0 = np.arange(0, w, PIXEL_GAP, dtype=np.float32).reshape(1, -1)
    y0 = np.arange(0, h, PIXEL_GAP, dtype=np.float32).reshape(-1, 1)

    X, Y = np.meshgrid(x0, y0)
    XY = np.array([*zip(X.ravel(), Y.ravel())], dtype=np.float32).reshape(-1, 1, 2)

    "Create a mask image for drawing purposes"
    mask = np.zeros_like(old_frame, dtype=np.uint8)
    pause = False
    p0 = XY.copy()
    momentum = np.zeros((len(p0), 2))

    while True:
        k = cv.waitKey(15) & 0xff
        if k == ord('q'):
            break
        elif k == 32:
            "space"
            pause = not pause
        if pause:
            time.sleep(0.2)
            continue

        ret, frame = cap.read()
        frame = imutils.resize(frame, height=HMAX)
        # print(frame.shape, frame.dtype)

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        "calculate optical flow"
        p0 = XY.copy()
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # p1 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        # print(st.shape)
        "Select good points"
        # if p1 is not None:
        #     good_new = p1[st == 1]
        #     good_old = p0[st == 1]
        #     # print(momentum.shape,st.shape)
        #     force = momentum[st.ravel() == 1]
        #     # print(good_new.shape)
        # else:
        #     continue

        "draw the tracks"
        for i, (new, old, force_delta) in enumerate(zip(p1, p0, momentum)):
            new = new.ravel()
            old = old.ravel()

            a, b = new
            # c, d = old.ravel()
            diff = new - old
            # diff = np.clip(diff, -5, 5)
            moment = diff * 0.1 + force_delta * 0.9
            momentum[i] = moment
            c, d = new + moment
            # mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 1, (10, 30, 60), -1)
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), (150, 150, 55), LINE_THICK)

        mask = (mask * BLEND).astype(np.uint8)
        img = cv.add(frame, mask)
        # img = np.clip(frame + mask, 0, 255)
        cv.imshow('LK-Grid', img)

        "Now update the previous frame and previous points"
        old_gray = frame_gray.copy()
        # p0 = good_new.reshape(-1, 1, 2)


def lk_grid_color():
    LINE_THICK = 1
    BLEND = 0.999
    PIXEL_GAP = 25
    # BLEND = 1

    cap = cv.VideoCapture(highway_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 2422)  # truck
    cap.set(cv2.CAP_PROP_POS_FRAMES, 422)  #
    # feature_params = dict(maxCorners=CORNERS,
    #                       qualityLevel=0.3,
    #                       minDistance=5,
    #                       blockSize=20)

    "Parameters for lucas kanade optical flow"
    lk_params = dict(winSize=(30, 30),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 5, 0.03))

    "Take first frame and find corners in it"
    ret, old_frame = cap.read()
    old_frame = imutils.resize(old_frame, height=HMAX)

    h, w, _ = old_frame.shape
    x0 = np.arange(0, w, PIXEL_GAP, dtype=np.float32).reshape(1, -1)
    y0 = np.arange(0, h, PIXEL_GAP, dtype=np.float32).reshape(-1, 1)

    X, Y = np.meshgrid(x0, y0)
    XY = np.array([*zip(X.ravel(), Y.ravel())], dtype=np.float32).reshape(-1, 1, 2)

    "Create a mask image for drawing purposes"
    mask = np.zeros_like(old_frame, dtype=np.uint8)
    pause = False
    p0 = XY.copy()
    momentum = np.zeros((len(p0), 2))

    while True:
        k = cv.waitKey(15) & 0xff
        if k == ord('q'):
            break
        elif k == 32:
            "space"
            pause = not pause
        if pause:
            time.sleep(0.2)
            continue

        ret, frame = cap.read()
        frame = imutils.resize(frame, height=HMAX)
        # print(frame.shape, frame.dtype)
        # frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        "calculate optical flow"
        p0 = XY.copy()
        p1, st, err = cv.calcOpticalFlowPyrLK(old_frame, frame, p0, None, **lk_params)

        "draw "
        for i, (new, old, force_delta) in enumerate(zip(p1, p0, momentum)):
            new = new.ravel()
            old = old.ravel()

            a, b = new
            # c, d = old.ravel()
            diff = new - old
            # diff = np.clip(diff, -5, 5)
            # moment = diff * 0.1 + force_delta * 0.9
            # momentum[i] = moment
            c, d = new + diff
            # mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 1, (10, 30, 60), -1)
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), (150, 150, 55), LINE_THICK)

        mask = (mask * BLEND).astype(np.uint8)
        img = cv.add(frame, mask)
        # img = np.clip(frame + mask, 0, 255)
        cv.imshow('LK-Color', img)

        "Now update the previous frame and previous points"
        old_frame = frame.copy()
        # p0 = good_new.reshape(-1, 1, 2)


lk_algo()
guner_farn()
lk_grid()
lk_grid_color()
