import cv2
import numpy as np

from detectors import finger_detector


def mark_fingers(image, proc_image):
    checker = finger_detector.FingerDetector()
    out = checker.detect(image)
    for (landmarks, stat) in out:
        checker.draw_landmarks(proc_image, landmarks)


def process(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.GaussianBlur(im, (5, 5), 0)
    _, im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    im = cv2.erode(im, np.ones((4, 4), np.uint8), iterations=1)

    return im


def post_process(im):
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    # im[:, :, :2] = 80
    im[:, :, 0] = 80

    return im


im = cv2.imread("img/1.png")
# im = imutils.resize(im, width=720)
im_proc = cv2.imread("img/1.png")
mark_fingers(im, im_proc)
# cv2.imshow("IM", im_proc)
# cv2.waitKey()
cv2.imwrite("out/art-1.png", im_proc)

