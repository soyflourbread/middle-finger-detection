import cv2
import numpy as np


class BackgroundDetector:
    def __init__(self, delta=0.05, threshold=0.05):
        self.background = None
        self.delta = delta
        self.threshold = threshold

    def detect(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.background is None:
            self.background = frame
            return False, None

        is_positive, thresh = self._detect(frame)
        if not is_positive:
            self.background = self.background * (1 - self.delta) + frame * self.delta
            self.background = self.background.astype(np.uint8)

        return is_positive, thresh

    def _detect(self, frame):
        diff = cv2.absdiff(self.background, frame)
        _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, np.ones((5, 5), np.uint8))

        return np.mean(thresh) > self.threshold, thresh
