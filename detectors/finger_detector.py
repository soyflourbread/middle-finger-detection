import math

import cv2
import mediapipe as mp

from enum import Enum


# Represents three possible hand gestures.
class HandGesture(Enum):
    No = 0  # Could not recognize
    Point = 1  # Pointing fingers
    Yeah = 2  # The "Peace" sign
    Mid = 3  # The middle finger


# A class for detecting the middle finger.
class FingerDetector:
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    drawing_styles = mp.solutions.drawing_styles

    def __init__(self):
        self.model = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    # Classifying hand gestures and checking for middle fingers.
    # Returns an array of (hand landmarks, hand gesture).
    # See also https://google.github.io/mediapipe/solutions/hands.html.
    def detect(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.model.process(image)

        out = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger_stat = self._check_finger(hand_landmarks)
                out.append((hand_landmarks, finger_stat))

        return out

    # Draws hand landmarks on an image.
    # Hand landmarks is returned from FingerDetector.check().
    def draw_landmarks(self, image, landmarks):
        self.mp_drawing.draw_landmarks(
            image, landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.drawing_styles.get_default_hand_landmark_style(),
            self.drawing_styles.get_default_hand_connection_style())

    def mid_finger_region(self, image, hand_landmarks):
        l = hand_landmarks.landmark
        h, w = image.shape[:2]

        HandLandmark = self.mp_hands.HandLandmark

        tip_pos = int(l[HandLandmark.MIDDLE_FINGER_TIP].x * w), int(l[HandLandmark.MIDDLE_FINGER_TIP].y * h)
        mcp_pos = int(l[HandLandmark.MIDDLE_FINGER_MCP].x * w), int(l[HandLandmark.MIDDLE_FINGER_MCP].y * h)

        return tip_pos, mcp_pos

    def _check_finger(self, hand_landmarks, threshold=0.6):
        HandLandmark = self.mp_hands.HandLandmark

        mid_len = self._finger_len(hand_landmarks, HandLandmark.MIDDLE_FINGER_TIP)

        index_len = self._finger_len(hand_landmarks, HandLandmark.INDEX_FINGER_TIP)
        ring_len = self._finger_len(hand_landmarks, HandLandmark.RING_FINGER_TIP)
        pinky_len = self._finger_len(hand_landmarks, HandLandmark.PINKY_TIP)

        if index_len * threshold > max(mid_len, ring_len, pinky_len):
            return HandGesture.Point
        if mid_len * threshold > max(index_len, ring_len, pinky_len):
            return HandGesture.Mid
        if (index_len + mid_len) / 2 * threshold > max(ring_len, pinky_len):
            return HandGesture.Yeah

        return HandGesture.No

    @staticmethod
    def _finger_len(hand_landmarks, finger_tip):
        l = hand_landmarks.landmark

        width = l[finger_tip].x - l[finger_tip - 3].x
        height = l[finger_tip].y - l[finger_tip - 3].y

        return math.sqrt(width ** 2 + height ** 2)
