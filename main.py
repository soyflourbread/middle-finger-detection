import cv2

import numpy as np

from detectors import finger_detector


ft = cv2.freetype.createFreeType2()
ft.loadFontData(fontFileName='/usr/share/fonts/cantarell/Cantarell-Thin.otf', id=0)


def main_fn():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = finger_detector.FingerDetector()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        result = detector.detect(image)
        for (landmarks, finger_stat) in result:
            if finger_stat == finger_detector.HandGesture.Point:
                ft.putText(image, "One", (200, 200), 120, (255, 255, 255), 2, cv2.LINE_AA, True)
            elif finger_stat == finger_detector.HandGesture.Yeah:
                ft.putText(image, "Two", (200, 200), 120, (255, 255, 255), 2, cv2.LINE_AA, True)
            elif finger_stat == finger_detector.HandGesture.Mid:
                reg = detector.mid_finger_region(image, landmarks)
                cv2.line(image, reg[0], reg[1], (255, 255, 255), 64)
                cv2.line(image, reg[0], reg[1], (0, 0, 0), 48)
                cv2.imwrite("out/middle.png", image)
            else:
                detector.draw_landmarks(image, landmarks)

        cv2.imshow('Friendly Python', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()


if __name__ == "__main__":
    cv2.namedWindow("Friendly Python", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Friendly Python", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    main_fn()
