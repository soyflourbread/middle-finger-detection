import cv2

from detectors import background_detector

detector = background_detector.BackgroundDetector()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    is_positive, thresh = detector.detect(image)

    if is_positive:
        cv2.imwrite("out/1.png", image)
        cv2.imwrite("out/2.png", detector.background)
        cv2.imwrite("out/3.png", thresh)

    cv2.imshow("WIN", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
