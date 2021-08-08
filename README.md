# middle-finger-detection

This repoository contains the source code for https://pyseez.com/2021/08/middle-finger-detection/.

This code detects simple hand gestures (pointing, peace sign, etc) as well as the middle finger
by using OpenCV and pretrained models in MediaPipe.
If the finger is detected, this program tries to censor it.

To test the code, run `python main.py`
after installing OpenCV 4 (with contrib) and MediaPipe dependencies.
