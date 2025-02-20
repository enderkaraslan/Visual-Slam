#!/usr/bin/env python3

import cv2
from display import *
import numpy as np

W = 1920 // 2
H = 1080 // 2
disp = Display2D(W, H)

class FeatureExtractor(object):
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1000)
    
    def extract(self, img):
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)
        return feats
fe = FeatureExtractor()
def process_frame(img):

    img = cv2.resize(img, (W, H))
    kp = fe.extract(img)
    for p in kp:
        u,v =map(lambda x: int(round(x)), p[0])
        cv2.circle(img, (u, v), 3, (0, 255, 0), thickness=-1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Renk dönüşümü
    disp.paint(img)


if __name__ == '__main__':
    cap = cv2.VideoCapture("test.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            process_frame(frame)
            pygame.time.delay(delay)  # FPS'e göre gecikme
        else:
            break

    cap.release()
    cv2.destroyAllWindows()