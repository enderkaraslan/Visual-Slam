#!/usr/bin/env python3

from display import *
from extractor import *

W = 1920 // 2
H = 1080 // 2
disp = Display2D(W, H)

fe = Extractor()


def process_frame(img):
    img = cv2.resize(img, (W, H))
    matches = fe.extract(img)  

    print("%d matches" % (len(matches)))
    
    for pt1, pt2 in matches: 
        u1, v1 = map(lambda x: int(round(x)), pt1)
        u2, v2 = map(lambda x: int(round(x)), pt2)
        cv2.circle(img, (u1, v1), 3, (0, 255, 0), thickness=-1)
        cv2.line(img, (u1, v1), (u2, v2), (255, 0, 0), 1)
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
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