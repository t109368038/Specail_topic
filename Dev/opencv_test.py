import cv2
import numpy as np

cam1 = cv2.VideoCapture(0)
cam2 = cv2.VideoCapture(1)

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out1 = cv2.VideoWriter('out1.avi', fourcc, 30, (200, 100))
out1 = cv2.VideoWriter('out2.avi', fourcc, 30, (200, 100))
img = np.zeros((320, 320, 3), np.uint8)

while True:
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()
    out1.write(frame1)
    cv2.imshow('Cam1', frame1)
    cv2.imshow('Cam2', frame2)
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
