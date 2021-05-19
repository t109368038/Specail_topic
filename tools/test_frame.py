import cv2


cam = cv2.VideoCapture(0)
cam = cam.set(cv2.CAP_PROP_FPS, 20)
print(cam.get(5))

while True:
