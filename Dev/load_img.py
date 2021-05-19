import numpy as np
import matplotlib.pyplot as plt
import cv2


img = np.load('../data/img/800.npy')
# img = cv2.imread('../data/img/5.npy')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("Window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

