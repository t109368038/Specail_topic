import sys

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mmwave.clustering as clu
import mmwave.dsp as dsp
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os



def get_gt(x1,y1,x2,y2):
    scale_x1 = 0.000875*1.5
    scale_y1 = 0.000916*1.5
    scale_x2 = 0.000703125*1.5
    scale_y2 = 0.0006875*1.5
    # if (x1 != None).all() == True:
    x1 = np.where(x1 != None, x1, np.double(0))

    x1 = x1.astype(np.double)
    x1 = np.round((x1 * scale_x1), 3)
    x1 = x1 - ((640 * scale_x1 / 2)) - 0.3
        # print(x1)
    if (y1 != None).all() == True:
        y1 = y1.astype(np.double)
        y1 = np.round((y1 * scale_y1), 3)
        y1 -= ((480 * scale_y1) / 2)
    # if (x2 != None).all() == True:
    x2 = np.where(x2 != None, x2, np.double(320))
    x2 = x2.astype(np.double)
    x2 = (x2 * scale_x2)
    x2 -= ((640 * scale_x2) / 2)
    x2 = np.round(x2, 3)

    y2 = np.where(y2 != None, y2, np.double(480))
    # if (y2 != None).all() == True:
    y2 = y2.astype(np.double)
    y2 = (y2 * scale_y2)
    y2 -= ((480 * scale_y2) / 2)
    y2 = np.round(y2, 3)


    print(y2)
    return x2, y2+0.33, y1