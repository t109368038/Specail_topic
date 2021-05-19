import read_binfile
import numpy as np
import mmwave as mm
from mmwave.dsp.utils import Window
import matplotlib.pyplot as plt

datapath = 'E:/ResearchData/ThuMouseData/0315index_finger_rawdata.npy'

data = np.load(datapath)
