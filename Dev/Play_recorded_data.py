import numpy as np
import matplotlib.pyplot as plt
from DSP_2t4r import *
import time

datapath = 'E:/ResearchData/ThuMouseData/TEST/'
filename = 'Null_0308_rawdata.npy'

data = np.load(datapath + filename)

num_frames = data.shape[0]
num_chirps = 32
num_rx = 4
num_tx = 3
num_samples = 64
adc_data = []
rdi = []

for f in range(num_frames):
    tmp_data = data[f]
    ret = np.zeros(len(tmp_data) // 2, dtype=complex)
    ret[0::2] = tmp_data[0::4] + 1j * tmp_data[2::4]
    ret[1::2] = tmp_data[1::4] + 1j * tmp_data[3::4]
    ret = ret.reshape([num_chirps * num_tx, num_rx, num_samples])
    adc_data.append(ret.transpose([0, 2, 1]))

adc_data = np.array(adc_data)
for f in range(num_frames):
    tx1 = adc_data[:, 0::3, :, :]
    tx3 = adc_data[:, 1::3, :, :]
    tx2 = adc_data[:, 2::3, :, :]



for f in range(num_frames):
    tmp_rdi = Range_Doppler(tx1[f], 1, padding_size=[128, 64])
    rdi.append(tmp_rdi)

for f in range(num_frames):
    plt.figure()
    plt.matshow(rdi[f][:, :, 0])
    plt.savefig('E:/ResearchData/ThuMouseData/TEST/tx1_fig/' + str(f))
    plt.close()
