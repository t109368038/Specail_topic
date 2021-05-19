import numpy as np
import DSP
import matplotlib.pyplot as plt

data_folder = "D:/pycharm_project/real-time-radar/data/Real_time_data/"
file1 = "Frame200.npy"  # -45
file2 = "Frame400.npy"  # 0
file3 = "Frame700.npy"  # +45

data1 = np.load(data_folder + file1)
data2 = np.load(data_folder + file2)
data3 = np.load(data_folder + file3)

rdi_1 = DSP.Range_Doppler(data1, mode=1, padding_size=[128, 64])
plt.figure(1)
plt.imshow(np.flip(rdi_1[:, :, 0]))

rdi_2 = DSP.Range_Doppler(data2, mode=1, padding_size=[128, 64])
plt.figure(2)
plt.imshow(np.flip(rdi_2[:, :, 0]))

rdi_3_raw, rdi_3 = DSP.Range_Doppler(data3, mode=2, padding_size=[128, 64])
plt.figure(3)
plt.imshow(np.flip(rdi_3[:, :, 0]))

rdi_3_raw = np.transpose(np.fft.fftshift(rdi_3_raw, axes=0), [1, 0, 2])

rai_1 = DSP.Range_Angle(data1, mode=1, padding_size=[128, 64, 32])
plt.figure(4)
plt.imshow(np.flip(rai_1[0, :, :]))

rai_2 = DSP.Range_Angle(data2, mode=1, padding_size=[128, 64, 32])
plt.figure(5)
plt.imshow(np.flip(rai_2[0, :, :]))

rai_3 = DSP.Range_Angle(data3, mode=1, padding_size=[128, 64, 32])
plt.figure(6)
plt.imshow(np.flip(rai_3[0, :, :]))

