import read_binfile
import numpy as np
import matplotlib.pyplot as plt

# data_path = 'C:/ti/mmwave_studio_02_00_00_02/mmWaveStudio/PostProc/'
# file_name = 'adc_data_Raw_0.bin'
data_path = '../data/0109/'

file_name = ['0105new110.npy', '0105new111.npy', '0105new112.npy', '0105new113.npy', '0105new114.npy']

# radar config
frame = 1
sample = 64
chirp = 32
tx_num = 1
rx_num = 4
config = [frame, sample, chirp, tx_num, rx_num]
# data = read_binfile.read_bin_file(data_path + file_name, config, 1, False, 1441)
data_sum = np.zeros([32, 64, 8], dtype=np.complex)

for i in file_name:
    data = np.load(data_path + i)
    tmp = data[:, 0:2:] + 1j * data[:, 2::]
    tmp = np.reshape(tmp, [32*2, -1, 64])
    tmp = tmp.transpose([0, 2, 1])
    ch1_data = tmp[0: 64: 2, :, :]
    ch3_data = tmp[1: 64: 2, :, :]
    tmp = np.concatenate([ch1_data, ch3_data], axis=2)
    data_sum += tmp
tmp = tmp / 5

dB_tmp = 10 * np.log10(np.abs(tmp))

w = np.hanning(64)
for k in range(8):
    tmp[0, :, k] = w.T * tmp[0, :, k]



plt.figure()
plt.subplot(2, 1, 1)
p0, = plt.plot(10*np.log10(np.abs(tmp[0, :, 0])))
p1, = plt.plot(10*np.log10(np.abs(tmp[0, :, 1])))
p2, = plt.plot(10*np.log10(np.abs(tmp[0, :, 2])))
p3, = plt.plot(10*np.log10(np.abs(tmp[0, :, 3])))
plt.ylim((0, 45))
plt.ylabel('Amplitude')
plt.xlabel('Sample')
plt.legend([p0, p1, p2, p3], ['rx0', 'rx1', 'rx2', 'rx3'], loc=9)
plt.title('Sample TX0')
plt.subplot(2, 1, 2)
p4, = plt.plot(10*np.log10(np.abs(tmp[0, :, 4])))
p5, = plt.plot(10*np.log10(np.abs(tmp[0, :, 5])))
p6, = plt.plot(10*np.log10(np.abs(tmp[0, :, 6])))
p7, = plt.plot(10*np.log10(np.abs(tmp[0, :, 7])))
plt.ylim((0, 45))
plt.ylabel('Amplitude')
plt.xlabel('Sample')
plt.legend([p4, p5, p6, p7], ['rx4', 'rx5', 'rx6', 'rx7'], loc=9)
plt.title('Sample TX2')
plt.tight_layout()


plt.show()
