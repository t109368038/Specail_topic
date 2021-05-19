import numpy as np
import matplotlib.pyplot as plt
import h5py
from read_binfile import read_bin_file
from remove_studio_header import remove_header


def read_data(filename):
    frame = 64
    sample = 64
    chirp = 32
    tx_num = 3
    aoa_data = remove_header(filename, 4322)
    aoa_data = np.reshape(aoa_data, [-1, 8])
    aoa_data = aoa_data[:, 0:4:] + 1j * aoa_data[:, 4::]
    cdata1 = np.reshape(aoa_data[:, 0], [frame, chirp, tx_num, sample])
    cdata1 = np.transpose(cdata1, [0, 1, 3, 2])  # frame, chirp, sample, channel
    cdata2 = np.reshape(aoa_data[:, 1], [frame, chirp, tx_num, sample])
    cdata2 = np.transpose(cdata2, [0, 1, 3, 2])  # frame, chirp, sample, channel
    cdata3 = np.reshape(aoa_data[:, 2], [frame, chirp, tx_num, sample])
    cdata3 = np.transpose(cdata3, [0, 1, 3, 2])  # frame, chirp, sample, channel
    cdata4 = np.reshape(aoa_data[:, 3], [frame, chirp, tx_num, sample])
    cdata4 = np.transpose(cdata4, [0, 1, 3, 2])  # frame, chirp, sample, channel
    cdata = np.array([cdata1[:, :, :, 0], cdata2[:, :, :, 0], cdata3[:, :, :, 0], cdata4[:, :, :, 0],
                      cdata1[:, :, :, 1], cdata2[:, :, :, 1], cdata3[:, :, :, 1], cdata4[:, :, :, 1],
                      cdata1[:, :, :, 2], cdata2[:, :, :, 2], cdata3[:, :, :, 2], cdata4[:, :, :, 2]])
    cdata = np.transpose(cdata, [1, 2, 3, 0])
    return cdata


data_folder = 'D:/1090918data/'
file_name = data_folder + 'adc_data_45_0_2_Raw_0.bin'
data = read_data(file_name)
data_copy = data
output = []
# for i in range(64):
i = 40
window = np.reshape(np.hanning(data[i].shape[1]), (data[i].shape[1], -1))
for j in range(12):
    data[i, :, :, j] = data[i, :, :, j] * window.T
range_bin = np.fft.fft(data[i], n=128, axis=1)
output.append(range_bin)
range_bin = np.array(range_bin)
test_chirp = range_bin[0, :, 0:4]
range_bin = np.abs(range_bin)

# hf = h5py.File('../data/test_chirp/chirp_1.h5', 'w')
# hf.create_dataset('chirp_1', data=test_chirp)
# hf.close()

#
# plt.figure(1)
# plt.imshow(range_bin[:, :, 0].T)
#
#
# f_0 = 77 * (10 ** 9)
# c_0 = 3 * (10 ** 8)
# antenna_d = 0.002
# j = 0
# f_c = f_0 + (121.134 * (j * (10 ** 6) / 2))
# common_factor = (-1j * 2 * np.pi * f_c) / c_0
# sin_para = np.arange(4, dtype='f')
# result_matrix = np.zeros([181, 4])
#
# for theta in range(-90, 91):
#     weights = np.exp(common_factor *
#                      (antenna_d * sin_para * np.sin(theta * np.pi / 180)))
#     print(np.shape(weights))
#     print(np.shape(test_chirp))
#     tmp = np.multiply(test_chirp, weights)
#     tmp_abs = np.abs(tmp)
#     tmp_power = tmp_abs ** 2
#     # result_matrix[theta + 90] = tmp
#
#
