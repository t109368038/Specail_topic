import numpy as np
import arlpy as ar
import matplotlib.pyplot as plt
import DSP_2t4r
from matplotlib.colors import ListedColormap

chirp_num = 32
tx_num = 2
adc_sample = 64
# data = np.load('../data/0109/0105new122.npy')
data = np.load('../data/0112/0105new958.npy')
data = data[:, 0:2:] + 1j * data[:, 2::]
data = np.reshape(data, [chirp_num * tx_num, -1, adc_sample])
data = data.transpose([0, 2, 1])
ch1_data = data[0: 64: 2, :, :]
ch3_data = data[1: 64: 2, :, :]
data = np.concatenate([ch1_data, ch3_data], axis=2)


w = np.hanning(data.shape[1])
new_w = np.array([w, w, w, w, w, w, w, w])


single_chirp = data[20, :, :]
# single_chirp = data[12, :, :]
tmp_data = new_w.T * single_chirp
rdi = np.fft.fft(tmp_data, 256, axis=0)

pos = [0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014]


rdi_raw, rdi_fft = DSP_2t4r.Range_Doppler(data, 2, [128, 64])
rai_fft = DSP_2t4r.Range_Angle(data, 1, [128, 64, 32])

steering = ar.bf.steering_plane_wave(pos, 3e8, np.linspace(-np.pi/2, np.pi/2, 181))
rai_bartlett = np.zeros([128, 64, 181], dtype=np.complex)

rdi = np.flip(rdi, axis=0)

# for i in range(128):
#     for j in range(64):
#         rai_bartlett[i, j, :] = ar.bf.music(rdi_raw[i, j, :].T, 77e9, steering)
# rai_bartlett = np.abs(rai_bartlett)
# r = ar.bf.bartlett(rdi_raw[32, :].T, 77e9, steering)

rai = ar.bf.bartlett(rdi[10, :].T, 77e9, steering)
# rai = ar.bf.capon(rdi[32, :].T, 79e9, steering)
# rai = ar.bf.music(rdi[32, :].T, 81e9, steering)


# my beamforming
# weight_matrix = np.zeros([181, 8], dtype=complex)
# out_matrix = np.zeros([8192, 181], dtype=complex)
# Fc = 77.2e9
# count = 0
# lambda_start = 3e8 / Fc
# for theta in range(-90, 91):
#     d = 0.5 * lambda_start * np.sin(theta * np.pi / 180)
#     beamforming_factor = np.array([0, d, 2 * d, 3 * d, 4 * d, 5 * d, 6 * d, 7 * d]) / (3e8 / Fc)
#     weight_matrix[count, :] = np.exp(-1j * 2 * np.pi * beamforming_factor)
#     count += 1
# rdi_raw = rdi_raw.reshape([-1, 8])
# for i in range(8192):
#     out_matrix[i, :] = np.matmul(weight_matrix, rdi_raw[i, :])
# my_bf = out_matrix.reshape([128, 64, -1])
# my_bf = np.abs(my_bf)
#


for i in range(1):
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(np.abs(rdi[:, 0:4]))
    plt.ylim((0, 4000))
    plt.legend(['rx0', 'rx1', 'rx2', 'rx3'])
    plt.xlabel('range-bin')
    # plt.xticks([])
    plt.title('Range-FFT TX0')

    plt.subplot(2, 1, 2)
    plt.plot(np.abs(rdi[:, 4:8]))
    plt.ylim((0, 4000))
    plt.legend(['rx4', 'rx5', 'rx6', 'rx7'])
    plt.xlabel('range-bin')
    # plt.xticks([])
    plt.title('Range-FFT TX2')
    # plt.plot(rdi_fft[64, :, 0])
    # plt.imshow(np.flip(rdi_fft[:, :, 0], axis=0))
    plt.tight_layout()
    plt.show()


colors = [[62, 38, 168, 255], [63, 42, 180, 255], [65, 46, 191, 255], [67, 50, 202, 255], [69, 55, 213, 255],
          [70, 60, 222, 255], [71, 65, 229, 255], [70, 71, 233, 255], [70, 77, 236, 255], [69, 82, 240, 255],
          [68, 88, 243, 255],
          [68, 94, 247, 255], [67, 99, 250, 255], [66, 105, 254, 255], [62, 111, 254, 255], [56, 117, 254, 255],
          [50, 123, 252, 255],
          [47, 129, 250, 255], [46, 135, 246, 255], [45, 140, 243, 255], [43, 146, 238, 255], [39, 150, 235, 255],
          [37, 155, 232, 255],
          [35, 160, 229, 255], [31, 164, 225, 255], [24, 173, 219, 255], [17, 177, 214, 255],
          [7, 181, 208, 255],
          [1, 184, 202, 255], [2, 186, 195, 255], [11, 189, 188, 255], [24, 191, 182, 255], [36, 193, 174, 255],
          [44, 195, 167, 255],
          [49, 198, 159, 255], [55, 200, 151, 255], [63, 202, 142, 255], [74, 203, 132, 255], [88, 202, 121, 255],
          [102, 202, 111, 255],
          [116, 201, 100, 255], [130, 200, 89, 255], [144, 200, 78, 255], [157, 199, 68, 255], [171, 199, 57, 255],
          [185, 196, 49, 255],
          [197, 194, 42, 255], [209, 191, 39, 255], [220, 189, 41, 255], [230, 187, 45, 255], [239, 186, 53, 255],
          [248, 186, 61, 255],
          [254, 189, 60, 255], [252, 196, 57, 255], [251, 202, 53, 255], [249, 208, 50, 255], [248, 214, 46, 255],
          [246, 220, 43, 255],
          [245, 227, 39, 255], [246, 233, 35, 255], [246, 239, 31, 255], [247, 245, 27, 255], [249, 251, 20, 255]]
colors = np.array(colors) / 256
new_cmp = ListedColormap(colors)

# plt.figure(2)
# plt.title('(AoA)Angle-FFT')
# # plt.plot(rai_fft[0, :, :])
# plt.imshow(np.flip(np.rot90(rai_fft[0, :, :], -2)[36:63], axis=1), cmap=new_cmp) #, vmin=1e3, vmax=1e5
# # plt.plot(rai_fft)


plt.figure(3)
plt.plot(rai)
plt.title('(AOA)Bartlett Beamforming')
# plt.xticks([0, 45, 60, 90, 120, 135, 180], ['-90', '-45', '-30', '0', '30', '45', '90'])
# plt.imshow(my_bf[0, 36:63, :], cmap=new_cmp)
plt.show()
#
# plt.figure(4)
# plt.plot(rai)
# plt.title('(AOA)Capon Beamforming')
# plt.xticks([0, 45, 60, 90, 120, 135, 180], ['-90', '-45', '-30', '0', '30', '45', '90'])
# # plt.imshow(np.flip(rai_bartlett[0, :, :], axis=[0, 1]))
# plt.show()



# plt.figure(5)
# plt.plot(rai)
# plt.title('(AOA)MUSIC')
# plt.xticks([0, 45, 60, 90, 120, 135, 180], ['-90', '-45', '-30', '0', '30', '45', '90'])
# # plt.imshow(np.flip(rai_bartlett[0, :, :], axis=[0, 1]))
# plt.show()



