import numpy as np
import DSP_2t4r
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


for i in range(120, 121):
    chirp_num = 32
    tx_num = 2
    adc_sample = 64
    data = np.load('../data/0109/0105new' + str(i) + '.npy')
    data = data[:, 0:2:] + 1j * data[:, 2::]
    data = np.reshape(data, [chirp_num * tx_num, -1, adc_sample])
    data = data.transpose([0, 2, 1])
    ch1_data = data[0: 64: 2, :, :]
    ch3_data = data[1: 64: 2, :, :]
    data = np.concatenate([ch1_data, ch3_data], axis=2)
    # np.save('../data/reshape/0111_frame' + str(i) + '.npy', data)

no_window, window = data, data

rai_no_window = DSP_2t4r.Range_Angle(no_window, 1, [128, 64, 32], windowing=False)
rai_window = DSP_2t4r.Range_Angle(window, 1, [128, 64, 32])
# rdi_no_window = DSP_2t4r.Range_Doppler(no_window, 1, [128, 64], windowing=False)
# rdi_window = DSP_2t4r.Range_Doppler(window, 1, [128, 64])


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

plt.figure(1)
plt.subplot(1, 2, 1)
# plt.imshow(np.rot90(rai_window.sum(0), -2)[36:63], cmap=new_cmp) # , vmin=1.2e4, vmax=1.7e6)
plt.imshow(np.flip(np.rot90(rai_window[0, :, :], -2)[36:63], axis=1), cmap=new_cmp)
# plt.xticks([])
# plt.yticks([])
plt.xlabel("Angle")
plt.ylabel("Range")
# plt.xticks([0, 90, 180], ['-90', '0', '+90'])
# plt.yticks([0, 26], ['+', '0m'])
# plt.gca().xaxis.set_ticks_position('none')
# plt.gca().yaxis.set_ticks_position('none')
plt.title('With Hanning Window')

plt.subplot(1, 2, 2)
# plt.imshow(np.rot90(rai_no_window.sum(0), -2)[36:63], cmap=new_cmp)
plt.imshow(np.flip(np.rot90(rai_no_window[0, :, :], -2)[36:63], axis=1), cmap=new_cmp)

# plt.xticks([])
# plt.yticks([])
plt.xlabel("Angle")
plt.ylabel("Range")
# plt.xticks([0, 90, 180], ['-90', '0', '+90'])
# plt.yticks([0, 26], ['+', '0m'])
# plt.gca().xaxis.set_ticks_position('none')
# plt.gca().yaxis.set_ticks_position('none')
plt.title('No Hanning Window')


#
# plt.figure(2)
# plt.subplot(1, 2, 1)
# plt.imshow(np.rot90(rdi_window.sum(2), -2)[40:63, 45:80], cmap=new_cmp)
# # plt.xticks([])
# # plt.yticks([])
# plt.xlabel("Doppler")
# plt.ylabel("Range")
# # plt.xticks([0, 90, 180], ['-90', '0', '+90'])
# # plt.yticks([0, 26], ['+', '0m'])
# # plt.gca().xaxis.set_ticks_position('none')
# # plt.gca().yaxis.set_ticks_position('none')
# plt.title('With Hanning Window')
#
# plt.subplot(1, 2, 2)
# plt.imshow(np.rot90(rdi_no_window.sum(2), -2)[40:63, 45:80], cmap=new_cmp)
# # plt.xticks([])
# # plt.yticks([])
# plt.xlabel("Doppler")
# plt.ylabel("Range")
# # plt.xticks([0, 90, 180], ['-90', '0', '+90'])
# # plt.yticks([0, 26], ['+', '0m'])
# # plt.gca().xaxis.set_ticks_position('none')
# # plt.gca().yaxis.set_ticks_position('none')
# plt.title('No Hanning Window')

plt.tight_layout()
plt.show()
