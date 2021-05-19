import matplotlib.pyplot as plt
import numpy as np
import os
import DSP_2t4r
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from read_binfile import read_bin_file
from remove_studio_header import remove_header


def read_data(filename, rm_header=True):
    frame = 64
    sample = 64
    chirp = 32
    tx_num = 3
    if rm_header:
        aoa_data = remove_header(filename, 4322)
    else:
        aoa_data = np.fromfile(filename, dtype=np.int16)

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


data_type = 'RAI'
# folder = '../data/py_process_data/' + data_type + '/'
# filename = 'adc_data_1t4r_0_0_0_002_processed.npy'
folder = 'D:/1090918data/'
filename = 'adc_data_45_0_2_Raw_0.bin'
# folder = 'D:/yen_li/2019畢_嚴勵/data/bin file_processed/new data(low powered)/3t4r/'
# filename = 'adc_data_3t4r_0_0_4_007_process_0.bin'

output_folder = '../data/py_process_data/RAI_FULL_plot/45_test_g4/'
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
# data = np.load(folder + filename)
# data = read_data(folder + filename, rm_header=False)

data = read_data(folder + filename, rm_header=True)
or_data = data
output = []
output_rdi = []
output_or = []
for f in range(64):
    tmp = DSP_2t4r.Range_Angle(data[f, :, :, 0:4], 1, [128, 64, 32])
    tmp2 = DSP_2t4r.Range_Doppler(data[f, :, :, 0:4], 0, [32, 64])
    output.append(tmp.sum(0)[:, :])
    output_or.append(tmp)
    output_rdi.append(tmp2)

data = np.array(output)
output_or = np.array(output_or)
data_rdi = np.array(output_rdi)

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
# for i in range(64):
#     plt.figure(i)
#     plt.imshow(data[i], cmap=new_cmp)
#     plt.savefig(output_folder + 'FULL_RAI_G4_frame_' + str(i + 1))
#     plt.close()

# fig = plt.figure()
# ax = Axes3D(fig)
# x = np.arange(0, 64)
# y = np.arange(0, 32)
# x, y = np.meshgrid(x, y)
#
# y3 = np.arctan2(x, y)
# ax.scatter(x, y, data[20], c=y3, s=50)
# # ax.set_xticks([0, 63])
# # ax.set_xticklabels(['+', '0m'])
# plt.show()




# for i in range(64):
#     plt.figure()
#     plt.imshow(data[i, :, :], cmap=new_cmp)
#     plt.savefig(output_folder + 'rai_g3' + str(i+1))
#     plt.close()

