import numpy as np
import matplotlib.pyplot as plt
from read_binfile import read_bin_file
from remove_studio_header import remove_header
from read_binfile import read_bin_file
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D


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


def coordinate_transform(datapoint):
    angle = np.angle(datapoint, True)
    rd = np.abs(datapoint)
    output = [angle, rd]
    output = np.reshape(output, [2, -1]).T
    return output


# Needs to remove header
# data_folder = "../data/AOA_test_data/"
# file_name = data_folder + 'adc_data_aoa_10_Raw_0.bin'
data_folder = 'D:/1090918data/'
file_name = data_folder + 'adc_data_45_0_2_Raw_0.bin'
data = read_data(file_name)

# radar config
frame = 64
sample = 64
chirp = 32
tx_num = 3
rx_num = 4
config = [frame, sample, chirp, tx_num, rx_num]

# Don't need to remove header
# data_folder = 'D:/mmWave/yanli_collect_data/3t4r/'
# file_name = data_folder + 'adc_data_3t4r_0_0_8_005_process_0.bin'
# data = read_bin_file(file_name, config, mode=0)

f_0 = 77 * (10 ** 9)
c_0 = 3 * (10 ** 8)
antenna_d = 0.002
# order TX1 TX2 TX3
cos_matrix = np.array([0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7], dtype='f')
sin_matrix = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], dtype='f')
# save power result
AOA_matrix = np.zeros([64, 181, 181])

frame_0 = data[0]
chirp_0 = frame_0[0]

# plot original data
result_matrix = np.zeros([64, 181, 181, 4], dtype=np.complex128)
for j in range(64):
    f_c = f_0 + (121.134 * (j * (10 ** 6) / 2))
    common_factor = (-1j * 2 * np.pi * f_c) / c_0
    # rr = coordinate_transform(np.concatenate([chirp_0[j, 0:4], chirp_0[j, 8:12]], axis=0))
    # plt.figure(1)
    # plt.clf()
    # i = 0
    # for x, y in rr:
    #     plt.polar([0, x], [0, y], marker='o')
    #     plt.text(x, y, 'rx' + str(i), horizontalalignment='center', verticalalignment='bottom')
    #     i += 1
    #     # set Y tick label to white
    #     # plt.yticks(ticks=[100, 200, 300, 400, 500, 600], labels='')
    # plt.title("Origin sample:")
    # plt.show()

    # beamforming
    # test_chirp = np.concatenate([chirp_0[j, 0:4]], axis=0)
    test_chirp = chirp_0[j, 0:4]
    # result_matrix = np.zeros([181, 181, 8], dtype=np.complex128)
    cos_para = np.arange(4, dtype='f')
    for phi in range(-90, 91):
        for theta in range(-90, 91):
            weights = np.exp(common_factor *
                             (antenna_d * cos_para * np.sin(theta * np.pi / 180)))
            tmp = np.multiply(test_chirp, weights)
            result_matrix[j, phi + 90, theta + 90] = tmp

            # result_0 = result_matrix[i, j, :]

            # plot result
            # tt = coordinate_transform(tmp)
            # plt.figure(2)
            # plt.clf()
            # i = 0
            # for x, y in tt:
            #     plt.polar([0, x], [0, y], marker='o')
            #     plt.text(x, y, 'rx' + str(i), horizontalalignment='center', verticalalignment='bottom')
            #     i += 1
            #     # set Y tick label to white
            #     # plt.yticks(ticks=[100, 200, 300, 400, 500, 600], labels='')
            # plt.title("After Beamforming:")
            # plt.show()
fig = plt.figure()
# for j in range(64):
#     img_ = np.multiply(result_matrix[j], result_matrix[j].conj())
#     img_ = np.real(img_)
#     plt.clf()
#     # img_ = np.log10(img_) * 10
#     img_ = np.sum(img_, axis=2)
#
#     plt.imshow(img_, cmap='gray_r')
#     plt.colorbar()
#     plt.show()

for j in range(64):
    plt.clf()
    img_ = np.multiply(result_matrix[j], result_matrix[j].conj())
    img_ = np.real(img_)
    img_ = np.sum(img_, axis=2)

    ax = Axes3D(fig)
    x = np.arange(-90, 90)
    y = np.arange(-90, 90)
    x, y = np.meshgrid(x, y)
    p = ax.plot_surface(x, y, img_[x + 90, y + 90], cmap=plt.get_cmap('rainbow'))
    ax.set_xlabel("phi")
    ax.set_ylabel("theta")
    plt.show()
    # ax.contourf(x,y,img_[x+90,y+90], zdir='z', cmap=plt.get_cmap('rainbow'))
    fig.colorbar(p)

# AOA_matrix = 10 * np.log10(AOA_matrix)
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# for i in range(-90, 91):
#     for j in range(-90, 91):
#         ax.scatter(AOA_matrix[0, i + 90, j + 90 ], i, j)
# plt.show()
