from remove_studio_header import remove_header
import numpy as np
import matplotlib.pyplot as plt
import os


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


data_folder = 'D:/1090918data/'
file_name = 'adc_data_0_0_Raw_0.bin'
data = read_data(data_folder + file_name)
# data_folder = 'D:/yen_li/2019畢_嚴勵/data/bin file_processed/new data(low powered)/3t4r/'
# file_name = 'adc_data_3t4r_0_0_4_001_process_0.bin'
# data = read_data(data_folder + file_name, rm_header=False)
# output_folder = '../data/py_process_data/RAI_FULL_plot/45_aoa_test_g4/'
# if not os.path.isdir(output_folder):
#     os.makedirs(output_folder)

# i = 20
# output = []
# window = np.reshape(np.hanning(data[i].shape[1]), (data[i].shape[1], -1))
# for j in range(12):
#     data[i, :, :, j] = data[i, :, :, j] * window.T
# range_bin = np.fft.fft(data[i], n=64, axis=1)
# output.append(range_bin)
# range_bin = np.array(range_bin)
# chirp_data_0 = range_bin[29, :, 0:4]

# chirp_data_0 = data[0, 23, :, 0:4]
Fc = 77e9
lambda_start = 3e8 / Fc
weights_matrix = np.zeros([91, 4], dtype=np.complex)
chirp_data_0 = data[0, 0, :, 0:4]
for theta in range(0, 91):
    d = 0.5 * lambda_start * np.sin(theta * np.pi / 180)
    beamforming_factor = np.array([0, d, 2 * d, 3 * d])
    weights_matrix[theta, :] = np.exp(-1j * 2 * np.pi * beamforming_factor / (Fc / 3e8))

out = []
plt.figure()
for i in range(64):
    out = np.matmul(data[i, 12, :, 0:4], weights_matrix.T)
    plt.plot(np.abs(out[20]).T)
    plt.show()
    plt.clf()
# angle_matrix = np.zeros([64, 64, 91])
# for i in range(64):
#     chirp_data_0 = data[i, 0, :, 0:4]
#
#     # beamforming
#
#     theta = 0
#
#     # lambda_start = 0.002
#     for theta in range(1, 91):
#         d = 0.5 * lambda_start * np.sin(theta * np.pi / 180)
#         beamforming_factor = np.array([0, d, 2 * d, 3 * d])
#         tmp_matrix = np.zeros(shape=chirp_data_0.shape, dtype=np.complex)
#         tmp_sum = np.zeros(shape=chirp_data_0.shape[0], dtype=np.complex)
#         tmp_abs = np.zeros(shape=chirp_data_0.shape[0])
#         for s in range(chirp_data_0.shape[0]):
#             weights = np.exp(-1j * 2 * np.pi * beamforming_factor / (3e8 / Fc))
#             look1 = weights
#             look2 = chirp_data_0[s]
#             tmp_matrix[s, :] = chirp_data_0[s] * weights
#             tmp_sum[s] = sum(tmp_matrix[s, :])
#             # tmp_abs[s] = abs(tmp_sum[s]) ** 2
#         angle_matrix[i, :, theta] = np.real(np.abs(tmp_sum))
#
# # plt.figure(i)
# # plt.plot(angle_matrix[i].T)
# # plt.xticks([0, 46, 91, 136, 181], ['-90', '-45', '0', '+45', '+90'])
# # plt.savefig(output_folder + "FULL_G4_frame_" + str(i + 1))
# # plt.close()
