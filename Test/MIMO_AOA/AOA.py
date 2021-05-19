import numpy as np
import time
from read_binfile import read_bin_file
from remove_studio_header import remove_header
import matplotlib.pyplot as plt



# data path
data_folder = 'E:\\mmWave\\yanli_collect_data\\3t4r\\'
filename = data_folder + "adc_data_3t4r_0_0_0_001_process_0.bin"
# radar config
frame = 64
sample = 64
chirp = 32
tx_num = 3
rx_num = 4
config = [frame, sample, chirp, tx_num, rx_num]
data = read_bin_file(filename, config, mode=0)
# order TX1 TX2 TX3
frame_1 = data[0]
chirp_1 = frame_1[0]

# theta = 60
# phi = 50
f_0 = 78 * 10 **9
c_0 = 3 * 10 ** 8
antenna_d = 0.002
# order TX1 TX2 TX3
cos_matrix = np.array([0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7])
sin_matrix = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])

AOA_matrix = np.zeros([64, 181, 181])


# for phi in range(-90, 90):
#     for theta in range(-90, 90):
#         beam_matrix = -0.5 * (cos_matrix * np.cos(theta) + sin_matrix * np.sin(theta))
#         # print(beam_matrix)
#         # print('===============')
#         beam_factor = np.exp(-1j * beam_matrix + np.sin(phi))
#         output = np.matmul(chirp_1, beam_factor)
#         AOA_matrix[phi, theta] = sum(abs(output))
#         # time.sleep(1)


data_folder = '../data/AOA_test_data/'
filename = 'adc_data_aoa_10_Raw_0.bin'
aoa_data = remove_header(data_folder + filename, 4322)

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



aoa_frame_1 = cdata[30]
aoa_chirp_1 = aoa_frame_1[15]
count = 0
for phi in range(0, 60):
    for theta in range(0, 60):
        beam_matrix = antenna_d * (cos_matrix * np.cos(np.pi*theta/180) + sin_matrix * np.sin(np.pi*theta/180)) /c_0
        # print(beam_matrix)
        # print('===============')
        beam_factor = np.exp(-2 * 1j * beam_matrix * np.pi* np.sin(np.pi*phi/180))
        # output = np.matmul(aoa_chirp_1, beam_factor)
        for sample in range(64):
            tmp = aoa_chirp_1[sample,0:4] * beam_factor[0:4]
            tmp = tmp * np.conjugate(tmp)
            tmp = tmp.sum()
            AOA_matrix[sample, phi, theta] = tmp



