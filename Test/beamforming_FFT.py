import numpy as np
import DSP
import matplotlib.pyplot as plt


folder = 'D:/pycharm_project/real-time-radar/data/small_object_data/'
filename = 'minus30.npy'

# folder = 'D:/pycharm_project/real-time-radar/data/Real_time_data/'
# filename = 'Frame83.npy'

# folder = 'D:/pycharm_project/real-time-radar/real-time/'
# filename = 'frame2'

data = np.load(folder + filename)

RAI = DSP.Range_Angle(data, 1, [128, 64, 32])
RDI_raw, RDI = DSP.Range_Doppler(data, 2, [32, 64])
RDI_reshape = RDI_raw.reshape([-1, 4])
# RDI_padding = np.pad(RDI_reshape, ((0, 0), (0, 60)), 'constant', constant_values=(0, 0))

# RDI_raw = np.transpose(np.fft.fftshift(RDI_raw, axes=0), [1, 0, 2])
# RDI_raw = np.flip(RDI_raw, axis=0)

# plt.figure(1)
# plt.imshow(RAI[20, :, :])
# plt.figure(2)
# plt.imshow(RDI[:, :, 1])

# rx1 = np.reshape(RDI_raw[:, :, 0], [-1, 1])
# rx2 = np.reshape(RDI_raw[:, :, 1], [-1, 1])
# rx3 = np.reshape(RDI_raw[:, :, 2], [-1, 1])
# rx4 = np.reshape(RDI_raw[:, :, 3], [-1, 1])
# rx = np.array([rx1[7360], rx2[7360], rx3[7360], rx4[7360]])
#
Fc = 77e9
lambda_start = 3e8 / Fc
weights_matrix = np.zeros([181, 4], dtype=complex)
count_2 = 0
phi_matrix = np.zeros([51, 181, 4], dtype=complex)
for phi in range(-25, 26):

    count = 0
    for theta in range(-90, 91):
        d = 0.5 * lambda_start * np.sin(theta * np.pi / 180) * np.cos(phi * np.pi / 180)
        beamforming_factor = np.array([0, d, 2 * d, 3 * d]) / (3e8 / Fc)
        weights_matrix[count, :] = np.exp(-1j * 2 * np.pi * beamforming_factor)
        count += 1
    phi_matrix[count_2] = weights_matrix
    count_2 += 1


out_matrix = np.zeros([51, 2048, 181], dtype=complex)
out_matrix_reshape = np.zeros([51, 32, 64, 181], dtype=complex)
for j in range(51):
    for i in range(2048):
        out_matrix[j, i, :] = np.matmul(phi_matrix[j], RDI_reshape[i, :])

    out_matrix_reshape[j] = out_matrix[j].reshape([32, 64, -1])

# out_matrix_reshape = np.fft.fftshift(out_matrix_reshape, axes=2)

# for k in range(10):
plt.figure()
y_label = [0, 32, 63]
x_label = [0, 90, 180]
plt.imshow(np.flip(np.abs(out_matrix_reshape[25, 0])), aspect='auto')
plt.xticks(x_label, ['-90', '0', '+90'])
plt.xlabel('Angle($^\circ$)')
plt.yticks(y_label, ['2.4', '1.2', '0'], rotation=45)
plt.ylabel('Range(m)', rotation=45)
plt.tick_params(axis='both', length=0)
plt.title('Beamforming')
    # plt.figure(2)
    # for j in range(64):
    #     plt.plot(np.abs(out_matrix_reshape[0, j, :]))
    #     plt.legend("Distance " + str(j))


# # out = np.abs(out)
# ang = np.angle(weights_matrix)
# plt.plot(ang)
# # plt.plot(np.abs(out[0]))
