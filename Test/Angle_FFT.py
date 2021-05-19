import DSP
import numpy as np
import h5py
import matplotlib.pyplot as plt


folder = 'D:/pycharm_project/real-time-radar/data/small_object_data/'
filename = 'minus30.npy'

data = np.load(folder + filename)
data = np.load(folder + filename)

# rdi = DSP.Range_Doppler(data, 0, [128, 64])
# rdi_reshape = rdi.reshape([-1, 4])
# rdi_padding = np.pad(rdi_reshape, ((0, 0), (0, 60)), 'constant', constant_values=(0, 0))
# FFT_matrix = np.zeros((64, 64), dtype=complex)
# OUT_matrix = np.zeros((8192, 64), dtype=complex)
# for k in range(1, 65):
#     for n in range(1, 65):
#         w = np.exp(-2 * np.pi * 1j / 64)
#         FFT_matrix[k - 1, n - 1] = pow(w, ((k - 1) * (n - 1)))
#
# for i in range(8192):
#     OUT_matrix[i, :] = np.matmul(FFT_matrix, rdi_padding[i, :])
#
# OUT_matrix_reshape = OUT_matrix.reshape([128, 64, 64])
# OUT_matrix_reshape = np.fft.fftshift(OUT_matrix_reshape, axes=2)
#
#
# plt.figure(1)
# plt.imshow(np.abs(OUT_matrix_reshape[0]))

# =============================================
# Plot RDI
rdi = DSP.Range_Doppler(data, 1, [64, 64])
# f1 = open('../data/CFAR/RDI.bin', 'wb')
# rdi = np.array(rdi, dtype=np.int16)
# rdi = rdi.reshape([-1, 1])
# f1.write(rdi)
# f1.close()

# plt.figure(1)
# y_label = [0, 32, 63]
# x_label = [0, 32, 63]
# plt.imshow(np.abs(rdi[:, :, 0]), aspect='auto')
# plt.xticks(x_label, ['-30', '0', '+30'])
# plt.xlabel('Velocity(m/s)')
# plt.yticks(y_label, ['2.4', '1.2', '0'], rotation=45)
# plt.ylabel('Range(m)', rotation=45)
# plt.tick_params(axis='both', length=0)
# plt.title('Range-Doppler with padding')
# # =============================================
# rdi_np = DSP.Range_Doppler(data, 1)
# plt.figure(2)
# y_label = [0, 32, 63]
# x_label = [0, 16, 31]
# plt.imshow(np.abs(rdi_np[:, :, 0]), aspect='auto')
# plt.xticks(x_label, ['-30', '0', '+30'])
# plt.xlabel('Velocity(m/s)')
# plt.yticks(y_label, ['2.4', '1.2', '0'], rotation=45)
# plt.ylabel('Range(m)', rotation=45)
# plt.tick_params(axis='both', length=0)
# plt.title('Range-Doppler no padding')
# =============================================
# Plot RAI
# rai = DSP.Range_Angle(data, 1, [64, 64, 32])
# plt.figure(3)
# y_label = [0, 32, 63]
# x_label = [0, 16, 31]
# plt.imshow(np.abs(rai[0, :, :]), aspect='auto')
# plt.xticks(x_label, ['-90', '0', '+90'])
# plt.xlabel('Angle($^\circ$)')
# plt.yticks(y_label, ['2.4', '1.2', '0'], rotation=45)
# plt.ylabel('Range(m)', rotation=45)
# plt.tick_params(axis='both', length=0)
# plt.title('Angle FFT with padding')
# # =============================================
# rai_np = DSP.Range_Angle(data, 1)
# plt.figure(4)
# y_label = [0, 32, 63]
# x_label = [0, 1.5, 3]
# plt.imshow(np.abs(rai[0, :, :]), aspect='auto')
# plt.imshow(np.abs(rai_np[0, :, :]), aspect='auto')
# plt.xticks(x_label, ['-90', '0', '+90'])
# plt.xlabel('Angle($^\circ$)')
# plt.yticks(y_label, ['2.4', '1.2', '0'], rotation=45)
# plt.ylabel('Range(m)', rotation=45)
# plt.tick_params(axis='both', length=0)
# plt.title('Angle FFT no padding')





