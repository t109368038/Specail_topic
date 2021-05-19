import numpy as np
import arlpy as ar


total = np.zeros(20, dtype=int)
for i in range(110, 130):
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

    pos = [0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014]
    w = np.hanning(data.shape[1])
    new_w = np.array([w, w, w, w, w, w, w, w])
    steering = ar.bf.steering_plane_wave(pos, 3e8, np.linspace(-np.pi / 2, np.pi / 2, 181))

    single_chirp = data[12, :, :]
    tmp_data = new_w.T * single_chirp
    rdi = np.fft.fft(tmp_data, 256, axis=0)
    pos = [0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014]
    rdi = np.flip(rdi, axis=0)
    rai = ar.bf.bartlett(rdi[32, :].T, 81e9, steering)

    total[i - 110] = np.argmax(rai)


rad = np.linspace(-np.pi/2, np.pi/2, 181)
aoa = np.zeros(total.shape)
i = 0
for r in total:
    aoa[i] = rad[r]
    i += 1
aoa = np.rad2deg(aoa)

mean = np.mean(aoa)
std = np.std(aoa)

print("The 20 frame's first chirp:\nMean:", mean, '\nStd:', std)





