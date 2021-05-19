import numpy as np
import DSP
import os
from read_binfile import read_bin_file

radar_device = 'XWR1443'
output_data_type = 'RAI'

sense = [0, 1]
gesture = np.arange(12)
person = np.arange(12)
times = 12

# radar config
frame = 64
sample = 64
chirp = 32
tx_num = 3
rx_num = 4
config = [frame, sample, chirp, tx_num, rx_num]

# data path
data_folder = 'C:\\Users\\user\\Desktop\\new\\'
output_folder = 'C:\\Users\\user\\Desktop\\new\\'


if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

# read bin data & process

rdi_collect = []
rai_collect = []

for t in range(times):
    file_name = data_folder + 'file' + str(t) + '.bin'
    print('Reading file:', file_name)

    if radar_device == 'XWR1443':
        data = read_bin_file(file_name, config, mode=0)

    elif radar_device == 'XWR1843':
        data = read_bin_file(file_name, config, mode=1)

    rdi = []
    rai = []
    for f in range(frame):
        # tmp_data = np.concatenate([data[f, :, :, 0:4], data[f, :, :, 8:12]], axis=2)
        # tmp_data = np.concatenate([data[f, :, :, 0:4], data[f, :, :, 4:8]], axis=2)
        # tmp_data = np.concatenate([data[f, :, :, 0:4]], axis=2)
        tmp_data =data[f, :, :, 0:4]
        # TX1 + TX3
        if output_data_type == 'RDI':
            tmp_rdi = DSP.Range_Doppler(tmp_data, mode=1, padding_size=[128, 64])
            rdi.append(tmp_rdi[32:64, 48:80, :])

        elif output_data_type == 'RAI':
            tmp_rai = DSP.Range_Angle(tmp_data, 1, [128, 64, 32])
            rai.append(tmp_rai.sum(0)[32:64, :])

    if output_data_type == 'RDI':
        rdi_collect.append(rdi)

    elif output_data_type == 'RAI':
        rai_collect.append(rai)


print(np.shape(rai))
if output_data_type == 'RDI':
    save_filename = output_data_type + "total"
    print("Save File：", save_filename)
    np.save(output_folder + save_filename, rdi_collect)

elif output_data_type == 'RAI':
    save_filename = output_data_type + "total"
    print("Save File：", save_filename)
    np.save(output_folder + save_filename, rai_collect)
