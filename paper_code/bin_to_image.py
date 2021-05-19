import numpy as np
import DSP_2t4r
import os
from read_binfile import read_bin_file

radar_device = 'XWR1443'
output_data_type = 'RAI'

sense = [0, 1]
gesture = np.arange(12)
person = np.arange(12)
times = 10

# radar config
frame = 64
sample = 64
chirp = 32
tx_num = 3
rx_num = 4
config = [frame, sample, chirp, tx_num, rx_num]

# data path
data_folder = 'D:/yen_li/2019畢_嚴勵/data/bin file_processed/new data(low powered)/3t4r/'
output_folder = '../data/py_process_data/' + output_data_type + '/'

if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

# read bin data & process
for s in sense:
    for g in gesture:
        print('Sense:', s, 'Gesture:', g)
        rdi_collect = []
        rai_collect = []
        for p in person:
            for t in range(times):
                file_name = data_folder + 'adc_data_3t4r_' + str(s) + '_' + str(p) + '_' + str(g) + '_00' + \
                            str(t + 1) + '_process_0.bin'
                print('Reading file:', file_name)

        #       bin to np data
                data = read_bin_file(file_name, config, mode=0)
                np.save('../data/adc_data_3t4r_np_format/np_3t4r{0}_{1}_{2}_00{3}'.format(str(s), str(p),
                                                                                          str(g), str(t + 1)), data)

        #         bin to image
        #         data = []
        #         if radar_device == 'XWR1443':
        #             data = read_bin_file(file_name, config, mode=0)
        #
        #         elif radar_device == 'XWR1843':
        #             data = read_bin_file(file_name, config, mode=1)
        #
        #         rdi = []
        #         rai = []
        #         for f in range(frame):
        #             # tmp_data = np.concatenate([data[f, :, :, 0:4], data[f, :, :, 8:12]], axis=2)
        #             # tmp_data = np.concatenate([data[f, :, :, 0:4], data[f, :, :, 4:8]], axis=2)
        #             tmp_data = data[f, :, :, 0:4]
        #             # tmp_data = data[f]
        #             # TX1 + TX3
        #             if output_data_type == 'RDI':
        #                 tmp_rdi = DSP.Range_Doppler(tmp_data, mode=1, padding_size=[128, 64])
        #                 rdi.append(tmp_rdi[32:64, 48:80, :])
        #
        #             elif output_data_type == 'RAI':
        #                 tmp_rai = DSP.Range_Angle(tmp_data, 1, [128, 64, 32])
        #                 rai.append(tmp_rai.sum(0)[32:64, :])
        #
        #         if output_data_type == 'RDI':
        #             rdi_collect.append(rdi)
        #             np.save(output_folder + 'adc_data_1t4r_' + str(s) + '_' + str(p) + '_' + str(g) + '_00' +
        #                     str(t + 1) + '_processed', rdi)
        #
        #         elif output_data_type == 'RAI':
        #             rai_collect.append(rai)
        #             np.save(output_folder + 'adc_data_1t4r_' + str(s) + '_' + str(p) + '_' + str(g) + '_00' +
        #                     str(t + 1) + '_processed', rai)
        #
        # # if output_data_type == 'RDI':
        # #     save_filename = output_data_type + '_S' + str(s) + 'G' + str(g)
        # #     print("Save File：", save_filename)
        # #     np.save(output_folder + save_filename, rdi_collect)
        # #
        # # elif output_data_type == 'RAI':
        # #     save_filename = output_data_type + '_S' + str(s) + 'G' + str(g)
        # #     print("Save File：", save_filename)
        # #     np.save(output_folder + save_filename, rai_collect)
