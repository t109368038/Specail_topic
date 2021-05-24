import threading as th
import numpy as np
import socket
import DSP_2t4r
import mmwave as mm
from mmwave.dsp.utils import Window
from PyQt5.QtCore import QThread, pyqtSignal



# class UdpListener(th.Thread):
#     def __init__(self, name, bin_data, data_frame_length, data_address, buff_size, save_data):
class UdpListener(QThread):
    bindata_signal = pyqtSignal(list)
    rawdata_signal = pyqtSignal(list)
    def __init__(self, name, data_frame_length, data_address, buff_size,bindata,rawdata):

        super(UdpListener, self).__init__()
        # th.Thread.__init__(self, name=name)
        """
        :param name: str
                        Object name
        :param bin_data: queue object
                        A queue used to store adc data from udp stream
        :param data_frame_length: int
                        Length of a single frame
        :param data_address: (str, int)
                        Address for binding udp stream, str for host IP address, int for host data port
        :param buff_size: int
                        Socket buffer size
        """
        self.bin_data = []
        self.frame_length = data_frame_length
        self.data_address = data_address
        self.buff_size = buff_size
        self.save_data = []
        self.status = 0
        self.bindata = bindata
        self.rawdata = rawdata

    def run(self):
        # convert bytes to data type int16
        dt = np.dtype(np.int16)
        dt = dt.newbyteorder('<')
        # array for putting raw data
        np_data = []
        # count frame
        count_frame = 0
        data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data_socket.bind(self.data_address)
        print("Create Data Socket Successfully")
        print("Waiting For The Data Stream")
        print('=======================================')
        # main loop
        while True:
            data, addr = data_socket.recvfrom(self.buff_size)
            data = data[10:]
            np_data.extend(np.frombuffer(data, dtype=dt))
            # while np_data length exceeds frame length, do following
            if len(np_data) >= self.frame_length:
                count_frame += 1
                if self.status == 1:
                    self.rawdata.put(np_data[0:self.frame_length])
                    # self.rawdata_signal.emit(np_data[0:self.frame_length])
                # print(np_data[0:self.frame_length])
                self.bindata.put(np_data[0:self.frame_length])
                # remove one frame length data from array
                np_data = np_data[self.frame_length:]


class DataProcessor(QThread):
# class DataProcessor(th.Thread):
    data_signal = pyqtSignal(np.ndarray,np.ndarray,np.ndarray)

    def __init__(self, name, config, bin, file_name=0, status=0):
        super(DataProcessor, self).__init__()
        # th.Thread.__init__(self, name=name)

        """
        :param name: str
                        Object name
        :param config: sequence of ints
                        Radar config in the order
                        [0]: samples number
                        [1]: chirps number
                        [3]: transmit antenna number
                        [4]: receive antenna number
        :param bin_queue: queue object
                        A queue for access data received by UdpListener
        :param rdi_queue: queue object
                        A queue for store RDI
        :param rai_queue: queue object
                        A queue for store RDI
        """
        self.adc_sample = config[0]
        self.chirp_num = config[1]
        self.tx_num = config[2]
        self.rx_num = config[3]
        self.filename = file_name
        self.status = status
        # self.weight_matrix = np.zeros([181, 8], dtype=complex)
        self.weight_matrix = np.zeros([181, 8], dtype=complex)
        self.weight_matrix1 = np.zeros([181, 2], dtype=complex)
        self.out_matrix = np.zeros([8192, 181], dtype=complex)
        self.out_matrix1 = np.zeros([8192, 181], dtype=complex)
        self.bindata = bin
        self.Sure_staic_RM = False
        Fc = 60
        count = 0
        lambda_start = 3e8 / Fc
        for theta in range(-90, 91):
            d = 0.5 * lambda_start * np.sin(theta * np.pi / 180)
            beamforming_factor = np.array([0, d, 2 * d, 3 * d, 4 * d, 5 * d, 6 * d, 7 * d]) / (3e8 / Fc)
            beamforming_factor1 = np.array([0, d, 2 * d, 3 * d]) / (3e8 / Fc)
            beamforming_factor1 = np.array([0, d]) / (3e8 / Fc)
            self.weight_matrix[count, :] = np.exp(-1j * 2 * np.pi * beamforming_factor)
            self.weight_matrix1[count, :] = np.exp(-1j * 2 * np.pi * beamforming_factor1)
            count += 1



    def run(self):
        while True:

            frame_count = 0

            range_resolution, bandwidth = mm.dsp.range_resolution(66,2000,121.134)
            doppler_resolution = mm.dsp.doppler_resolution(bandwidth, 60, 33.02, 9.43, 16, 3)


            data = self.bindata.get()
            data = np.reshape(data, [-1, 4])
            data = data[:, 0:2:] + 1j * data[:, 2::]
            raw_data = np.reshape(data, [self.chirp_num * self.tx_num, -1, self.adc_sample])

            radar_cube = mm.dsp.range_processing(raw_data, window_type_1d=Window.HANNING)
            assert radar_cube.shape == (
                48, 4, 64), "[ERROR] Radar cube is not the correct shape!" #(numChirpsPerFrame, numRxAntennas, numADCSamples)

            # (3) Doppler Processing
            det_matrix, aoa_input = mm.dsp.doppler_processing(radar_cube, num_tx_antennas=3,
                                                           clutter_removal_enabled=self.Sure_staic_RM,
                                                           # clutter_removal_enabled=True,
                                                           # clutter_removal_enabled=False,
                                                           window_type_2d=Window.HANNING, accumulate=True)

            det_matrix_vis = np.fft.fftshift(det_matrix, axes=1)


            def Range_Angle(data, padding_size):
                rai_abs = np.fft.fft(data, n=padding_size, axis=2)
                rai_abs = np.fft.fftshift(np.abs(rai_abs), axes=2)
                rai_abs = np.flip(rai_abs, axis=1)
                return rai_abs

            # (4) Angle Processing sample, channel, chirp
            azimuth_ant_1 = aoa_input[:, :2 * 4, :]
            azimuth_ant_2 = aoa_input[:, 2 * 4:, :]
            for ixx in range(4):
                elevation_ant_1 = aoa_input[:, 2+ixx, :]
                elevation_ant_2 = aoa_input[:, 8+ixx, :]
                elevation_combine = np.array([elevation_ant_1, elevation_ant_2]).transpose([1, 0, 2])
                elevation_combine = elevation_combine.transpose([2, 0, 1])
                tmp = Range_Angle(elevation_combine, 90)
                if ixx == 0:
                    elevation_map = tmp.copy()
                else:
                    elevation_map += tmp

            # (4-1) Range Angle change to chirps, samples, channels
            azimuth_ant_1 = azimuth_ant_1.transpose([2, 0, 1])
            azimuth_map = Range_Angle(azimuth_ant_1, 90)


            # (5) Object Detection
            fft2d_sum = det_matrix.astype(np.int64)


            thresholdDoppler, noiseFloorDoppler = np.apply_along_axis(func1d=mm.dsp.ca_,
                                                                      axis=0,
                                                                      arr=fft2d_sum.T,
                                                                      l_bound=1.5,
                                                                      guard_len=2,
                                                                      noise_len=4)

            thresholdRange, noiseFloorRange = np.apply_along_axis(func1d=mm.dsp.ca_,
                                                                  axis=0,
                                                                  arr=fft2d_sum,
                                                                  l_bound=2.5,
                                                                  guard_len=2,
                                                                  noise_len=4)

            thresholdDoppler, noiseFloorDoppler = thresholdDoppler.T, noiseFloorDoppler.T

            det_doppler_mask = (det_matrix > thresholdDoppler)
            det_range_mask = (det_matrix > thresholdRange)


            # Get indices of detected peaks
            full_mask = (det_doppler_mask & det_range_mask)
            det_peaks_indices = np.argwhere(full_mask == True)

            # peakVals and SNR calculation
            peakVals = fft2d_sum[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
            snr = peakVals - noiseFloorRange[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]

            dtype_location = '(' + str(3) + ',)<f4' # 3 == numTxAntennas
            dtype_detObj2D = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
                                       'formats': ['<i4', '<i4', '<f4', dtype_location, '<f4']})
            detObj2DRaw = np.zeros((det_peaks_indices.shape[0],), dtype=dtype_detObj2D)
            detObj2DRaw['rangeIdx'] = det_peaks_indices[:, 0].squeeze()
            detObj2DRaw['dopplerIdx'] = det_peaks_indices[:, 1].squeeze()
            detObj2DRaw['peakVal'] = peakVals.flatten()
            detObj2DRaw['SNR'] = snr.flatten()

            detObj2DRaw = mm.dsp.prune_to_peaks(detObj2DRaw, det_matrix, 16, reserve_neighbor=True) # 16 = numDopplerBins

            # --- Peak Grouping
            detObj2D = mm.dsp.peak_grouping_along_doppler(detObj2DRaw, det_matrix, 16) # 16 = numDopplerBins

            # SNRThresholds2 = np.array([[2, 23], [10, 11.5], [35, 16]])
            # peakValThresholds2 = np.array([[2, 275], [1, 400], [500, 0]])
            SNRThresholds2 = np.array([[0, 15], [10, 16], [0 , 20]])
            peakValThresholds2 = np.array([[0, 20], [10, 0], [0 , 0]])

            detObj2D = mm.dsp.range_based_pruning(detObj2D, SNRThresholds2, peakValThresholds2, 59, 50, # 64== numRangeBins
                                               range_resolution)

            azimuthInput = aoa_input[detObj2D['rangeIdx'], :, detObj2D['dopplerIdx']]
            # print(np.shape(detObj2D['dopplerIdx']))
            Psi, Theta, Ranges, velocity, xyzVec = mm.dsp.beamforming_naive_mixed_xyz(azimuthInput,
                                                                                   63-detObj2D['rangeIdx'],
                                                                                   detObj2D['dopplerIdx'],
                                                                                   range_resolution,
                                                                                   method='Bartlett')

            # self.rdi_queue.put(np.flip(det_matrix_vis))
            # self.rai_queue.put(np.flip(azimuth_map.sum(0)))
            # # self.rai_queue.put(np.flip(elevation_map.sum(0))/4)
            # self.pd_queue.put(xyzVec)

            self.data_signal.emit(np.flip(det_matrix_vis), np.flip(azimuth_map.sum(0)), xyzVec)