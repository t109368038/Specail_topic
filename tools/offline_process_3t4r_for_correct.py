import numpy as np
import mmwave as mm
from mmwave.dsp.utils import Window

class DataProcessor_offline():
    def __init__(self):
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
        self.weight_matrix = np.zeros([181, 8], dtype=complex)
        self.weight_matrix1 = np.zeros([181, 2], dtype=complex)
        self.out_matrix = np.zeros([1024, 181], dtype=complex)
        self.out_matrix1 = np.zeros([8192, 181], dtype=complex)
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

    def run_proecss(self,raw_data,RAI_mode,Sure_staic_RM,chirp):
        frame_count = 0
        while True:

            range_resolution, bandwidth = mm.dsp.range_resolution(128)
            doppler_resolution = mm.dsp.doppler_resolution(bandwidth, 60, 33.02, 9.43, 16, 3)

            raw_data = np.reshape(raw_data,[-1,4,64])
            # print(np.shape(raw_data))
            radar_cube = mm.dsp.range_processing(raw_data, window_type_1d=Window.HANNING)
            assert radar_cube.shape == (
                48, 4, 64), "[ERROR] Radar cube is not the correct shape!" #(numChirpsPerFrame, numRxAntennas, numADCSamples)

            # (3) Doppler Processing
            det_matrix, aoa_input = mm.dsp.doppler_processing(radar_cube, num_tx_antennas=3,
                                                              clutter_removal_enabled=Sure_staic_RM,
                                                              window_type_2d=Window.HANNING, accumulate=True)

            det_matrix_vis = np.fft.fftshift(det_matrix, axes=1)


            # (4) Angle Processing sample, channel, chirp
            azimuth_ant_1 = aoa_input[:, :2 * 4, :]
            azimuth_ant_2 = aoa_input[:, 2 * 4:, :]
            elevation_ant_1 = aoa_input[:, 2, :]
            elevation_ant_2 = aoa_input[:, 8, :]
            elevation_combine = np.array([elevation_ant_1, elevation_ant_2]).transpose([1, 0, 2])

            # (4-1) Range Angle change to chirps, samples, channels
            azimuth_ant_1 = azimuth_ant_1.transpose([2, 0, 1])
            elevation_combine = elevation_combine.transpose([2, 0, 1])

            def Range_Angle(data, padding_size):
                rai_abs = np.fft.fft(data, n=padding_size, axis=2)
                rai_abs = np.fft.fftshift(np.abs(rai_abs), axes=2)
                rai_abs = np.flip(rai_abs, axis=1 )
                return rai_abs

            if RAI_mode == 0:
                azimuth_map = Range_Angle(azimuth_ant_1, 90)
                elevation_map = Range_Angle(elevation_combine, 90)

            elif RAI_mode == 1:
                print(np.shape(azimuth_ant_1))
                rdi_raw = azimuth_ant_1.reshape([-1, 8])
                for i in range(64*chirp):
                    self.out_matrix[i, :] = np.matmul(self.weight_matrix, rdi_raw[i, :])
                rai = self.out_matrix.reshape([chirp, 64, -1])
                rai = np.flip(np.abs(rai), axis=1)

                azimuth_map = np.abs(rai)

            elif RAI_mode == 2:
                # pass
                azimuth_map = Range_Angle(azimuth_ant_1, 90)
                # print("azimuth_map:{}".format(np.shape(azimuth_map)))
                ang_thresholdDoppler, ang_noiseFloorDoppler = np.apply_along_axis(func1d=mm.dsp.ca_,
                                                                          axis=0,
                                                                          arr=azimuth_map.T,
                                                                          l_bound=4000,
                                                                          guard_len=16,
                                                                          noise_len=32)

                ang_thresholdRange, ang_noiseFloorRange = np.apply_along_axis(func1d=mm.dsp.ca_,
                                                                      axis=0,
                                                                      arr=azimuth_map,
                                                                      l_bound=4000,
                                                                      guard_len=16,
                                                                      noise_len=32)

                ang_thresholdDoppler, ang_noiseFloorDoppler = ang_thresholdDoppler.T, ang_noiseFloorDoppler.T

                ang_det_doppler_mask = (azimuth_map > ang_thresholdDoppler)
                ang_det_range_mask = (azimuth_map > ang_thresholdRange)
                # print("ang_det_doppler_mask:{}".format(np.shape(ang_det_doppler_mask)))
                # print("ang_det_range_mask:{}".format(np.shape(ang_det_range_mask)))
                azimuth_map=ang_det_doppler_mask
                # Get indices of detected peaks
                azimuth_map = (ang_det_doppler_mask & ang_det_range_mask)
                # azimuth_map = np.argwhere(full_mask == True)
                # print("azimuth_map:{}".format(np.shape(azimuth_map)))

            # (5) Object Detection
            fft2d_sum = det_matrix.astype(np.int64)


            thresholdDoppler, noiseFloorDoppler = np.apply_along_axis(func1d=mm.dsp.ca_,
                                                                      axis=0,
                                                                      arr=fft2d_sum.T,
                                                                      l_bound=5,
                                                                      guard_len=2,
                                                                      noise_len=4)

            thresholdRange, noiseFloorRange = np.apply_along_axis(func1d=mm.dsp.ca_,
                                                                  axis=0,
                                                                  arr=fft2d_sum,
                                                                  l_bound=5,
                                                                  guard_len=5,
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

            detObj2DRaw = mm.dsp.prune_to_peaks(detObj2DRaw, det_matrix, chirp, reserve_neighbor=True) # 16 = numDopplerBins
            # --- Peak Grouping
            detObj2D = mm.dsp.peak_grouping_along_doppler(detObj2DRaw, det_matrix, 16) # 16 = numDopplerBins

            SNRThresholds2 = np.array([[2, 23], [10, 11.5], [35, 16]])
            peakValThresholds2 = np.array([[2, 275], [1, 400], [500, 0]])
            # SNRThresholds2 = np.array([[0, 15], [10, 16], [0 , 20]])
            # SNRThresholds2 = np.array([[0, 20], [10, 0], [0 , 0]])

            detObj2D = mm.dsp.range_based_pruning(detObj2D, SNRThresholds2, peakValThresholds2, 58, 55, # 64== numRangeBins
                                                  range_resolution)

            azimuthInput = aoa_input[detObj2D['rangeIdx'], :, detObj2D['dopplerIdx']]


            # print(np.shape(detObj2D['dopplerIdx']))
            Psi, Theta, Ranges, velocity, xyzVec = mm.dsp.beamforming_naive_mixed_xyz(azimuthInput,
                                                                                      63-detObj2D['rangeIdx'],
                                                                                      63-detObj2D['dopplerIdx'],
                                                                                      range_resolution,
                                                                                      method='Bartlett')
            # print("psi = {}".format(Psi))
            # print("Theta = {}".format(Theta))
            # print(xyzVec)
            if RAI_mode==2:
                # det_matrix_vis  =np.fft.fftshift(full_mask, axes=1)
                # det_matrix_vis= det_matrix_vis&full_mask
                # print(full_mask)
                # print(np.shape(np.where(np.any(full_mask == False))))
                # det1 =  azimuth_ant_1
                det1 = np.zeros([16,64,8])
                mask_tupe = (np.where(full_mask==True))
                x =list(mask_tupe[0])
                y =list(mask_tupe[1])

                for i in range(len(x)):
                    det1[y[i],x[i],:]  = azimuth_ant_1[y[i],x[i],:]
                # det1 = np.fft.fftshift(det1, axes=1)
                azimuth_map = Range_Angle(det1, 90)
            # return  np.flip(det_matrix_vis),np.flip(azimuth_map),xyzVec
            # else:
            # return  det_matrix_vis,azimuth_map.sum(0),xyzVec


            # return  det_matrix_vis,azimuth_map.sum(0)[6:9,:],elevation_map.sum(0)[6:9,:].T,xyzVec
            return  det_matrix_vis,azimuth_map.sum(0),np.zeros([1,1]),xyzVec

        output_a_angles.append((180 / np.pi) * np.arcsin(np.sin(a_angle) * np.cos(e_angle)))

