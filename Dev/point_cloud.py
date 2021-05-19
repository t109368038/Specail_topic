import numpy as np
import matplotlib.pyplot as plt
from DSP_2t4r import Range_Doppler
from scipy.ndimage import convolve1d


def ca_(x, guard_len=4, noise_len=8, mode='wrap', l_bound=4000):
    """Uses Cell-Averaging CFAR (CA-CFAR) to calculate a threshold that can be used to calculate peaks in a signal.
    Args:
        x (~numpy.ndarray): Signal.
        guard_len (int): Number of samples adjacent to the CUT that are ignored.
        noise_len (int): Number of samples adjacent to the guard padding that are factored into the calculation.
        mode (str): Specify how to deal with edge cells. Examples include 'wrap' and 'constant'.
        l_bound (float or int): Additive lower bound while calculating peak threshold.
    Returns:
        Tuple [ndarray, ndarray]
            1. (ndarray): Upper bound of noise threshold.
            #. (ndarray): Raw noise strength.
    Examples:
        >>> signal = np.random.randint(100, size=10)
        >>> signal
            array([41, 76, 95, 28, 25, 53, 10, 93, 54, 85])
        >>> threshold = mm.dsp.ca_(signal, l_bound=20, guard_len=1, noise_len=3)
        >>> threshold
            (array([70, 76, 64, 79, 81, 91, 74, 71, 70, 79]), array([50, 56, 44, 59, 61, 71, 54, 51, 50, 59]))
        Perform a non-wrapping CFAR thresholding
        >>> signal = np.random.randint(100, size=10)
        >>> signal
            array([41, 76, 95, 28, 25, 53, 10, 93, 54, 85])
        >>> threshold = mm.dsp.ca_(signal, l_bound=20, guard_len=1, noise_len=3, mode='constant')
        >>> threshold
            (array([44, 37, 41, 65, 81, 91, 67, 51, 34, 46]), array([24, 17, 21, 45, 61, 71, 47, 31, 14, 26]))
    """
    if isinstance(x, list):
        x = np.array(x)
    assert type(x) == np.ndarray

    kernel = np.ones(1 + (2 * guard_len) + (2 * noise_len), dtype=x.dtype) / (2 * noise_len)
    kernel[noise_len:noise_len + (2 * guard_len) + 1] = 0

    noise_floor = convolve1d(x, kernel, mode=mode)
    threshold = noise_floor + l_bound

    return threshold, noise_floor


def prune_to_peaks(det_obj2_d_raw, det_matrix, num_doppler_bins, reserve_neighbor=False):
    """Reduce the CFAR detected output to local peaks.
    Reduce the detected output to local peaks. If reserveNeighbor is toggled, will also return the larger neighbor. For
    example, given an array [2, 1, 5, 3, 2], default method will return [2, 5] while reserve neighbor will return
    [2, 5, 3]. The neighbor has to be a larger neighbor of the two immediate ones and also be part of the peak. the 1st
    element "1" in the example is not returned because it's smaller than both sides so that it is not part of the peak.
    Args:
        det_obj2_d_raw (np.ndarray): The detected objects structured array which contains the range_idx, doppler_idx,
         peakVal and SNR, etc.
        det_matrix (np.ndarray): Output of doppler FFT with virtual antenna dimensions reduced. It has the shape of
            (num_range_bins, num_doppler_bins).
        num_doppler_bins (int): Number of doppler bins.
        reserve_neighbor (boolean): if toggled, will return both peaks and the larger neighbors.
    Returns:
        cfar_det_obj_index_pruned (np.ndarray): Pruned version of cfar_det_obj_index.
        cfar_det_obj_SNR_pruned (np.ndarray): Pruned version of cfar_det_obj_SNR.
    """

    range_idx = det_obj2_d_raw['rangeIdx']
    doppler_idx = det_obj2_d_raw['dopplerIdx']
    next_idx = doppler_idx + 1
    next_idx[doppler_idx == num_doppler_bins - 1] = 0
    prev_idx = doppler_idx - 1
    prev_idx[doppler_idx == 0] = num_doppler_bins - 1

    prev_val = det_matrix[range_idx, prev_idx]
    current_val = det_matrix[range_idx, doppler_idx]
    next_val = det_matrix[range_idx, next_idx]

    if reserve_neighbor:
        next_next_idx = next_idx + 1
        next_next_idx[next_idx == num_doppler_bins - 1] = 0
        prev_prev_idx = prev_idx - 1
        prev_prev_idx[prev_idx == 0] = num_doppler_bins - 1

        prev_prev_val = det_matrix[range_idx, prev_prev_idx]
        next_next_val = det_matrix[range_idx, next_next_idx]
        is_neighbor_of_peak_next = (current_val > next_next_val) & (current_val > prev_val)
        is_neighbor_of_peak_prev = (current_val > prev_prev_val) & (current_val > next_val)

        pruned_idx = (current_val > prev_val) & (current_val > next_val) | is_neighbor_of_peak_next | is_neighbor_of_peak_prev
    else:
        pruned_idx = (current_val > prev_val) & (current_val > next_val)

    det_obj2_d_pruned = det_obj2_d_raw[pruned_idx]

    return det_obj2_d_pruned


def peak_grouping_along_doppler(det_obj_2d,
                                det_matrix,
                                num_doppler_bins):
    """Perform peak grouping along the doppler direction only.
    This is a temporary remedy for the slow and old implementation of peak_grouping_qualified() function residing in
    dsp.py currently. Will merge this back to there to enable more generic peak grouping.
    """
    num_det_objs = det_obj_2d.shape[0]
    range_idx = det_obj_2d['rangeIdx']
    doppler_idx = det_obj_2d['dopplerIdx']
    kernel = np.zeros((num_det_objs, 3), dtype=np.float32)
    kernel[:, 0] = det_matrix[range_idx, doppler_idx - 1]
    kernel[:, 1] = det_obj_2d['peakVal'].astype(np.float32)
    kernel[:, 2] = det_matrix[range_idx, (doppler_idx + 1) % num_doppler_bins]
    detectedFlag = (kernel[:, 1] > kernel[:, 0]) & (kernel[:, 1] > kernel[:, 2])
    return det_obj_2d[detectedFlag]


def range_based_pruning(det_obj_2d_raw,
                        snr_thresh,
                        peak_val_thresh,
                        max_range,
                        min_range,
                        range_resolution):
    """Filter out the objects out of the range and not sufficing SNR/peakVal requirement.
    Filter out the objects based on the two following conditions:
    1. Not within [min_range and max_range].
    2. Does not satisfy SNR/peakVal requirement, where it requires higher standard when closer and lower when further.
    """
    det_obj_2d = det_obj_2d_raw[(det_obj_2d_raw['rangeIdx'] >= min_range) & \
                                (det_obj_2d_raw['rangeIdx'] <= max_range)]
    snr_idx1 = (det_obj_2d['SNR'] > snr_thresh[0, 1]) & (det_obj_2d['rangeIdx'] * range_resolution < snr_thresh[0, 0])
    snr_idx2 = (det_obj_2d['SNR'] > snr_thresh[1, 1]) & \
              (det_obj_2d['rangeIdx'] * range_resolution < snr_thresh[1, 0]) & \
              (det_obj_2d['rangeIdx'] * range_resolution >= snr_thresh[0, 0])
    snr_idx3 = (det_obj_2d['SNR'] > snr_thresh[2, 1]) & (det_obj_2d['rangeIdx'] * range_resolution > snr_thresh[1, 0])
    snr_idx = snr_idx1 | snr_idx2 | snr_idx3

    peak_val_idx = np.logical_not((det_obj_2d['peakVal'] < peak_val_thresh[0, 1]) & \
                                (det_obj_2d['rangeIdx'] * range_resolution < peak_val_thresh[0, 0]))
    combined_idx = snr_idx & peak_val_idx
    det_obj_2d = det_obj_2d[combined_idx]

    return det_obj_2d


def naive_xyz(virtual_ant, num_tx=2, num_rx=4, fft_size=64):
    """ Estimate the phase introduced from the elevation of the elevation antennas
    Args:
        virtual_ant: Signal received by the rx antennas, shape = [#angleBins, #detectedObjs], zero-pad #virtualAnts to #angleBins
        num_tx: Number of transmitter antennas used
        num_rx: Number of receiver antennas used
        fft_size: Size of the fft performed on the signals
    Returns:
        x_vector (float): Estimated x axis coordinate in meters (m)
        y_vector (float): Estimated y axis coordinate in meters (m)
        z_vector (float): Estimated z axis coordinate in meters (m)
    """
    assert num_tx > 2, "need a config for more than 2 TXs"
    num_detected_obj = virtual_ant.shape[1]

    # Zero pad azimuth
    azimuth_ant = virtual_ant[:2 * num_rx, :]
    azimuth_ant_padded = np.zeros(shape=(fft_size, num_detected_obj), dtype=np.complex_)
    azimuth_ant_padded[:2 * num_rx, :] = azimuth_ant

    # Process azimuth information
    azimuth_fft = np.fft.fft(azimuth_ant_padded, axis=0)
    k_max = np.argmax(np.abs(azimuth_fft), axis=0)  # shape = (num_detected_obj, )
    # peak_1 = azimuth_fft[k_max]
    peak_1 = np.zeros_like(k_max, dtype=np.complex_)
    for i in range(len(k_max)):
        peak_1[i] = azimuth_fft[k_max[i], i]

    k_max[k_max > (fft_size // 2) - 1] = k_max[k_max > (fft_size // 2) - 1] - fft_size
    wx = 2 * np.pi / fft_size * k_max  # shape = (num_detected_obj, )
    x_vector = wx / np.pi

    # Zero pad elevation
    elevation_ant = virtual_ant[2 * num_rx:, :]
    elevation_ant_padded = np.zeros(shape=(fft_size, num_detected_obj), dtype=np.complex_)
    # elevation_ant_padded[:len(elevation_ant)] = elevation_ant
    elevation_ant_padded[:num_rx, :] = elevation_ant

    # Process elevation information
    elevation_fft = np.fft.fft(elevation_ant, axis=0)
    elevation_max = np.argmax(np.log2(np.abs(elevation_fft)), axis=0)  # shape = (num_detected_obj, )
    peak_2 = np.zeros_like(elevation_max, dtype=np.complex_)
    # peak_2 = elevation_fft[np.argmax(np.log2(np.abs(elevation_fft)))]
    for i in range(len(elevation_max)):
        peak_2[i] = elevation_fft[elevation_max[i], i]

    # Calculate elevation phase shift
    wz = np.angle(peak_1 * peak_2.conj() * np.exp(1j * 2 * wx))
    z_vector = wz / np.pi
    y_vector = np.sqrt(1 - x_vector ** 2 - z_vector ** 2)
    return x_vector, y_vector, z_vector
data = 'E:/ResearchData/ThuMouseData/TEST/'
# name = '3t4r'
name = 'thumb_0302'
radar_data = np.load(data + name + '_rawdata.npy')

# Radar Parameters
chirp_num = 32
tx_num = 2
adc_sample = 64

radar_frame1 = radar_data[50]
# Reshape Radar Data Cube
radar_frame1 = np.reshape(radar_frame1, [-1, 4])
radar_frame1 = radar_frame1[:, 0:2:] + 1j * radar_frame1[:, 2::]
radar_frame1 = np.reshape(radar_frame1, [chirp_num * tx_num, -1, adc_sample])
radar_frame1 = radar_frame1.transpose([0, 2, 1])
ch1_data = radar_frame1[0: 64: 2, :, :]
ch3_data = radar_frame1[1: 64: 2, :, :]
# radar_frame1 = np.array([ch1_data, ch3_data]) # radarCube[Ntx][Ndc][Nrx][Nr]
radar_frame1 = np.concatenate([ch1_data, ch3_data], axis=2)
radarcube_raw, radarcube = Range_Doppler(radar_frame1, 2, padding_size=[128, 64])

radarcube_raw_1 = radarcube_raw[:, :, 0].T
aoa_input = radarcube_raw[:, :, 0].T
radarcube_raw_1 = np.abs(radarcube_raw_1)



# Object Detection
# CFAR, SNR
fft2d_sum = radarcube_raw_1.astype(np.int64)
thresholdDoppler, noiseFloorDoppler = np.apply_along_axis(func1d=ca_,
                                                          axis=0,
                                                          arr=fft2d_sum.T,
                                                          l_bound=1.5,
                                                          guard_len=4,
                                                          noise_len=16)

thresholdRange, noiseFloorRange = np.apply_along_axis(func1d=ca_,
                                                      axis=0,
                                                      arr=fft2d_sum,
                                                      l_bound=2.5,
                                                      guard_len=4,
                                                      noise_len=16)

thresholdDoppler, noiseFloorDoppler = thresholdDoppler.T, noiseFloorDoppler.T
det_doppler_mask = (radarcube_raw_1 > thresholdDoppler)
det_range_mask = (radarcube_raw_1 > thresholdRange)
# Get indices of detected peaks
full_mask = (det_doppler_mask & det_range_mask)
det_peaks_indices = np.argwhere(full_mask == True)

# peakVals and SNR calculation
peakVals = fft2d_sum[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
snr = peakVals - noiseFloorRange[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]

dtype_location = '(' + str(tx_num) + ',)<f4'
dtype_detObj2D = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
                           'formats': ['<i4', '<i4', '<f4', dtype_location, '<f4']})
detObj2DRaw = np.zeros((det_peaks_indices.shape[0],), dtype=dtype_detObj2D)
detObj2DRaw['rangeIdx'] = det_peaks_indices[:, 0].squeeze()
detObj2DRaw['dopplerIdx'] = det_peaks_indices[:, 1].squeeze()
detObj2DRaw['peakVal'] = peakVals.flatten()
detObj2DRaw['SNR'] = snr.flatten()

# Further peak pruning. This increases the point cloud density but helps avoid having too many detections around one
# object.
detObj2DRaw = prune_to_peaks(detObj2DRaw, radarcube_raw_1, 128, reserve_neighbor=True)

# Peak grouping
detObj2D = peak_grouping_along_doppler(detObj2DRaw, radarcube_raw_1, 128)
SNRThresholds2 = np.array([[2, 23], [10, 11.5], [35, 16.0]])
peakValThresholds2 = np.array([[4, 275], [1, 400], [500, 0]])
detObj2D = range_based_pruning(detObj2D, SNRThresholds2, peakValThresholds2, 64, 0.5, 3.75)



# azimuthInput = aoa_input[detObj2D['rangeIdx'], :, detObj2D['dopplerIdx']]
#
# x, y, z = naive_xyz(azimuthInput.T)
# xyzVecN = np.zeros((3, x.shape[0]))
# xyzVecN[0] = x * 3.75 * detObj2D['rangeIdx']
# xyzVecN[1] = y * 3.75 * detObj2D['rangeIdx']
# xyzVecN[2] = z * 3.75 * detObj2D['rangeIdx']


cfar_out = np.multiply(full_mask, radarcube_raw_1)

plt.figure(1)
plt.imshow(radarcube_raw_1)

plt.figure(2)
plt.imshow(cfar_out)
