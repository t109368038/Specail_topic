import sys
import numpy as np
import mmwave.dsp as dsp
import mmwave.clustering as clu
from mmwave.dataloader import DCA1000
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


def Range_Angle(data, padding_size):
    rai_abs = np.fft.fft(data, n=padding_size, axis=2)
    rai_abs = np.fft.fftshift(np.abs(rai_abs), axes=2)
    rai_abs = np.flip(rai_abs, axis=1)
    return rai_abs


def ellipse_visualize(fig, clusters, points):
    """Visualize point clouds and outputs from 3D-DBSCAN

    Args:
        Clusters (np.ndarray): Numpy array containing the clusters' information including number of points, center and size of
                the clusters in x,y,z coordinates and average velocity. It is formulated as the structured array for numpy.
        points (dict): A dictionary that stores x,y,z's coordinates in np arrays

    Returns:
        N/A
    """
    # fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim3d(bottom=-5, top=5)
    ax.set_ylim(bottom=0, top=10)
    ax.set_xlim(left=-4, right=4)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # ax.set_aspect('equal')

    # scatter plot
    # ax.scatter(points['x'], points['y'], points['z'])
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2])

    # number of ellipsoids
    ellipNumber = len(clusters)

    norm = colors.Normalize(vmin=0, vmax=ellipNumber)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    for indx in range(ellipNumber):
        center = [clusters['center'][indx][0], clusters['center'][indx][1], clusters['center'][indx][2]]

        radii = np.zeros([3, ])
        radii[0] = clusters['size'][indx][0]
        radii[1] = clusters['size'][indx][1]
        radii[2] = clusters['size'][indx][2]

        u = np.linspace(0.0, 2.0 * np.pi, 60)
        v = np.linspace(0.0, np.pi, 60)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]],
                                                     np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])) + center

        ax.plot_surface(x, y, z, rstride=3, cstride=3, color=m.to_rgba(indx), linewidth=0.1, alpha=1, shade=True)

    plt.show()


def movieMaker(fig, ims, title, save_dir):
    import matplotlib.animation as animation

    # Set up formatting for the Range Azimuth heatmap movies
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)

    plt.title(title)
    print('Done')
    im_ani = animation.ArtistAnimation(fig, ims, interval=500, repeat_delay=3000, blit=True)
    print('Check')
    im_ani.save(save_dir, writer=writer)
    print('Complete')


if __name__ == '__main__':
    print("==============================Program Start==============================")
    ims = []
    max_size = 0
    # (1) Loading data
    data_path = 'E:/ResearchData/ThuMouseData/RESULT0413/'
    # file_name = '0413_move_azimuth_rawdata.npy'
    file_name = '0413_move_elevation_rawdata.npy'
    adc_data = np.load(data_path + file_name)
    numFrames = adc_data.shape[0]
    numADCSamples = 64
    numTxAntennas = 3
    numRxAntennas = 4
    numLoopsPerFrame = 16
    numChirpsPerFrame = numTxAntennas * numLoopsPerFrame
    numRangeBins = numADCSamples
    numDopplerBins = numLoopsPerFrame
    numAngleBins = 64
    # range resolution, bandwidth
    range_resolution, bandwidth = dsp.range_resolution(numADCSamples, 2000, 121.134)
    doppler_resolution = dsp.doppler_resolution(bandwidth, 77, 33.02, 9.43, 16, 3)
    print("Range Resolution:" + str(range_resolution) + " m", "\nBandwidth:" + str(bandwidth) + " Hz")
    adc_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=numChirpsPerFrame,
                                   num_rx=numRxAntennas, num_samples=numADCSamples)
    print("Data Loaded")

    fig = plt.figure()
    nice = Axes3D(fig)
    for n, frame in enumerate(adc_data):

        # (2) Range Processing
        from mmwave.dsp.utils import Window

        radar_cube = dsp.range_processing(frame, window_type_1d=Window.HANNING)
        assert radar_cube.shape == (
            numChirpsPerFrame, numRxAntennas, numADCSamples), "[ERROR] Radar cube is not the correct shape!"

        # (3) Doppler Processing
        det_matrix, aoa_input = dsp.doppler_processing(radar_cube, num_tx_antennas=3, clutter_removal_enabled=True,
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

        azimuth_map = Range_Angle(azimuth_ant_1, 90)
        elevation_map = Range_Angle(elevation_combine, 90)

        # ims.append((plt.imshow(np.rot90(azimuth_map.sum(0), -2)), ))
        # ims.append((plt.imshow(np.rot90(elevation_map.sum(0), -2)), ))
        # plt.imshow(azimuth_map[0, :, :])

        # plt.show()
        # time.sleep(2)
        # plt.clf()

        # (5) Object Detection
        fft2d_sum = det_matrix.astype(np.int64)
        thresholdDoppler, noiseFloorDoppler = np.apply_along_axis(func1d=dsp.ca_,
                                                                  axis=0,
                                                                  arr=fft2d_sum.T,
                                                                  l_bound=1.5,
                                                                  guard_len=4,
                                                                  noise_len=16)

        thresholdRange, noiseFloorRange = np.apply_along_axis(func1d=dsp.ca_,
                                                              axis=0,
                                                              arr=fft2d_sum,
                                                              l_bound=2.5,
                                                              guard_len=4,
                                                              noise_len=16)

        thresholdDoppler, noiseFloorDoppler = thresholdDoppler.T, noiseFloorDoppler.T

        det_doppler_mask = (det_matrix > thresholdDoppler)
        det_range_mask = (det_matrix > thresholdRange)

        # Get indices of detected peaks
        full_mask = (det_doppler_mask & det_range_mask)
        det_peaks_indices = np.argwhere(full_mask == True)

        # peakVals and SNR calculation
        peakVals = fft2d_sum[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
        snr = peakVals - noiseFloorRange[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]

        dtype_location = '(' + str(numTxAntennas) + ',)<f4'
        dtype_detObj2D = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
                                   'formats': ['<i4', '<i4', '<f4', dtype_location, '<f4']})
        detObj2DRaw = np.zeros((det_peaks_indices.shape[0],), dtype=dtype_detObj2D)
        detObj2DRaw['rangeIdx'] = det_peaks_indices[:, 0].squeeze()
        detObj2DRaw['dopplerIdx'] = det_peaks_indices[:, 1].squeeze()
        detObj2DRaw['peakVal'] = peakVals.flatten()
        detObj2DRaw['SNR'] = snr.flatten()

        detObj2DRaw = dsp.prune_to_peaks(detObj2DRaw, det_matrix, numDopplerBins, reserve_neighbor=True)

        # --- Peak Grouping
        detObj2D = dsp.peak_grouping_along_doppler(detObj2DRaw, det_matrix, numDopplerBins)

        SNRThresholds2 = np.array([[2, 23], [10, 11.5], [35, 16.0]])
        peakValThresholds2 = np.array([[4, 275], [1, 400], [500, 0]])

        detObj2D = dsp.range_based_pruning(detObj2D, SNRThresholds2, peakValThresholds2, numRangeBins, 0.5,
                                           range_resolution)

        azimuthInput = aoa_input[detObj2D['rangeIdx'], :, detObj2D['dopplerIdx']]
        Psi, Theta, Ranges, velocity, xyzVec = dsp.beamforming_naive_mixed_xyz(azimuthInput, detObj2D['rangeIdx'],
                                                                               detObj2D['dopplerIdx'], range_resolution, method='Bartlett')

        # (6) 3D-Cluster
        # detObj2D must be fully populated and completely accurate right here
        numDetObjs = detObj2D.shape[0]
        dtf = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
                        'formats': ['<f4', '<f4', '<f4', dtype_location, '<f4']})
        detObj2D_f = detObj2D.astype(dtf)
        detObj2D_f = detObj2D_f.view(np.float32).reshape(-1, 7)

        # Fully populate detObj2D_f with correct info
        for i, currRange in enumerate(Ranges):
            if i >= (detObj2D_f.shape[0]):
                # copy last row
                detObj2D_f = np.insert(detObj2D_f, i, detObj2D_f[i - 1], axis=0)
            if currRange == detObj2D_f[i][0]:
                detObj2D_f[i][3] = xyzVec[0][i]
                detObj2D_f[i][4] = xyzVec[1][i]
                detObj2D_f[i][5] = xyzVec[2][i]
            else:  # Copy then populate
                detObj2D_f = np.insert(detObj2D_f, i, detObj2D_f[i - 1], axis=0)
                detObj2D_f[i][3] = xyzVec[0][i]
                detObj2D_f[i][4] = xyzVec[1][i]
                detObj2D_f[i][5] = xyzVec[2][i]

                # radar_dbscan(epsilon, vfactor, weight, numPoints)
        #        cluster = radar_dbscan(detObj2D_f, 1.7, 3.0, 1.69 * 1.7, 3, useElevation=True)
        if len(detObj2D_f) > 0:
            cluster = clu.radar_dbscan(detObj2D_f, 0, doppler_resolution, use_elevation=True)

            cluster_np = np.array(cluster['size']).flatten()
            if cluster_np.size != 0:
                if max(cluster_np) > max_size:
                    max_size = max(cluster_np)



        # (7) visualizer
        nice.set_zlim3d(bottom=-2, top=2)
        nice.set_ylim(bottom=0, top=2)
        nice.set_xlim(left=-2, right=2)
        nice.set_xlabel('X Label')
        nice.set_ylabel('Y Label')
        nice.set_zlabel('Z Label')

        ims.append((nice.scatter(xyzVec[0], xyzVec[1], xyzVec[2], c='r', marker='o', s=2),))
        if n % 10 == 0:
            print('Current Frame:', n)

        # plt.pause(0.05)
        # plt.clf()
        # if i > 50:
        #     break


    plt.rcParams['animation.ffmpeg_path'] = \
        'C:/Users/lab210/Downloads/ffmpeg-2021-03-14-git-1d61a31497-full_build/bin/ffmpeg.exe'
    makeMovieTitle = ''
    # makeMovieDirectory = data_path + '/azimuth.mp4'
    # makeMovieDirectory = data_path + '/elevation.mp4'
    makeMovieDirectory = data_path + '/PointCloud.mp4'
    movieMaker(fig, ims, makeMovieTitle, makeMovieDirectory)

    # plt.imshow(np.rot90(det_matrix_vis / det_matrix_vis.max(), -2))
    # plt.title("Range-Doppler plot " + str(i))
    # plt.plot(np.abs(radar_cube[0, 0, :]))
    # plt.show()
