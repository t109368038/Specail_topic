import numpy as np
import DSP_2t4r
import matplotlib.pyplot as plt
import mmwave.dsp as dsp
import mmwave.clustering as clu
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as colors
import sys

def ReshapeRadarCube(frame, num_tx, num_chirp, num_rx, num_sample):
    rdata = np.reshape(frame, [-1, 4])
    rdata = rdata[:, 0:2:] + 1j * rdata[:, 2::]
    rdata = np.reshape(rdata, [num_chirp, num_tx, num_rx, num_sample])
    cdata1 = rdata[:, :, 0, :]
    cdata1 = np.transpose(cdata1, [0, 2, 1])
    cdata2 = rdata[:, :, 1, :]
    cdata2 = np.transpose(cdata2, [0, 2, 1])
    cdata3 = rdata[:, :, 2, :]
    cdata3 = np.transpose(cdata3, [0, 2, 1])
    cdata4 = rdata[:, :, 3, :]
    cdata4 = np.transpose(cdata4, [0, 2, 1])
    cdata = np.concatenate((cdata1, cdata2, cdata3, cdata4), axis=2)
    # print(np.shape(cdata))
    tx1 = cdata[:, :, 0::3]
    tx3 = cdata[:, :, 1::3]
    tx2 = cdata[:, :, 2::3]
    # print(np.shape(tx1))
    # print(np.shape(tx2))
    # print(np.shape(tx3))
    ret_data = np.concatenate((tx1, tx3, tx2), axis=2)
    # print(np.shape(ret_data))
    return ret_data


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
    writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)

    plt.title(title)
    print('Done')
    im_ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=3000, blit=True)
    print('Check')
    im_ani.save(save_dir, writer=writer)
    print('Complete')


plotRangeDopp = False
plot2DscatterXY = False
plot2DscatterXZ = False
plot3Dscatter = True
plotCustomPlt = False

plotMakeMovie = True
makeMovieTitle = " "
makeMovieDirectory = "E:/ResearchData/ThuMouseData/0330_2.mp4"
visTrigger = plot2DscatterXY + plot2DscatterXZ + plot3Dscatter + plotRangeDopp + plotCustomPlt

data_path = 'E:/ResearchData/ThuMouseData/'
# file_name = '0316_hand_rawdata.npy'
file_name = '1_rawdata.npy'

data = np.load(data_path + file_name)
radarcube = np.apply_along_axis(ReshapeRadarCube, 1, data, 3, 16, 4, 64)
# [frame, chirp, rx, sample]
# radarcube = radarcube.transpose([0, 1, 3, 2])

numTxAntennas = 3
max_size = 0
ims = []
count = 0
# (1.5) Required Plot Declarations
if plot2DscatterXY or plot2DscatterXZ:
    fig, axes = plt.subplots(1, 2)
elif plot3Dscatter and plotMakeMovie:
    fig = plt.figure()
    nice = fig.add_subplot(111, projection='3d')
elif plot3Dscatter:
    fig = plt.figure()
elif plotRangeDopp:
    fig = plt.figure()
elif plotCustomPlt:
    print("Using Custom Plotting")

print('Total frame:', radarcube.shape[0])
for i, frame in enumerate(radarcube):
    print('Current frame:', str(i))
    rangedoppler = DSP_2t4r.Range_Doppler(frame, 0, [128, 64])
    rangedoppler = rangedoppler.transpose([0, 2, 1])
    # cmoval = dsp.clutter_removal(rangedoppler)

    detect_matrix = np.sum(np.log2(np.abs(np.fft.fftshift(rangedoppler, axes=[0]))), axis=1)
    aoa_input = rangedoppler
    fft2d_sum = detect_matrix.astype(np.int64)
    azimuth_angle = aoa_input[:, 0:8, :]
    elevation_angle = aoa_input[:, 8::, :]
    #     * * * *
    # * * * * * * * *

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

    det_doppler_mask = (detect_matrix > thresholdDoppler)
    det_range_mask = (detect_matrix > thresholdRange)
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
    numDopplerBins = 64
    # Further peak pruning. This increases the point cloud density but helps avoid having too many detections around one object.
    detObj2DRaw = dsp.prune_to_peaks(detObj2DRaw, detect_matrix, numDopplerBins, reserve_neighbor=True)
    detObj2D = dsp.peak_grouping_along_doppler(detObj2DRaw, detect_matrix, numDopplerBins)
    SNRThresholds2 = np.array([[0, 0], [0, 0], [0, 0]])
    peakValThresholds2 = np.array([[4, 275], [1, 400], [500, 0]])
    numRangeBins = 128
    range_resolution = 0.04
    detObj2D = dsp.range_based_pruning(detObj2D, SNRThresholds2, peakValThresholds2, 15, 0.0, range_resolution)

    # azimuth_input = azimuth_angle[]
    azimuth_input = aoa_input[detObj2D['rangeIdx'], :, detObj2D['dopplerIdx']]
    x, y, z = dsp.naive_xyz(azimuth_input.T)
    xyzVecN = np.zeros((4, x.shape[0]))
    xyzVecN[0] = x * range_resolution * detObj2D['rangeIdx']
    xyzVecN[1] = y * range_resolution * detObj2D['rangeIdx']
    xyzVecN[2] = z * range_resolution * detObj2D['rangeIdx']
    xyzVecN[3] = detObj2D['dopplerIdx']

    Psi, Theta, Ranges, velocity, xyzVec = dsp.beamforming_naive_mixed_xyz(azimuth_input, detObj2D['rangeIdx'],
                                                                           detObj2D['dopplerIdx'], range_resolution,
                                                                           method='Bartlett')

    # (5) 3D-Clustering
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
    doppler_resolution = 0.04
    if len(detObj2D_f) > 0:
        count+=1
        cluster = clu.radar_dbscan(detObj2D_f, 1, doppler_resolution, use_elevation=True)

        cluster_np = np.array(cluster['size']).flatten()
        if cluster_np.size != 0:
            if max(cluster_np) > max_size:
                max_size = max(cluster_np)



        # (6) Visualization
        if plotRangeDopp:
            continue
        if plot2DscatterXY or plot2DscatterXZ:

            if plot2DscatterXY:
                xyzVec = xyzVec[:, (np.abs(xyzVec[2]) < 1.5)]
                xyzVecN = xyzVecN[:, (np.abs(xyzVecN[2]) < 1.5)]
                axes[0].set_ylim(bottom=0, top=10)
                axes[0].set_ylabel('Range')
                axes[0].set_xlim(left=-4, right=4)
                axes[0].set_xlabel('Azimuth')
                axes[0].grid(b=True)

                axes[1].set_ylim(bottom=0, top=10)
                axes[1].set_xlim(left=-4, right=4)
                axes[1].set_xlabel('Azimuth')
                axes[1].grid(b=True)

            elif plot2DscatterXZ:
                axes[0].set_ylim(bottom=-5, top=5)
                axes[0].set_ylabel('Elevation')
                axes[0].set_xlim(left=-4, right=4)
                axes[0].set_xlabel('Azimuth')
                axes[0].grid(b=True)

                axes[1].set_ylim(bottom=-5, top=5)
                axes[1].set_xlim(left=-4, right=4)
                axes[1].set_xlabel('Azimuth')
                axes[1].grid(b=True)

            if plotMakeMovie and plot2DscatterXY:
                ims.append((axes[0].scatter(xyzVec[0], xyzVec[1], c='r', marker='o', s=2),
                            axes[1].scatter(xyzVecN[0], xyzVecN[1], c='b', marker='o', s=2)))
            elif plotMakeMovie and plot2DscatterXZ:
                ims.append((axes[0].scatter(xyzVec[0], xyzVec[2], c='r', marker='o', s=2),
                            axes[1].scatter(xyzVecN[0], xyzVecN[2], c='b', marker='o', s=2)))
            elif plot2DscatterXY:
                axes[0].scatter(xyzVec[0], xyzVec[1], c='r', marker='o', s=3)
                axes[1].scatter(xyzVecN[0], xyzVecN[1], c='b', marker='o', s=3)
                plt.pause(0.1)
                axes[0].clear()
                axes[1].clear()
            elif plot2DscatterXZ:
                axes[0].scatter(xyzVec[0], xyzVec[2], c='r', marker='o', s=3)
                axes[1].scatter(xyzVecN[0], xyzVecN[2], c='b', marker='o', s=3)
                plt.pause(0.1)
                axes[0].clear()
                axes[1].clear()
        elif plot3Dscatter and plotMakeMovie:
            nice.set_zlim3d(bottom=-5, top=5)
            nice.set_ylim(bottom=0, top=10)
            nice.set_xlim(left=-4, right=4)
            nice.set_xlabel('X Label')
            nice.set_ylabel('Y Label')
            nice.set_zlabel('Z Label')
            # for i, v in enumerate(velocity):
            #     if v > 50:
            #         nice.scatter(xyzVec[0, i], xyzVec[1, i], xyzVec[2, i], c='r', marker='o', s=3)
            #     elif v > 40:
            #         nice.scatter(xyzVec[0, i], xyzVec[1, i], xyzVec[2, i], c='y', marker='o', s=3)
            #     elif v > 30:
            #         nice.scatter(xyzVec[0, i], xyzVec[1, i], xyzVec[2, i], c='m', marker='o', s=2)
            #     elif v > 20:
            #         nice.scatter(xyzVec[0, i], xyzVec[1, i], xyzVec[2, i], c='c', marker='o', s=2)
            #     else:
            #         nice.scatter(xyzVec[0, i], xyzVec[1, i], xyzVec[2, i], c='k', marker='o', s=1)
            # ims.append(nice)

            ims.append((nice.scatter(xyzVec[0], xyzVec[1], xyzVec[2], c=xyzVec[3], cmap='Reds', vmin=0, vmax=63, marker='o', s=2),))

        # elif plot3Dscatter:
        #     if singFrameView:
        #         ellipse_visualize(fig, cluster, detObj2D_f[:, 3:6])
        #     else:
        #         ellipse_visualize(fig, cluster, detObj2D_f[:, 3:6
        #         plt.pause(0.1)
        #         plt.clf()
        else:
            sys.exit("Unknown plot options.")

print(count)
plt.rcParams['animation.ffmpeg_path'] = 'C:/Users/lab210/Downloads/ffmpeg-2021-03-14-git-1d61a31497-full_build/bin/ffmpeg.exe'
if visTrigger and plotMakeMovie:
    movieMaker(fig, ims, makeMovieTitle, makeMovieDirectory)
#     if plotRangeDoppler:
#         # plt.subplot(1, 2, 1)
#         plt.imshow(detect_matrix[:, 0, :].T)
#         plt.title("Range-Doppler:" + str(i))
#
#         # matrix = np.fft.fftshift(cmoval, axes=[0])
#         # plt.subplot(1, 2, 2)
#         # plt.imshow(np.abs(matrix)[:, 0, :].T)
#         # plt.title("Range-Doppler Clutter Removal:" + str(i))
#         plt.pause(0.5)
#         plt.clf()
#
#
# plt.close()