import sys

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mmwave.clustering as clu
import mmwave.dsp as dsp
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import DSP_2t4r
import os

def plot_rdi(rdi):
    plt.title('Heatmap of 2D normally distributed data points')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    #
    # plt.hist2d(x, y, bins=N_bins, normed=False, cmap='plasma')


def plot_pd(pos):
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim3d(bottom=-5, top=5)
    ax.set_ylim(bottom=0, top=10)
    ax.set_xlim(left=-4, right=4)
    ax.view_init(elev=-174, azim=-90)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])
    ax.scatter(0,0,0,c='red')
    # plt.show()
    return ax
def plot_hand_cm(ax,elev,azim,x1,x2,y2,px,py,pz,mode,file_mode):

    # ax = fig.gca(projection='3d')
    x_len = 640 * 0.0063
    y_len = 480 * 0.0062
    z_len = 640 * 0.0096
    # ax.set_xlim(0-x_len/2, x_len-x_len/2)  # X軸，橫向向右方向
    # ax.set_ylim(0-y_len/2, y_len-y_len/2)  # Y軸,左向與X,Z軸互為垂直
    # ax.set_zlim(0-z_len/2,z_len-z_len/2 )  # 豎向為Z軸
    if mode == "avg":
        mark = "."
        ax.plot3D(x1[1:5], x2[1:5], y2[1:5], marker=mark, color="gray")
        ax.plot3D(x1[5:9], x2[5:9], y2[5:9], marker=mark, color="gray")
        ax.plot3D(x1[9:13], x2[9:13], y2[9:13], marker=mark, color="gray")
        ax.plot3D(x1[13:17], x2[13:17], y2[13:17], marker=mark, color="gray")
        ax.plot3D(x1[17:], x2[17:], y2[17:], marker=mark, color="gray")
        # ax.scatter(0, -y_len/2, 0, marker="o")
        ax.scatter(0, 0, 0, marker="o")
        # ax.view_init(elev=elev, azim=azim)
        # ax.view_init(elev=-174, azim=-130)
        ax.view_init(elev=-174, azim=-90)

        ax.set_xlabel("x axis(1unit/10cm)")
        ax.set_ylabel("y axis(1unit/10cm)")
        ax.set_zlabel("z axis(1unit/10cm)")
        # palm = []
        palmx = [x1[0], x1[5], x1[9], x1[13], x1[17], x1[0], x1[1]]
        palmy = [x2[0], x2[5], x2[9], x2[13], x2[17], x2[0], x2[1]]
        palmz = [y2[0], y2[5], y2[9], y2[13], y2[17], y2[0], y2[1]]
        ax.plot3D(palmx, palmy, palmz, marker=mark, color="gray")
        ax.plot3D(px, py, pz, color="Red")
        # ax.plot3D(x1[8], x2[8], y2[8], marker=mark, color="Red")
        if file_mode==0:
            ax.scatter(x1[4], x2[4], y2[4], marker=mark, color="Red")
            plt.draw()
            return x1[4],x2[4],y2[4]
        else:
            ax.scatter(x1[8], x2[8], y2[8], marker=mark, color="Red")
            plt.draw()
            return x1[8],x2[8],y2[8]

if __name__ == '__main__':

    PDdata = np.load("D:/kaiku_report/20210414/pd.npy", allow_pickle=True)
    data_path = 'D:/kaiku_report/20210414/'
    cam_hp = np.load(data_path + "cam_hp.npy", allow_pickle=True)
    cam1_hp = np.load(data_path + "cam_hp1.npy", allow_pickle=True)
    cam_x = cam_hp[::2]
    cam_y = cam_hp[1::2]
    cam1_x = cam1_hp[::2]
    cam1_y = cam1_hp[1::2]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # plot_pd(PDdata[0])
    # plt.show()
    x = []
    y = []
    z = []
    plt.ion()
    for i in range(len(PDdata)):
        print(i)
        ax = plot_pd(PDdata[i])
        x1 = cam_x[i]
        y1 = cam_y[i]
        x2 = cam1_x[i]
        y2 = cam1_x[i]
        ax.set_zlim3d(bottom=-2.5, top=2.5)
        ax.set_ylim(bottom=0, top=3)
        ax.set_xlim(left=-2, right=2)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        if (x1 != None).all() == True:
            x1 = x1.astype(np.double)
            x1 = np.round((x1 * 0.0063), 3)
            x1 = x1 - ((640 * 0.0063 / 2)) - 0.3
            print(x1)
        if (y1 != None).all() == True:
            y1 = y1.astype(np.double)
            y1 = np.round((y1 * 0.0062), 3)
            # y1 -= ((480 * 0.0062) / 2)
        if (x2 != None).all() == True:
            x2 = x2.astype(np.double)
            x2 = (x2 * 0.0096)
            x2 -= ((640 * 0.0096) / 2)
            x2 = np.round(x2, 3)
        # tx, ty, tz = plot_hand_cm(ax, -147, -65, x1, y1, x2, x, y, z, "avg",file_mode)
        tx, ty, tz = plot_hand_cm(ax, -175, -175, x1, y1, x2, x, y, z, "avg", 2)
        if len(x) > 20:
            x = np.delete(x, 0)
            y = np.delete(y, 0)
            z = np.delete(z, 0)
        x = np.append(x, tx)
        y = np.append(y, ty)
        z = np.append(z, tz)
        plt.draw()
        plt.pause(0.0001)
        plt.clf()