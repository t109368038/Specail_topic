import sys
import platform
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent)
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient)
from PyQt5.QtWidgets import *

from app_modules import Ui_MainWindow,Style
from ui_functions import *

# -------------- real-time-radar -----------------
from real_time_process_3t4r import UdpListener, DataProcessor
#from CameraCapture import CamCapture
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from radar_config import SerialConfig
from queue import Queue
from tkinter import filedialog
import tkinter as tk
import queue
import serial
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication
import pyqtgraph.opengl as gl
import numpy as np
import threading
import time
import sys
import socket
import cv2

# -----------------------------------------------
# from app_layout_2t4r_test import Ui_MainWindow_radar
from app_layout_2t4r import Ui_MainWindow_radar
from R3t4r_to_point_cloud_for_realtime import plot_pd
# from caclulate_mediapipe import deal_imag
# -----------------------------------------------
config = '../radar_config/IWR1843_cfg_3t4r_v3.4_1.cfg'
#config = '../radar_config/xwr68xx_profile_2021_03_23T08_12_36_405.cfg'
# config = '../radar_config/xwr18xx_profile_2021_03_09T10_45_11_974.cfg'
# config = '../radar_config/IWR1843_3d.cfg'
# config = '../radar_config/xwr18xx_profile_2021_03_05T07_10_37_413.cfg'

set_radar = SerialConfig(name='ConnectRadar', CLIPort='/dev/tty.usbmodemR21010501', BaudRate=115200)
# set_radar = SerialConfig(name='ConnectRadar', CLIPort='COM3', BaudRate=115200)
# set_radar = SerialConfig(name='ConnectRadar', CLIPort='COM21', BaudRate=115200)

class Realtime_sys():
    def __init__(self,mainwindow):
        super().__init__()
        self.pd_save_status = 0
        self.pd_save = []
        self.rdi = []
        self.rai = []
        self.data = []
        self.ui = mainwindow
    def send_cmd(self,code):
            # command code list
            CODE_1 = (0x01).to_bytes(2, byteorder='little', signed=False)
            CODE_2 = (0x02).to_bytes(2, byteorder='little', signed=False)
            CODE_3 = (0x03).to_bytes(2, byteorder='little', signed=False)
            CODE_4 = (0x04).to_bytes(2, byteorder='little', signed=False)
            CODE_5 = (0x05).to_bytes(2, byteorder='little', signed=False)
            CODE_6 = (0x06).to_bytes(2, byteorder='little', signed=False)
            CODE_7 = (0x07).to_bytes(2, byteorder='little', signed=False)
            CODE_8 = (0x08).to_bytes(2, byteorder='little', signed=False)
            CODE_9 = (0x09).to_bytes(2, byteorder='little', signed=False)
            CODE_A = (0x0A).to_bytes(2, byteorder='little', signed=False)
            CODE_B = (0x0B).to_bytes(2, byteorder='little', signed=False)
            CODE_C = (0x0C).to_bytes(2, byteorder='little', signed=False)
            CODE_D = (0x0D).to_bytes(2, byteorder='little', signed=False)
            CODE_E = (0x0E).to_bytes(2, byteorder='little', signed=False)

            # packet header & footer
            header = (0xA55A).to_bytes(2, byteorder='little', signed=False)
            footer = (0xEEAA).to_bytes(2, byteorder='little', signed=False)

            # data size
            dataSize_0 = (0x00).to_bytes(2, byteorder='little', signed=False)
            dataSize_6 = (0x06).to_bytes(2, byteorder='little', signed=False)

            # data
            data_FPGA_config = (0x01020102031e).to_bytes(6, byteorder='big', signed=False) # lvds 4
            # data_FPGA_config = (0x01020102011e).to_bytes(6, byteorder='big', signed=False)   # lvds 2
            data_packet_config = (0xc005350c0000).to_bytes(6, byteorder='big', signed=False)

            # connect to DCA1000
            connect_to_FPGA = header + CODE_9 + dataSize_0 + footer
            read_FPGA_version = header + CODE_E + dataSize_0 + footer
            config_FPGA = header + CODE_3 + dataSize_6 + data_FPGA_config + footer
            config_packet = header + CODE_B + dataSize_6 + data_packet_config + footer
            start_record = header + CODE_5 + dataSize_0 + footer
            stop_record = header + CODE_6 + dataSize_0 + footer

            if code == '9':
                re = connect_to_FPGA
            elif code == 'E':
                re = read_FPGA_version
            elif code == '3':
                re = config_FPGA
            elif code == 'B':
                re = config_packet
            elif code == '5':
                re = start_record
            elif code == '6':
                re = stop_record
            else:
                re = 'NULL'
            # print('send command:', re.hex())
            return re

    def RDI_update(self):
            global count, view_rai, p13d, RDIData
            win_param = [8, 8, 3, 3]
            # cfar_rai = CA_CFAR(win_param, threshold=2.5, rd_size=[64, 181])
            if not RDIData.empty():
                rd = RDIData.get()
                # ------no static remove------
                img_rdi.setImage(np.rot90(rd, 1), levels=[40, 180])
                # ------ static remove------
                # img_rdi.setImagemage(np.rot90(rd, 1), levels=[40, 150])

    def RAI_update(self):
            global count, view_rai, p13d, RAIData
            if not RAIData.empty():
                a= RAIData.get()
                # ------no static remove------
                img_rai.setImage(np.fliplr(np.flip(a, axis=0)).T, levels=[10.0e4, 30.0e4])
                # ------ static remove ------
                # img_rai.setImage(np.fliplr(np.flip(a, axis=0)).T, levels=[0.5e4, 9.0e4])

    def PD_update(self):
            global count, view_rai, p13d,nice,ax,rawData,pointcloud
            # --------------- plot 3d ---------------
            # image_pro.process(cam1.get_frame(),cam2.get_frame(),'aaa')

            pos = pointcloud.get()
            meanpos=np.mean(pos,axis=1)

            # --------------- Time Average ----------

            # pd_time_avg[0:-1, ::] = pd_time_avg[1:, ::]
            # pd_time_avg[-1, ::] = pd_time_avg[0, ::]
            # pd_time_avg[maxsize-1, ::] = meanpos
            #
            # meanpos_timeavg=np.mean(pd_time_avg,axis=0)
            # print(meanpos_timeavg)

            # ---------------------------------------

            # pos = np.delete(pos[:3], np.where((pos[:,3] >10) & (pos[:,3]<50))[0], axis=0)
            pos = np.transpose(pos,[1,0])
            # self.Run_PD(pos)
            if self.pd_save_status == 1 :
                # self.pd_save.append(pos)
                # self.rdi = np.append(self.rdi, RDIData.get())
                # self.rai = np.append(self.rai, RAIData.get())
                self.data.append(rawData.get())
            p13d.setData(pos=pos[:,:3],color= [1,0.35,0.02,1],pxMode= True)

            # self.origin_QQ.setData(pos=np.array([meanpos[0],meanpos[1],meanpos[2]]),color=[140,140,140,255],size=20)
            # self.origin_QQ.setData(pos=np.array([meanpos_timeavg[0], meanpos_timeavg[1], meanpos_timeavg[2]]), color=[140, 140, 140, 255], size=20)
            # print(meanpos)

    def update_figure(self):
            global count,view_rai,p13d

            QtCore.QTimer.singleShot(1, self.RDI_update)
            QtCore.QTimer.singleShot(1, self.RAI_update)
            QtCore.QTimer.singleShot(1, self.PD_update)
            QtCore.QTimer.singleShot(1, self.update_figure)
            # QApplication.processEvents()
            # QtCore.QTimer.singleShot(1, Run_PD())
            now = ptime.time()
            updateTime = now
            count += 1

    def openradar(self):
            # if not CAMData.empty():
            #     set_radar.StopRadar()
            #     set_radar.SendConfig(config)
            #     update_figure()
            set_radar.StopRadar()
            set_radar.SendConfig(config)
            print('======================================')
            self.update_figure()

    def StartRecord(self):
            # processor.status = 1
            self.collector.status = 1
            # cam1.status = 1
            # cam2.status = 1
            self.pd_save_status = 1
            print('Start Record Time:', (time.ctime(time.time())))
            print('=======================================')

    def StopRecord(self):
            # processor.status = 0
            self.collector.status = 0
            # cam1.status = 0
            # cam2.status = 0
            self.pd_save_status=0
            print('Stop Record Time:', (time.ctime(time.time())))
            print('=======================================')

    def ConnectDca(self):
            global sockConfig, FPGA_address_cfg
            print('Connect to DCA1000')
            print('=======================================')
            config_address = ('192.168.33.30', 4096)
            FPGA_address_cfg = ('192.168.33.180', 4096)
            cmd_order = ['9', 'E', '3', 'B', '5', '6']
            sockConfig = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sockConfig.bind(config_address)
            for k in range(5):
                # Send the command
                sockConfig.sendto(self.send_cmd(cmd_order[k]), FPGA_address_cfg)
                time.sleep(0.1)
                # Request data back on the config port
                msg, server = sockConfig.recvfrom(2048)
                # print('receive command:', msg.hex())

    def SelectFolder(self):
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.asksaveasfilename(parent=root, initialdir='D:/kaiku_report/20210429/')
            self.path = file_path
            return file_path

    def Run_PD(self,data):
            plot_pd(data)
            # return  None

    def SaveData(self):
            global savefilename, sockConfig, FPGA_address_cfg
            # set_radar.StopRadar(
            path = self.SelectFolder()

            # np.save(self.path+"pd.npy", self.pd_save)
            # np.save(self.path+"RDI.npy", self.rdi)
            # np.save(self.path+"RAI.npy", self.rai)
            np.save(self.path+"data.npy", self.data)
            # QtCore.QTimer.singleShot(0, self.app.deleteLater)

            # img_rdi.clear()
            # img_cam.clear()

    def plot(self):
            global img_rdi, img_rai, updateTime, view_text, count, angCurve, ang_cuv, img_cam, savefilename,view_rai,p13d,nice
            # ---------------------------------------------------
            # self.app = QtWidgets.QApplication(sys.argv)

            tmp_data = np.zeros(181)
            # angCurve = pg.plot(tmp_data, pen='r')
            ui = self.ui

            view_rdi = ui.graphicsView.addViewBox()
            view_rai = ui.graphicsView_2.addViewBox()
            view_cam = ui.graphicsView_3
            starbtn = ui.pushButton_start
            exitbtn = ui.pushButton_exit
            recordbtn = ui.pushButton_record
            stoprecordbtn = ui.pushButton_stop_record
            savebtn = ui.pushButton_save
            dcabtn = ui.pushButton_DCA
            savefilename = ui.label_3
            pd_btn = ui.pd_btn
            # savefilename = ui.textEdit
            # ---------------------------------------------------
            # lock the aspect ratio so pixels are always square
            view_rdi.setAspectLocked(True)
            view_rai.setAspectLocked(True)
            # view_cam.setAspectLocked(True)
            img_rdi = pg.ImageItem(border='w')
            img_rai = pg.ImageItem(border='w')
            # img_cam = pg.ImageItem(border='w')
            #-----------------
            xgrid = gl.GLGridItem()
            ygrid = gl.GLGridItem()
            zgrid = gl.GLGridItem()
            view_cam.addItem(xgrid)
            view_cam.addItem(ygrid)
            view_cam.addItem(zgrid)
            xgrid.translate(0,10,-10)
            ygrid.translate(0, 0, 0)
            zgrid.translate(0, 10, -10)
            xgrid.rotate(90, 0, 1, 0)
            ygrid.rotate(90, 1, 0, 0)
            pos = np.random.randint(-10, 10, size=(1000, 3))
            pos[:, 2] = np.abs(pos[:, 2])
            p13d = gl.GLScatterPlotItem(pos = pos,color=[50, 50, 50, 255])
            # origin = gl.GLScatterPlotItem(pos = np.zeros([1,3]),color=[255, 0, 0, 255])\

            origin = gl.GLScatterPlotItem(pos = np.array([[0, 0.075 ,0],[0, 0.075*2 ,0],[0, 0.075*3 ,0],[0, 0.075*4 ,0],[0, 0.075*5 ,0],[0, 0.075*6 ,0]]),color=[255, 255, 255, 255])
            origin1 = gl.GLScatterPlotItem(pos = np.array([[0.075*-3, 0.3 ,0],[0.075*-2,0.3,0],[0.075*-1,0.3,0],[0.075*1,0.3,0],[0.075*2,0.3,0],[0.075*3,0.3,0]]),color=[255, 255, 255, 255])
            # origin2 = gl.GLScatterPlotItem(pos = np.array([[0, 0.3 ,0.075*-3],[0,0.3,0.075*-2],[0,0.3,0.075*-1],[0,0.3,0.075*1],[0,0.3,0.075*2],[0,0.3,0.075*3]]),color=[255, 255, 255, 255])
            # view_cam.addItem(origin1)
            # view_cam.addItem(origin2)
            origin_P = gl.GLScatterPlotItem(pos=np.array(
                [[0, 0, 0]]), color=[255, 0, 0, 255])
            view_cam.addItem(origin_P)
            '''# x軸
            self.hand_line = gl.GLLinePlotItem(pos=np.array([[[0, 0, 0], [0, 0.075*10, 0]]]),
                                               color=[128, 255, 128, 255], antialias=False)
            view_cam.addItem(self.hand_line)
            # y軸
            self.hand_line1= gl.GLLinePlotItem(pos=np.array([[[-0.075*8, 0.075*4, 0], [0.075*8, 0.075*4, 0]]]),
                                               color=[128, 255, 128, 255], antialias=False)
            view_cam.addItem(self.hand_line1)
    
            self.hand_liney1 = gl.GLLinePlotItem(pos=np.array([[[-0.075 * 8, 0.075 * 4, 0.075*2], [0.075 * 8, 0.075 * 4, 0.075*2]]]),
                                                color=[128, 255, 128, 255], antialias=False)
            view_cam.addItem(self.hand_liney1)
    
            self.hand_liney2 = gl.GLLinePlotItem(pos=np.array([[[-0.075 * 8, 0.075 * 4, -0.075*2], [0.075 * 8, 0.075 * 4, -0.075*2]]]),
                                                 color=[128, 255, 128, 255], antialias=False)
            view_cam.addItem(self.hand_liney2)
            # z軸
            self.hand_line2= gl.GLLinePlotItem(pos=np.array([[[0, 0.075*4, -0.075*8], [0, 0.075*4, 0.075*8]]]),
                                               color=[128, 255, 128, 255], antialias=False)
            view_cam.addItem(self.hand_line2)'''

            '''self.hand_line = gl.GLLinePlotItem(pos=np.array([[[0.075*1, 0.075*4, 0.075*4], [0.075*1, 0.075 * 4, -0.075*4]]]),
                                               color=[128, 255, 128, 255], antialias=False)
            view_cam.addItem(self.hand_line)
    
            self.hand_line1 = gl.GLLinePlotItem(pos=np.array([[[-0.075 * 1, 0.075 * 4, 0.075 * 4], [-0.075 * 1, 0.075 * 4, -0.075 * 4]]]),
                                               color=[128, 255, 128, 255], antialias=False)
            view_cam.addItem(self.hand_line1)
    
            self.hand_line2 = gl.GLLinePlotItem(pos=np.array([[[0.075 * 3, 0.075 * 4, 0.075 * 4], [0.075 * 3, 0.075 * 4, -0.075 * 4]]]),
                                               color=[128, 255, 128, 255], antialias=False)
            view_cam.addItem(self.hand_line2)
    
            self.hand_line3 = gl.GLLinePlotItem(pos=np.array([[[-0.075 * 3, 0.075 * 4, 0.075 * 4], [-0.075 * 3, 0.075 * 4, -0.075 * 4]]]),
                                                color=[128, 255, 128, 255], antialias=False)
            view_cam.addItem(self.hand_line3)
    
            self.hand_line4 = gl.GLLinePlotItem(
                pos=np.array([[[0.075 * 4, 0.075 * 4, 0.075 * 1], [-0.075 * 4, 0.075 * 4, 0.075 * 1]]]),
                color=[128, 255, 128, 255], antialias=False)
            view_cam.addItem(self.hand_line4)
    
            self.hand_line5 = gl.GLLinePlotItem(
                pos=np.array([[[0.075 * 4, 0.075 * 4, -0.075 * 1], [-0.075 * 4, 0.075 * 4, -0.075 * 1]]]),
                color=[128, 255, 128, 255], antialias=False)
            view_cam.addItem(self.hand_line5)
    
            self.hand_line6 = gl.GLLinePlotItem(
                pos=np.array([[[0.075 * 4, 0.075 * 4, 0.075 * 3], [-0.075 * 4, 0.075 * 4, 0.075 * 3]]]),
                color=[128, 255, 128, 255], antialias=False)
            view_cam.addItem(self.hand_line6)
    
            self.hand_line7 = gl.GLLinePlotItem(
                pos=np.array([[[0.075 * 4, 0.075 * 4, -0.075 * 3], [-0.075 * 4, 0.075 * 4, -0.075 * 3]]]),
                color=[128, 255, 128, 255], antialias=False)
            view_cam.addItem(self.hand_line7)'''

            self.hand_line = gl.GLLinePlotItem(
                pos=np.array([[[0.075 * 4, 0.075 * 1, 0], [-0.075 * 4, 0.075 * 1, 0]]]),
                color=[128, 255, 128, 255], antialias=False)
            view_cam.addItem(self.hand_line)
            self.hand_line2 = gl.GLLinePlotItem(
                pos=np.array([[[0.075 * 4, 0.075 * 3, 0], [-0.075 * 4, 0.075 * 3, 0]]]),
                color=[128, 255, 128, 255], antialias=False)
            view_cam.addItem(self.hand_line2)
            self.hand_line3 = gl.GLLinePlotItem(
                pos=np.array([[[0.075 * 4, 0.075 * 5, 0], [-0.075 * 4, 0.075 * 5, 0]]]),
                color=[128, 255, 128, 255], antialias=False)
            view_cam.addItem(self.hand_line3)
            self.hand_line4 = gl.GLLinePlotItem(
                pos=np.array([[[0.075 * 4, 0.075 * 7, 0], [-0.075 * 4, 0.075 * 7, 0]]]),
                color=[128, 255, 128, 255], antialias=False)
            view_cam.addItem(self.hand_line4)

            self.hand_line5 = gl.GLLinePlotItem(
                pos=np.array([[[0.075 * 1, 0.075 * 1, 0], [0.075 * 1, 0.075 * 7, 0]]]),
                color=[128, 255, 128, 255], antialias=False)
            view_cam.addItem(self.hand_line5)

            self.hand_line6= gl.GLLinePlotItem(
                pos=np.array([[[-0.075 * 1, 0.075 * 1, 0], [-0.075 * 1, 0.075 * 7, 0]]]),
                color=[128, 255, 128, 255], antialias=False)
            view_cam.addItem(self.hand_line6)

            self.hand_line7 = gl.GLLinePlotItem(
                pos=np.array([[[0.075 * 3, 0.075 * 1, 0], [0.075 * 3, 0.075 * 7, 0]]]),
                color=[128, 255, 128, 255], antialias=False)
            view_cam.addItem(self.hand_line7)

            self.hand_line8 = gl.GLLinePlotItem(
                pos=np.array([[[-0.075 * 3, 0.075 * 1, 0], [-0.075 * 3, 0.075 * 7, 0]]]),
                color=[128, 255, 128, 255], antialias=False)
            view_cam.addItem(self.hand_line8)

            self.origin_QQ = gl.GLScatterPlotItem(pos=np.array([0,0,0]), color=[140, 140, 140, 255],size=20)
            view_cam.addItem(self.origin_QQ)


            # coord = gl.GLAxisItem(glOptions="opaque")
            # coord.setSize(10, 10, 10)
            view_cam.addItem(p13d)
            # view_cam.addItem(coord)
            # view_cam.addItem(origin)



            # ang_cuv = pg.PlotDataItem(tmp_data, pen='r')
            # Colormap
            position = np.arange(64)
            position = position / 64
            position[0] = 0
            position = np.flip(position)
            colors = [[62, 38, 168, 255], [63, 42, 180, 255], [65, 46, 191, 255], [67, 50, 202, 255], [69, 55, 213, 255],
                      [70, 60, 222, 255], [71, 65, 229, 255], [70, 71, 233, 255], [70, 77, 236, 255], [69, 82, 240, 255],
                      [68, 88, 243, 255],
                      [68, 94, 247, 255], [67, 99, 250, 255], [66, 105, 254, 255], [62, 111, 254, 255], [56, 117, 254, 255],
                      [50, 123, 252, 255],
                      [47, 129, 250, 255], [46, 135, 246, 255], [45, 140, 243, 255], [43, 146, 238, 255], [39, 150, 235, 255],
                      [37, 155, 232, 255],
                      [35, 160, 229, 255], [31, 164, 225, 255], [28, 129, 222, 255], [24, 173, 219, 255], [17, 177, 214, 255],
                      [7, 181, 208, 255],
                      [1, 184, 202, 255], [2, 186, 195, 255], [11, 189, 188, 255], [24, 191, 182, 255], [36, 193, 174, 255],
                      [44, 195, 167, 255],
                      [49, 198, 159, 255], [55, 200, 151, 255], [63, 202, 142, 255], [74, 203, 132, 255], [88, 202, 121, 255],
                      [102, 202, 111, 255],
                      [116, 201, 100, 255], [130, 200, 89, 255], [144, 200, 78, 255], [157, 199, 68, 255], [171, 199, 57, 255],
                      [185, 196, 49, 255],
                      [197, 194, 42, 255], [209, 191, 39, 255], [220, 189, 41, 255], [230, 187, 45, 255], [239, 186, 53, 255],
                      [248, 186, 61, 255],
                      [254, 189, 60, 255], [252, 196, 57, 255], [251, 202, 53, 255], [249, 208, 50, 255], [248, 214, 46, 255],
                      [246, 220, 43, 255],
                      [245, 227, 39, 255], [246, 233, 35, 255], [246, 239, 31, 255], [247, 245, 27, 255], [249, 251, 20, 255]]
            colors = np.flip(colors, axis=0)
            color_map = pg.ColorMap(position, colors)
            lookup_table = color_map.getLookupTable(0.0, 1.0, 256)
            img_rdi.setLookupTable(lookup_table)
            img_rai.setLookupTable(lookup_table)
            view_rdi.addItem(img_rdi)
            view_rai.addItem(img_rai)
            # view_cam.addItem(img_cam)
            # view_angCurve.addItem(ang_cuv)
            #
            # Set initial view bounds
            view_rdi.setRange(QtCore.QRectF(-5, 0, 140, 80))
            view_rai.setRange(QtCore.QRectF(10, 0, 160, 80))
            updateTime = ptime.time()
            #----------------- btn clicked connet -----------------
            starbtn.clicked.connect(self.openradar)
            recordbtn.clicked.connect(self.StartRecord)
            stoprecordbtn.clicked.connect(self.StopRecord)
            savebtn.clicked.connect(self.SaveData)
            dcabtn.clicked.connect(self.ConnectDca)
            exitbtn.clicked.connect(self.app.instance().exit)
            # pd_btn.btn.clicked.connect(self.Run_PD)
            # -----------------------------------------------------
            # self.app.instance().exec_()
            set_radar.StopRadar()
            print('=======================================')

    # def run(self):
    #     global BinData, RDIData, RAIData, rawData, pointcloud
    #     print('======Real Time Data Capture Tool======')
    #     count = 0
    #     # Queue for access data
    #     maxsize = 5
    #     realtime = Realtime_sys()
    #     BinData = Queue()
    #     RDIData = Queue()
    #     RAIData = Queue()
    #     CAMData = Queue()
    #     CAMData2 = Queue()
    #     rawData = Queue()
    #     pointcloud = Queue()
    #     cam_rawData = Queue()
    #     cam_rawData2 = Queue()
    #     pd_time_avg = np.zeros(maxsize * 4).reshape(maxsize, 4)
    #     # Radar config
    #     adc_sample = 64
    #     chirp = 16
    #     tx_num = 3
    #     rx_num = 4
    #     radar_config = [adc_sample, chirp, tx_num, rx_num]
    #     frame_length = adc_sample * chirp * tx_num * rx_num * 2
    #     # Host setting
    #     address = ('192.168.33.30', 4098)
    #     buff_size = 2097152
    #
    #     # # config DCA1000 to receive bin data
    #     # config_address = ('192.168.33.30', 4096)
    #     # FPGA_address_cfg = ('192.168.33.180', 4096)
    #     # cmd_order = ['9', 'E', '3', 'B', '5', '6']
    #     # sockConfig = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #     # sockConfig.bind(config_address)
    #
    #     opencamera = False
    #     # image_pro  = deal_imag()
    #     # image_pro.start()
    #     # lock = threading.Lock()
    #     if opencamera:
    #         cam1 = CamCapture(1, 'First', 1, "lock", CAMData, cam_rawData, mode=1, mp4_path="D:/kaiku_report/20210429/")
    #         cam2 = CamCapture(0, 'Second', 0, "lock", CAMData2, cam_rawData2, mode=1,
    #                           mp4_path="D:/kaiku_report/20210429/")
    #     collector = UdpListener('Listener', BinData, frame_length, address, buff_size, rawData)
    #     processor = DataProcessor('Processor', radar_config, BinData, RDIData, RAIData, pointcloud, 0, "0105", status=0)
    #     if opencamera:
    #         cam1.start()
    #         cam2.start()
    #     collector.start()
    #     processor.start()
    #     plotIMAGE = threading.Thread(target=realtime.plot())
    #     plotIMAGE.start()
    #
    #     # sockConfig.sendto(send_cmd('6'), FPGA_address_cfg)
    #     # sockConfig.close()
    #     cam1.join(timeout=1)
    #     cam2.join(timeout=1)
    #     collector.join(timeout=1)
    #     processor.join(timeout=1)
    #     if opencamera:
    #         cam1.close()
    #         cam2.close()
    #
    #     print("Program Close")

class realtime(QThread):
    def __init__(self):
        super().__init__()
    def run(self):
        global count,BinData, RDIData, RAIData, rawData, pointcloud, ui_radar
        print('======Real Time Data Capture Tool======')
        count = 0
        # Queue for access data
        maxsize = 5
        realtime = Realtime_sys(ui_radar)
        BinData = Queue()
        RDIData = Queue()
        RAIData = Queue()
        CAMData = Queue()
        CAMData2 = Queue()
        rawData = Queue()
        pointcloud = Queue()
        cam_rawData = Queue()
        cam_rawData2 = Queue()
        pd_time_avg = np.zeros(maxsize * 4).reshape(maxsize, 4)
        # Radar config
        adc_sample = 64
        chirp = 16
        tx_num = 3
        rx_num = 4
        radar_config = [adc_sample, chirp, tx_num, rx_num]
        frame_length = adc_sample * chirp * tx_num * rx_num * 2
        # Host setting
        address = ('192.168.33.30', 4098)
        buff_size = 2097152

        # # config DCA1000 to receive bin data
        # config_address = ('192.168.33.30', 4096)
        # FPGA_address_cfg = ('192.168.33.180', 4096)
        # cmd_order = ['9', 'E', '3', 'B', '5', '6']
        # sockConfig = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # sockConfig.bind(config_address)

        # image_pro  = deal_imag()
        # image_pro.start()
        # lock = threading.Lock()

        collector = UdpListener('Listener', BinData, frame_length, address, buff_size, rawData)

        processor = DataProcessor('Processor', radar_config, BinData, RDIData, RAIData, pointcloud, 0, "0105", status=0)

        collector.start()
        processor.start()

        plotIMAGE = threading.Thread(target=realtime.plot())
        plotIMAGE.start()

        # sockConfig.sendto(send_cmd('6'), FPGA_address_cfg)
        # sockConfig.close()

        collector.join(timeout=1)
        processor.join(timeout=1)


        print("Program Close")





class MainWindow(QMainWindow):
    def __init__(self):
        global ui_radar
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        Wid_radar = QWidget()
        ui_radar = Ui_MainWindow_radar()
        ui_radar.setupUi(Wid_radar)

        self.Query=realtime()

        # self.send_Query=Realtime_sys(self.ui_radar)

        # Window title
        self.setWindowTitle("車載主機")
        UIFunctions.labelTitle(self, 'Application of Real-time-Radar')
        UIFunctions.labelDescription(self, 'NTUT_Project')

        # WINDOW SIZE ==> DEFAULT SIZE
        startSize = QSize(1000, 720)
        self.resize(startSize)
        self.setMinimumSize(startSize)

        # ==> TOGGLE MENU SIZE
        self.ui.btn_toggle_menu.clicked.connect(lambda: UIFunctions.toggleMenu(self, 220, True))
        # ==> END ##

        # ==> ADD CUSTOM MENUS 新增左側的BUTTON
        self.ui.stackedWidget.setMinimumWidth(20)
        UIFunctions.addNewMenu(self, "HOME", "btn_home", "url(:/16x16/icons/16x16/cil-home.png)", True)
        #UIFunctions.addNewMenu(self, "Add User", "btn_new_user", "url(:/16x16/icons/16x16/cil-user-follow.png)", True)
        UIFunctions.addNewMenu(self, "Custom Widgets", "btn_widgets", "url(:/16x16/icons/16x16/cil-equalizer.png)",
                               False)
        UIFunctions.addNewMenu(self, "Radar", "btn_radar","url(:/16x16/icons/16x16/cil-equalizer.png)",
                               False)

        UIFunctions.addNewMenu(self, "Calculator", "btn_Cal", "url(:/16x16/icons/16x16/cil-plus.png)", True)
        UIFunctions.addNewMenu(self, "Music Player", "btn_MP3" ,"url(:/16x16/icons/16x16/cil-speaker.png)",True)
        # ==> END ##

        # START MENU => SELECTION
        UIFunctions.selectStandardMenu(self, "btn_home")
        # ==> END ##

        # ==> START PAGE
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_home)
        # ==> END ##

        # USER ICON ==> SHOW HIDE (False)
        UIFunctions.userIcon(self, "WM", "", False)
        # ==> END ##

        # ==> MOVE WINDOW / MAXIMIZE / RESTORE
        ########################################################################
        def moveWindow(event):
            # IF MAXIMIZED CHANGE TO NORMAL
            if UIFunctions.returStatus() == 1:
                UIFunctions.maximize_restore(self)

            # MOVE WINDOW
            if event.buttons() == Qt.LeftButton:
                self.move(self.pos() + event.globalPos() - self.dragPos)
                self.dragPos = event.globalPos()
                event.accept()

        # WIDGET TO MOVE
        self.ui.frame_label_top_btns.mouseMoveEvent = moveWindow
        # ==> END ##

        # ==> LOAD DEFINITIONS
        ########################################################################
        UIFunctions.uiDefinitions(self)
        # ==> END ##

        # ==> QTableWidget PARAMETERS
        ########################################################################
        self.ui.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        # ==> END ##

        # SHOW ==> MAIN WINDOW
        ########################################################################
        self.show()
        # ==> END ##

    def Button(self):
        global ui_radar
        # GET BT CLICKED
        btnWidget = self.sender()

        # PAGE HOME
        if btnWidget.objectName() == "btn_home":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_home)
            UIFunctions.resetStyle(self, "btn_home")
            UIFunctions.labelPage(self, "Home")
            btnWidget.setStyleSheet(UIFunctions.selectMenu(btnWidget.styleSheet()))

        # PAGE NEW USER
        if btnWidget.objectName() == "btn_new_user":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_home)
            UIFunctions.resetStyle(self, "btn_new_user")
            UIFunctions.labelPage(self, "New User")
            btnWidget.setStyleSheet(UIFunctions.selectMenu(btnWidget.styleSheet()))

        # PAGE Test
        if btnWidget.objectName() == "btn_Cal":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_cal)
            UIFunctions.resetStyle(self, "btn_Cal")
            UIFunctions.labelPage(self, "Calculator")
            btnWidget.setStyleSheet(UIFunctions.selectMenu(btnWidget.styleSheet()))

        # PAGE Test
        if btnWidget.objectName() == "btn_MP3":
            ui_radar.show()
            # self.Query.start()
            # self.ui.stackedWidget.setCurrentWidget(self.ui.page_mp3)
            # UIFunctions.resetStyle(self, "btn_MP3")
            # UIFunctions.labelPage(self, "music_player")
            # btnWidget.setStyleSheet(UIFunctions.selectMenu(btnWidget.styleSheet()))

        # PAGE WIDGETS
        if btnWidget.objectName() == "btn_widgets":
            self.ui.stackedWidget.setCurrentWidget(self.ui.page_widgets)
            UIFunctions.resetStyle(self, "btn_widgets")
            UIFunctions.labelPage(self, "Custom Widgets")
            btnWidget.setStyleSheet(UIFunctions.selectMenu(btnWidget.styleSheet()))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    QtGui.QFontDatabase.addApplicationFont('./fonts/segoeui.ttf')
    QtGui.QFontDatabase.addApplicationFont('./fonts/segoeuib.ttf')
    window = MainWindow()
    sys.exit(app.exec_())