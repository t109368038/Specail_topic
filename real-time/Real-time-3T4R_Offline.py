from offline_process_3t4r import DataProcessor
from tkinter import filedialog
import tkinter as tk
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
import pyqtgraph.opengl as gl
import numpy as np
import threading
import cv2
import sys
import read_binfile
import time
from camera_offine import CamCapture
# -----------------------------------------------
from app_layout_2t4r_offline import Ui_MainWindow
from R3t4r_to_point_cloud_for_realtime import plot_pd
from  plot_ground_truth import get_gt
class Realtime_sys():
    def __init__(self):
        self.pd_save_status = 0
        self.pd_save = []
        self.rdi = []
        self.rai = []
        self.pd = []
        self.btn_status = False
        self.run_state  = False
        self.sure_next = True
        self.frame_count = 0
        self.data_proecsss = DataProcessor()
        self.cam_hp = np.load('D:\\kaiku_report\\20210429\\' + "cam_hp.npy", allow_pickle=True)
        self.cam1_hp = np.load('D:\\kaiku_report\\20210429\\' + "cam_hp1.npy", allow_pickle=True)
        self.cam_x = self.cam_hp[::2]
        self.cam_y = self.cam_hp[1::2]
        self.cam1_x = self.cam1_hp[::2]
        self.cam1_y = self.cam1_hp[1::2]



    def start(self):
        self.run_state = True
        self.frame_count = 0
        self.stop_btn.setText("stop")
        self.th_cam1 = CamCapture(self.file_path_cam1)
        self.th_cam2 = CamCapture(self.file_path_cam2)
        self.update_figure()


    def stop(self):
        if self.run_state:
            self.stop_btn.setText("Continues")
        else:
            self.stop_btn.setText("stop")
        self.run_state = not(self.run_state)
        self.sure_next = True
        self.update_figure()

    def RDI_update(self):
        rd = self.RDI
        img_rdi.setImage(np.rot90(rd, -1))

    def RAI_update(self):
        global count, view_rai, p13d
        a = self.RAI
        img_rai.setImage(np.fliplr((a)).T)

    def PD_update(self):
        global count, view_rai, p13d,nice,ax
        rd = self.RDI
        img_rdi.setImage(np.rot90(rd, -1))
        a = self.RAI
        img_rai.setImage(np.fliplr((a)).T)
        # --------------- plot 3d ---------------
        pos = self.PD
        pos = np.transpose(pos,[1,0])
        self.pd.append(pos)
        p13d.setData(pos=pos[:,:3],color= [1,0.35,0.02,1],pxMode= True)
    def plot_gt(self):
        x,y,z =get_gt(self.cam_x[self.frame_count],self.cam_y[self.frame_count],self.cam1_x[self.frame_count],self.cam1_y[self.frame_count])
        # hand = [np.array(x),np.array(y),np.array(z)]
        hand =np.zeros([21,3])
        hand[:,0] = x * -1
        hand[:,1] = y * 1
        hand[:,2] = z * -1
        print(hand.shape)
        line = np.array(
            [[hand[0, :], hand[1, :]], [hand[1, :], hand[2, :]], [hand[2, :], hand[3, :]], [hand[3, :], hand[4, :]],
             [hand[0, :], hand[5, :]],
             [hand[5, :], hand[6, :]], [hand[6, :], hand[7, :]], [hand[7, :], hand[8, :]], [hand[5, :], hand[9, :]],
             [hand[9, :], hand[10, :]],
             [hand[10, :], hand[11, :]], [hand[11, :], hand[12, :]], [hand[9, :], hand[13, :]],
             [hand[13, :], hand[14, :]], [hand[14, :], hand[15, :]],
             [hand[15, :], hand[16, :]], [hand[13, :], hand[17, :]], [hand[17, :], hand[18, :]],
             [hand[18, :], hand[19, :]], [hand[19, :], hand[20, :]],
             [hand[0, :], hand[17, :]]])

        self.hand_line.setData(pos=line, color=[0.5, 0.7, 0.9, 255], antialias=False)

        self.indexfinger.setData(pos=hand[8, :], color=pg.glColor((250, 0, 255)), pxMode=True)
        self.thumb.setData(pos=hand[4, :], color=pg.glColor((255, 255, 0)), pxMode=True)


    def update_figure(self):
        global count,view_rai,p13d
        self.Sure_staic_RM = self.static_rm.isChecked()
        if len(self.rawData)<= self.frame_count:
            np.save('D:\\kaiku_report\\20210429\\rm_pd.npy',self.pd)

        if self.run_state:
            self.RDI ,self.RAI,self.PD = self.data_proecsss.run_proecss(self.rawData[self.frame_count],\
                                                            self.rai_mode,self.Sure_staic_RM,self.chirp)
            self.RDI_update()
            self.RAI_update()
            self.PD_update()
            self.plot_gt()
            time.sleep(0.05)
            if self.sure_next:
                self.frame_count +=1
                QtCore.QTimer.singleShot(1, self.update_figure)
                QApplication.processEvents()
            self.image_label1.setPixmap(self.th_cam1.get_frame())
            img2 = self.th_cam2.get_frame()
            print(type(img2))
            self.image_label2.setPixmap(img2)
        else :
            pass
        # print(self.frame_count)

    def pre_frame(self):
        if self.frame_count >0:
            self.frame_count -=1
            self.sure_next = False
            self.run_state=True
            self.update_figure()

    def next_frame(self):
        if self.frame_count<=62:
            self.frame_count -= 1
            self.sure_next = False
            self.run_state = True
            self.update_figure()

    def SelectFolder(self):
        root = tk.Tk()
        root.withdraw()
        # self.file_path = filedialog.askopenfilename(parent=root, initialdir='D:\\Matt_yen_data\\NAS\\data\\bin file_processed\\new data(low powered)\\3t4r')
        self.file_path = filedialog.askopenfilename(parent=root, initialdir='D:/kaiku_report/20210429/')

        # self.file_path = "D:\\kaiku_report\\20210429\\rigt5data.npy"
        self.browse_text.setText(self.file_path)

        return self.file_path

    def SelectFolder_cam1(self):
        # root = tk.Tk()
        # root.withdraw()
        # self.file_path = filedialog.askopenfilename(parent=root, initialdir='D:\\Matt_yen_data\\NAS\\data\\bin file_processed\\new data(low powered)\\3t4r')
        # self.file_path_cam1 = filedialog.askopenfilename(parent=root, initialdir='D:/kaiku_report/2021-0418for_posheng/')
        self.file_path_cam1 = "D:\\kaiku_report\\20210429\\video0.mp4"
        self.browse_text_cam1.setText(self.file_path_cam1)

        return self.browse_text_cam1

    def SelectFolder_cam2(self):
        # root = tk.Tk()
        # root.withdraw()
        # self.file_path = filedialog.askopenfilename(parent=root, initialdir='D:\\Matt_yen_data\\NAS\\data\\bin file_processed\\new data(low powered)\\3t4r')
        # self.file_path_cam2 = filedialog.askopenfilename(parent=root, initialdir='D:/kaiku_report/2021-0418for_posheng/')
        self.file_path_cam2 = "D:\\kaiku_report\\20210429\\video1.mp4"
        self.browse_text_cam2.setText(self.file_path_cam2)

        return self.file_path_cam2


    def enable_btns(self,state):
        self.pre_btn.setEnabled(state)
        self.next_btn.setEnabled(state)
        self.start_btn.setEnabled(state)
        self.stop_btn.setEnabled(state)

    def slot(self, object):
        print("Key was pressed, id is:", self.radio_group.id(object))
        '''
        raimode /0/1/2:
                0 -> FFT-RAI
                1 -> beamformaing RAI 
                3 -> static clutter removal
        '''
        self.rai_mode = self.radio_group.id(object)
        if self.rai_mode ==1:
            self.view_rai.setRange(QtCore.QRectF(10, 0, 170, 80))
        else:
            self.view_rai.setRange(QtCore.QRectF(-5, 0, 100, 60))

    def load_file(self):
        load_mode = 1
        if load_mode == 0 :
            self.rawData =read_binfile.read_bin_file(self.file_path,[64,64,32,3,4],mode=0,header=False,packet_num=4322)
            self.rawData = np.transpose(self.rawData,[0,1,3,2])
            self.chirp = 32
        elif load_mode == 1:
            data  =np.load(self.file_path,allow_pickle=True)
            data = np.reshape(data, [-1, 4])
            data = data[:, 0:2:] + 1j * data[:, 2::]
            self.rawData = np.reshape(data,[-1,48,4,64])
            self.chirp = 16



        print(np.shape(self.rawData))
        # print(self.browse_text_cam1.text)
        # if self.browse_text_cam1 != None
        # self.th_cam1 = CamCapture()
        # self.th_cam1 = CamCapture()

        self.enable_btns(True)


    def plot(self):
        global img_rdi, img_rai, updateTime, view_text, count, angCurve, ang_cuv, img_cam, savefilename,view_rai,p13d,nice
        # ---------------------------------------------------
        self.app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        MainWindow.show()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)

        self.browse_btn = ui.browse_btn
        self.browse_text = ui.textEdit
        self.browse_text_cam1 = ui.textEdit_cam1
        self.browse_text_cam2 = ui.textEdit_cam2
        self.load_btn = ui.load_btn
        self.start_btn = ui.start_btn
        self.stop_btn  =ui.stop_btn
        self.next_btn  =ui.next_btn
        self.pre_btn = ui.pre_btn
        self.radio_group =  ui.radio_btn_group
        self.static_rm = ui.sure_static
        self.cam1 = ui.label_cam1
        self.cam2 = ui.label_cam2
        self.cam1_btn =  ui.browse_cam1_btn
        self.cam2_btn =  ui.browse_cam2_btn
        self.image_label1 =ui.image_label1
        self.image_label2 =ui.image_label2
        # #----------------- btn clicked connet -----------------
        self.browse_btn.clicked.connect(self.SelectFolder)
        self.cam1_btn.clicked.connect(self.SelectFolder_cam1)
        self.cam2_btn.clicked.connect(self.SelectFolder_cam2)
        self.load_btn.clicked.connect(self.load_file)
        self.start_btn.clicked.connect(self.start)
        self.next_btn.clicked.connect(self.next_frame)
        self.pre_btn.clicked.connect(self.pre_frame)
        self.stop_btn.clicked.connect(self.stop)
        self.radio_group.buttonClicked.connect(self.slot)
        # # -----------------------------------------------------
        self.view_rdi = ui.graphicsView.addViewBox()
        self.view_rai = ui.graphicsView_2.addViewBox()
        self.view_PD = ui.graphicsView_3
        # ---------------------------------------------------
        # lock the aspect ratio so pixels are always square
        self.view_rdi.setAspectLocked(True)
        self.view_rai.setAspectLocked(True)
        img_rdi = pg.ImageItem(border='w')
        img_rai = pg.ImageItem(border='w')
        img_cam = pg.ImageItem(border='w')
        #-----------------
        xgrid = gl.GLGridItem()
        ygrid = gl.GLGridItem()
        zgrid = gl.GLGridItem()
        self.view_PD.addItem(xgrid)
        self.view_PD.addItem(ygrid)
        self.view_PD.addItem(zgrid)
        xgrid.translate(0,10,-10)
        ygrid.translate(0, 0, 0)
        zgrid.translate(0, 10, -10)
        xgrid.rotate(90, 0, 1, 0)
        ygrid.rotate(90, 1, 0, 0)
        pos = np.random.randint(-10, 10, size=(1000, 3))
        pos[:, 2] = np.abs(pos[:, 2])
        p13d = gl.GLScatterPlotItem(pos = pos,color=[50, 50, 50, 255])
        # origin = gl.GLScatterPlotItem(pos = np.array([[0, 0, 0],[0, 0.1 ,0],[0, 0.2 ,0],[0, 0.3 ,0],[0, 0.4 ,0]]),color=[255, 0, 0, 255])

        coord = gl.GLAxisItem(glOptions="opaque")
        coord.setSize(10, 10, 10)
        self.view_PD.addItem(p13d)
        # self.view_PD.addItem(coord)
        # self.view_PD.addItem(origin)
        self.enable_btns(False)
        ###---------------------------------------------
        self.hand = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=[0, 255, 0, 255], pxMode=True)
        self.view_PD.addItem(self.hand)

        # self.hand_line = gl.GLLinePlotItem(pos=np.array([[[0, 0, 0], [2.5, 3.2, 1.5]], [[0, 0, 0], [1, 3.5, 4]]]),
        #                                    color=[128, 255, 128, 255], antialias=False)
        # self.view_PD.addItem(self.hand_line)

        self.indexfinger = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=[255, 0, 0, 255], pxMode=True)
        self.view_PD.addItem(self.indexfinger)

        self.thumb = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=[255, 0, 0, 255], pxMode=True)
        self.view_PD.addItem(self.thumb)

        origin = gl.GLScatterPlotItem(pos = np.array([[0, 0.075 ,0],[0, 0.075*2 ,0],[0, 0.075*3 ,0],[0, 0.075*4 ,0],[0, 0.075*5 ,0],[0, 0.075*6 ,0]]),color=[255, 255, 255, 255])
        origin1 = gl.GLScatterPlotItem(pos = np.array([[0.075*-3, 0.3 ,0],[0.075*-2,0.3,0],[0.075*-1,0.3,0],[0.075*1,0.3,0],[0.075*2,0.3,0],[0.075*3,0.3,0]]),color=[255, 255, 255, 255])
        origin2 = gl.GLScatterPlotItem(pos = np.array([[0, 0.3 ,0.075*-3],[0,0.3,0.075*-2],[0,0.3,0.075*-1],[0,0.3,0.075*1],[0,0.3,0.075*2],[0,0.3,0.075*3]]),color=[255, 255, 255, 255])
        self.view_PD.addItem(origin)
        self.view_PD.addItem(origin1)
        self.view_PD.addItem(origin2)
        origin_P = gl.GLScatterPlotItem(pos=np.array(
            [[0, 0, 0]]), color=[255, 0, 0, 255])
        self.view_PD.addItem(origin_P)
        self.hand_line = gl.GLLinePlotItem(pos=np.array([[[0, 0, 0], [0, 0.075*10, 0]]]),
                                           color=[128, 255, 128, 255], antialias=False)
        self.hand_linex = gl.GLLinePlotItem(pos=np.array([[[0, 0, 0], [0, 0.075*10, 0]]]),
                                           color=[128, 255, 128, 255], antialias=False)
        self.view_PD.addItem(self.hand_linex)
        self.view_PD.addItem(self.hand_line)
        self.hand_line1= gl.GLLinePlotItem(pos=np.array([[[-0.075*8, 0.075*4, 0], [0.075*8, 0.075*4, 0]]]),
                                           color=[128, 255, 128, 255], antialias=False)
        self.view_PD.addItem(self.hand_line1)
        self.hand_line2= gl.GLLinePlotItem(pos=np.array([[[0, 0.075*4, -0.075*8], [0, 0.075*4, 0.075*8]]]),
                                           color=[128, 255, 128, 255], antialias=False)
        self.view_PD.addItem(self.hand_line2)
        ###---------------------------------------------
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
        self.view_rdi.addItem(img_rdi)
        self.view_rai.addItem(img_rai)
        self.view_rdi.setRange(QtCore.QRectF(0, 0, 30, 70))
        self.view_rai.setRange(QtCore.QRectF(10, 0, 160, 80))
        self.view_PD.pan(0.1,0.1,0.1)
        updateTime = ptime.time()
        self.app.instance().exec_()
        # print('=======================================')


if __name__ == '__main__':
    print('======Real Time Data Capture Tool======')
    count = 0
    realtime = Realtime_sys()
    lock = threading.Lock()

    plotIMAGE = threading.Thread(target=realtime.plot())
    plotIMAGE.start()

    print("Program Close")
    sys.exit()
