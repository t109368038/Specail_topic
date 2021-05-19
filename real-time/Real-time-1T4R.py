from real_time_process_1t4r import UdpListener, DataProcessor
from radar_config import SerialConfig
from queue import Queue
from scipy import signal
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import numpy as np
import threading
import time
import sys
import socket

# -----------------------------------------------
from app_layout_2t4r import Ui_MainWindow

# -----------------------------------------------
config = '../radar_config/IWR1843_cfg_1t4r.cfg'


# class CA_CFAR():
#     """
#     Description:
#     ------------
#         Cell Averaging - Constant False Alarm Rate algorithm
#         Performs an automatic detection on the input range-Doppler matrix with an adaptive thresholding.
#         The threshold level is determined for each cell in the range-Doppler map with the estimation
#         of the power level of its surrounding noise. The average power of the noise is estimated on a
#         rectangular window, that is defined around the CUT (Cell Under Test). In order the mitigate the effect
#         of the target reflection energy spreading some cells are left out from the calculation in the immediate
#         vicinity of the CUT. These cells are the guard cells.
#         The size of the estimation window and guard window can be set with the win_param parameter.
#     Implementation notes:
#     ---------------------
#         Implementation based on https://github.com/petotamas/APRiL
#     Parameters:
#     -----------
#     :param win_param: Parameters of the noise power estimation window
#                       [Est. window length, Est. window width, Guard window length, Guard window width]
#     :param threshold: Threshold level above the estimated average noise power
#     :type win_param: python list with 4 elements
#     :type threshold: float
#     Return values:
#     --------------
#     """
#
#     def __init__(self, win_param, threshold, rd_size):
#         win_width = win_param[0]
#         win_height = win_param[1]
#         guard_width = win_param[2]
#         guard_height = win_param[3]
#
#         # Create window mask with guard cells
#         self.mask = np.ones((2 * win_height + 1, 2 * win_width + 1), dtype=bool)
#         self.mask[win_height - guard_height:win_height + 1 + guard_height, win_width - guard_width:win_width + 1 + guard_width] = 0
#
#         # Convert threshold value
#         self.threshold = 10 ** (threshold / 10)
#
#         # Number cells within window around CUT; used for averaging operation.
#         self.num_valid_cells_in_window = signal.convolve2d(np.ones(rd_size, dtype=float), self.mask, mode='same')
#
#     def __call__(self, rd_matrix):
#         """
#         Description:
#         ------------
#             Performs the automatic detection on the input range-Doppler matrix.
#         Implementation notes:
#         ---------------------
#         Parameters:
#         -----------
#         :param rd_matrix: Range-Doppler map on which the automatic detection should be performed
#         :type rd_matrix: R x D complex numpy array
#         Return values:
#         --------------
#         :return hit_matrix: Calculated hit matrix
#         """
#         # Convert range-Doppler map values to power
#         rd_matrix = np.abs(rd_matrix) ** 2
#
#         # Perform detection
#         rd_windowed_sum = signal.convolve2d(rd_matrix, self.mask, mode='same')
#         rd_avg_noise_power = rd_windowed_sum / self.num_valid_cells_in_window
#         rd_snr = rd_matrix / rd_avg_noise_power
#         hit_matrix = rd_snr > self.threshold
#
#         return hit_matrix

def send_cmd(code):
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
    data_FPGA_config = (0x01020102031e).to_bytes(6, byteorder='big', signed=False)
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
    print('send command:', re.hex())
    return re


def update_figure():
    global img_rdi, img_rai, updateTime, ang_cuv
    win_param = [8, 8, 3, 3]
    # cfar_rai = CA_CFAR(win_param, threshold=2.5, rd_size=[64, 181])
    img_rdi.setImage(np.abs(RDIData.get()[:, :, 0].T))
    # img_rai.setImage(cfar_rai(np.fliplr(RAIData.get()[0, :, :])).T)

    xx = RAIData.get()[:, :, :].sum(0)
    img_rai.setImage(np.fliplr(np.flip(xx, axis=0)).T)
    # angCurve.plot((np.fliplr(np.flip(xx, axis=0)).T)[:, 10:12].sum(1), clear=True)
    ang_cuv.setData(np.fliplr(np.flip(xx, axis=0)).T[:, 10:12].sum(1), clear=True)
    QtCore.QTimer.singleShot(1, update_figure)
    now = ptime.time()
    updateTime = now


def openradar():
    global tt
    tt = SerialConfig(name='ConnectRadar', CLIPort='COM4', BaudRate=115200)
    tt.StopRadar()
    tt.SendConfig(config)
    update_figure()


def plot():
    global img_rdi, img_rai, updateTime, view_text, count, angCurve, ang_cuv
    # ---------------------------------------------------
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    MainWindow.show()
    tmp_data = np.zeros(181)
    # angCurve = pg.plot(tmp_data, pen='r')
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    view_rdi = ui.graphicsView.addViewBox()
    view_rai = ui.graphicsView_2.addViewBox()
    view_angCurve = ui.graphicsView_3.addViewBox()
    starbtn = ui.pushButton_start
    exitbtn = ui.pushButton_exit
    # ---------------------------------------------------

    # lock the aspect ratio so pixels are always square
    view_rdi.setAspectLocked(True)
    view_rai.setAspectLocked(True)
    img_rdi = pg.ImageItem(border='w')
    img_rai = pg.ImageItem(border='w')
    ang_cuv = pg.PlotDataItem(tmp_data, pen='r')
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
    view_angCurve.addItem(ang_cuv)
    # Set initial view bounds
    view_rdi.setRange(QtCore.QRectF(-5, 0, 140, 80))
    view_rai.setRange(QtCore.QRectF(10, 0, 160, 80))
    updateTime = ptime.time()

    starbtn.clicked.connect(openradar)
    exitbtn.clicked.connect(app.instance().exit)
    app.instance().exec_()
    tt.StopRadar()


# Queue for access data
BinData = Queue()
RDIData = Queue()
RAIData = Queue()
# Radar config
adc_sample = 64
chirp = 32
tx_num = 1
rx_num = 4
radar_config = [adc_sample, chirp, tx_num, rx_num]
frame_length = adc_sample * chirp * tx_num * rx_num * 2
# Host setting
address = ('192.168.33.30', 4098)
buff_size = 2097152

# config DCA1000 to receive bin data
config_address = ('192.168.33.30', 4096)
FPGA_address_cfg = ('192.168.33.180', 4096)
cmd_order = ['9', 'E', '3', 'B', '5', '6']
sockConfig = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sockConfig.bind(config_address)

for k in range(5):
    # Send the command
    sockConfig.sendto(send_cmd(cmd_order[k]), FPGA_address_cfg)
    time.sleep(0.1)
    # Request data back on the config port
    msg, server = sockConfig.recvfrom(2048)
    print('receive command:', msg.hex())

collector = UdpListener('Listener', BinData, frame_length, address, buff_size)
processor = DataProcessor('Processor', radar_config, BinData, RDIData, RAIData, "1130")
collector.start()
processor.start()
plotIMAGE = threading.Thread(target=plot())
plotIMAGE.start()

sockConfig.sendto(send_cmd('6'), FPGA_address_cfg)
sockConfig.close()
collector.join(timeout=1)
processor.join(timeout=1)

print("Program close")
sys.exit()
