import sys
import numpy as np
import DSP
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
from PyQt5.QtWidgets import QDialog, QApplication
from app_layout import Ui_MainWindow
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow


class Main(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)


folder = 'D:/pycharm_project/real-time-radar/data/small_object_data/'
filename = 'minus30.npy'
data = np.load(folder + filename)
rdi = DSP.Range_Doppler(data, 1, [128, 64])
frame = rdi[:, :, 0]


if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = Main()
    MainWindow.show()
    sys.exit(app.exec_())
