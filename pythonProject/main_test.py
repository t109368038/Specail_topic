from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cal
import music_player
import sys



class MyWindow(QWidget):

     def __init__(self):
      super().__init__()
      self.initUI()
      self.setFixedSize(520,230)

     def initUI(self):
      self.setWindowTitle("車載主機")
      self.setWindowIcon(QIcon('pic/cal.png'))
      self.main_widget=QWidget(self)
      self.main_layout=QHBoxLayout(self)

      self.init_left()
      self.init_right()

      self.stackedWidget=QStackedWidget()
      self.stackedWidget.addWidget(self.right_widget)
      self.stackedWidget.addWidget(self.right_widget1)

      self.main_layout.addWidget(self.left_widget)
      self.main_layout.addWidget(self.stackedWidget)
      self.main_widget.setLayout(self.main_layout)

     def init_left(self):
      self.left_widget=QWidget()
      self.left_widget.setStyleSheet("QWidget"
                                     "{"
                                     "background-color:cyan;"
                                     "border:2px solid;"
                                     "}")
      self.left_layout=QVBoxLayout()

      self.btn1 = QPushButton()
      #self.btn1.setText("測試")
      self.btn1.setFixedSize(50,50)
      self.btn1.clicked.connect(self.btn1sw)
      self.btn1.setStyleSheet("border-image:url(./pic/cal.png)")

      self.btn2 = QPushButton()
      #self.btn2.setText("測試1")
      self.btn2.setFixedSize(50,50)
      self.btn2.clicked.connect(self.btn2sw)
      self.btn2.setStyleSheet("border-image:url(./pic/music.png)")

      self.left_layout.addWidget(self.btn1)
      self.left_layout.addWidget(self.btn2)
      self.left_widget.setLayout(self.left_layout)
      #調整widget固定高度
      self.left_widget.setFixedHeight(220)


     def btn1sw(self):
      self.stackedWidget.setCurrentIndex(0)

     def btn2sw(self):
      self.stackedWidget.setCurrentIndex(1)

     def init_right(self):
      self.right_widget = cal.MyWindow()

      self.right_widget1=music_player.MainWindow()







if __name__ == '__main__':
  app = QtWidgets.QApplication(sys.argv)

  #顯示icon在dock中
  app.setWindowIcon(QIcon('pic/cal.png'))

  window = MyWindow() #開始創建視窗
  window.show()
  sys.exit(app.exec_()) #有這行才會保持