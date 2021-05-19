from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys

class MyWindow(QWidget):

     def __init__(self):
      super().__init__()
      self.initUI()
      self.resize(500,500)

     def initUI(self):
      hbox = QHBoxLayout(self)
      left = QFrame(self)
      left.resize(100,100)
      right= QFrame(self)
      splitter1 = QSplitter(Qt.Horizontal)
      splitter1.addWidget(left)
      splitter1.setSizes([30,])  # 設置分隔條位置
      splitter1.addWidget(right)
      hbox.addWidget(splitter1)
      self.setLayout(hbox)

      self.btn1=QPushButton(left)
      self.btn1.setText("測試")
      self.btn1.setGeometry(0,10,100,50)
      self.btn1.clicked.connect(self.btn1sw)

      self.btn2 = QPushButton(left)
      self.btn2.setGeometry(0,80,100,30)
      self.btn2.setText("測試1")
      self.btn2.clicked.connect(self.btn2sw)

      # 設置stackedWidget
      self.stackedWidget = QStackedWidget(right)

      # 設置第一個面板
      self.form1 = QWidget()
      #self.formLayout1 = QHBoxLayout(self.form1)
      '''self.button1=QPushButton()
      self.button1.setText("按鈕1")
      self.label1 = QLabel()
      self.label1.setText("測試")
      self.label1.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
      self.label1.setAlignment(Qt.AlignCenter)
      self.label1.setFont(QFont("Roman times", 50, QFont.Bold))
      self.formLayout1.addWidget(self.label1)
      self.formLayout1.addWidget(self.button1)'''


      self.g_layout =  QGridLayout()
      self.h1_layout = QHBoxLayout()
      self.h2_layout = QHBoxLayout()
      self.h3_layout = QHBoxLayout()
      self.h4_layout = QHBoxLayout()
      self.h5_layout = QHBoxLayout()
      self.h6_layout = QHBoxLayout()

      self.h1_layoutUI()
      self.h2_layoutUI()
      self.h3_layoutUI()
      self.h4_layoutUI()
      self.h5_layoutUI()
      self.h6_layoutUI()

      self.g_layout.addItem(self.h1_layout, 0, 0, 1, 1)
      self.g_layout.addItem(self.h2_layout, 1, 0, 1, 1)
      self.g_layout.addItem(self.h3_layout, 2, 0, 1, 1)
      self.g_layout.addItem(self.h4_layout, 3, 0, 1, 1)
      self.g_layout.addItem(self.h5_layout, 4, 0, 1, 1)
      self.g_layout.addItem(self.h6_layout, 5, 0, 1, 1)
      self.form1.setLayout(self.g_layout)



      # 設置第二個面板
      self.form2 = QWidget()
      self.formLayout2 = QHBoxLayout(self.form2)
      self.button2=QPushButton()
      self.button2.setText("按鈕2")
      self.label2 = QLabel()
      self.label2.setText("測試1")
      self.label2.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
      self.label2.setAlignment(Qt.AlignCenter)
      self.label2.setFont(QFont("Roman times", 50, QFont.Bold))
      self.formLayout2.addWidget(self.label2)
      self.formLayout2.addWidget(self.button2)

      # 將兩個面板，加入stackedWidget
      self.stackedWidget.addWidget(self.form1)
      self.stackedWidget.addWidget(self.form2)

     def btn1sw(self):
      self.stackedWidget.setCurrentIndex(0)
     def btn2sw(self):
      self.stackedWidget.setCurrentIndex(1)

     def h1_layoutUI(self):
      self.label = QLabel()
      # setting geometry to the label
      # creating label multi line
      self.label.setWordWrap(True)
      # setting style sheet to the label
      self.label.setStyleSheet("QLabel"
                               "{"
                               "border : 4px solid black;"
                               "background : white;"
                               "}")
      # setting alignment to the label
      self.label.setAlignment(Qt.AlignRight)
      # setting font
      self.label.setFont(QFont('Arial', 15))

      self.h1_layout.addWidget(self.label)

     def h2_layoutUI(self):
      self.push_clear = QPushButton("Clear")
      self.push_clear.setStyleSheet("QPushButton::pressed"
                                    "{"
                                    "background-color: red"
                                    "}")
      self.push_del = QPushButton("Del")
      self.push_del.setStyleSheet("QPushButton::pressed"
                                  "{"
                                  "background-color: red"
                                  "}")

      self.push_clear.clicked.connect(self.action_clear)
      self.push_del.clicked.connect(self.action_del)

      self.h2_layout.addWidget(self.push_clear)
      self.h2_layout.addWidget(self.push_del)

     def h3_layoutUI(self):
      self.push_1 = QPushButton("1")
      self.push_1.setStyleSheet("QPushButton::pressed"
                                "{"
                                "background-color: red"
                                "}")
      self.push_2 = QPushButton("2")
      self.push_2.setStyleSheet("QPushButton::pressed"
                                "{"
                                "background-color: red"
                                "}")
      self.push_3 = QPushButton("3")
      self.push_3.setStyleSheet("QPushButton::pressed"
                                "{"
                                "background-color: red"
                                "}")
      self.push_mul = QPushButton("*")
      self.push_mul.setStyleSheet("QPushButton::pressed"
                                  "{"
                                  "background-color: red"
                                  "}")

      self.push_1.clicked.connect(self.action1)
      self.push_2.clicked.connect(self.action2)
      self.push_3.clicked.connect(self.action3)
      self.push_mul.clicked.connect(self.action_mul)

      self.h3_layout.addWidget(self.push_1)
      self.h3_layout.addWidget(self.push_2)
      self.h3_layout.addWidget(self.push_3)
      self.h3_layout.addWidget(self.push_mul)

     def h4_layoutUI(self):
      self.push_4 = QPushButton("4")
      self.push_4.setStyleSheet("QPushButton::pressed"
                                "{"
                                "background-color: red"
                                "}")
      self.push_5 = QPushButton("5")
      self.push_5.setStyleSheet("QPushButton::pressed"
                                "{"
                                "background-color: red"
                                "}")
      self.push_6 = QPushButton("6")
      self.push_6.setStyleSheet("QPushButton::pressed"
                                "{"
                                "background-color: red"
                                "}")
      self.push_minus = QPushButton("-")
      self.push_minus.setStyleSheet("QPushButton::pressed"
                                    "{"
                                    "background-color: red"
                                    "}")

      self.push_4.clicked.connect(self.action4)
      self.push_5.clicked.connect(self.action5)
      self.push_6.clicked.connect(self.action6)
      self.push_minus.clicked.connect(self.action_minus)

      self.h4_layout.addWidget(self.push_4)
      self.h4_layout.addWidget(self.push_5)
      self.h4_layout.addWidget(self.push_6)
      self.h4_layout.addWidget(self.push_minus)

     def h5_layoutUI(self):
      self.push_7 = QPushButton("7")
      self.push_7.setStyleSheet("QPushButton::pressed"
                                "{"
                                "background-color: red"
                                "}")
      self.push_8 = QPushButton("8")
      self.push_8.setStyleSheet("QPushButton::pressed"
                                "{"
                                "background-color: red"
                                "}")
      self.push_9 = QPushButton("9")
      self.push_9.setStyleSheet("QPushButton::pressed"
                                "{"
                                "background-color: red"
                                "}")
      self.push_plus = QPushButton("+")
      self.push_plus.setStyleSheet("QPushButton::pressed"
                                   "{"
                                   "background-color: red"
                                   "}")

      self.push_7.clicked.connect(self.action7)
      self.push_8.clicked.connect(self.action8)
      self.push_9.clicked.connect(self.action9)
      self.push_plus.clicked.connect(self.action_plus)

      self.h5_layout.addWidget(self.push_7)
      self.h5_layout.addWidget(self.push_8)
      self.h5_layout.addWidget(self.push_9)
      self.h5_layout.addWidget(self.push_plus)

     def h6_layoutUI(self):
      self.push_0 = QPushButton("0")
      self.push_0.setStyleSheet("QPushButton::pressed"
                                "{"
                                "background-color: red"
                                "}")
      self.push_point = QPushButton(".")
      self.push_point.setStyleSheet("QPushButton::pressed"
                                    "{"
                                    "background-color: red"
                                    "}")
      self.push_div = QPushButton("/")
      self.push_div.setStyleSheet("QPushButton::pressed"
                                  "{"
                                  "background-color: red"
                                  "}")
      self.push_equal = QPushButton("=")
      self.push_equal.setStyleSheet("QPushButton::pressed"
                                    "{"
                                    "background-color: red"
                                    "}")

      self.push_0.clicked.connect(self.action0)
      self.push_point.clicked.connect(self.action_point)
      self.push_div.clicked.connect(self.action_div)
      self.push_equal.clicked.connect(self.action_equal)

      self.h6_layout.addWidget(self.push_0)
      self.h6_layout.addWidget(self.push_point)
      self.h6_layout.addWidget(self.push_div)
      self.h6_layout.addWidget(self.push_equal)

     def action_equal(self):

      # get the label text
      equation = self.label.text()
      try:
       # getting the ans
       ans = eval(equation)

       # setting text to the label
       self.label.setText(str(ans))

      except:
       # setting text to the label
       self.label.setText("Wrong Input")

     def action_plus(self):
      # appending label text
      text = self.label.text()
      self.label.setText(text + " + ")

     def action_minus(self):
      # appending label text
      text = self.label.text()
      self.label.setText(text + " - ")

     def action_div(self):
      # appending label text
      text = self.label.text()
      self.label.setText(text + " / ")

     def action_mul(self):
      # appending label text
      text = self.label.text()
      self.label.setText(text + " * ")

     def action_point(self):
      # appending label text
      text = self.label.text()
      self.label.setText(text + ".")

     def action0(self):
      # appending label text
      text = self.label.text()
      self.label.setText(text + "0")

     def action1(self):
      # appending label text
      text = self.label.text()
      self.label.setText(text + "1")

     def action2(self):
      # appending label text
      text = self.label.text()
      self.label.setText(text + "2")

     def action3(self):
      # appending label text
      text = self.label.text()
      self.label.setText(text + "3")

     def action4(self):
      # appending label text
      text = self.label.text()
      self.label.setText(text + "4")

     def action5(self):
      # appending label text
      text = self.label.text()
      self.label.setText(text + "5")

     def action6(self):
      # appending label text
      text = self.label.text()
      self.label.setText(text + "6")

     def action7(self):
      # appending label text
      text = self.label.text()
      self.label.setText(text + "7")

     def action8(self):
      # appending label text
      text = self.label.text()
      self.label.setText(text + "8")

     def action9(self):
      # appending label text
      text = self.label.text()
      self.label.setText(text + "9")

     def action_clear(self):
      # clearing the label text
      self.label.setText("")

     def action_del(self):
      # clearing a single digit
      text = self.label.text()
      print(text[:len(text) - 1])
      self.label.setText(text[:len(text) - 1])




app = QtWidgets.QApplication(sys.argv)
window = MyWindow() #開始創建視窗
window.show()
sys.exit(app.exec_()) #有這行才會保持