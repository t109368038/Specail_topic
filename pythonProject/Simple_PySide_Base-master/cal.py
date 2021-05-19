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
        self.resize(500, 500)

    def initUI(self):
        self.v_layout=QVBoxLayout()
        self.h1_layout=QHBoxLayout()
        self.h2_layout=QHBoxLayout()
        self.h3_layout=QHBoxLayout()
        self.h4_layout=QHBoxLayout()
        self.h5_layout=QHBoxLayout()
        self.h6_layout=QHBoxLayout()

        self.h1_layoutUI()
        self.h2_layoutUI()
        self.h3_layoutUI()
        self.h4_layoutUI()
        self.h5_layoutUI()
        self.h6_layoutUI()

        self.v_layout.addItem(self.h1_layout)
        self.v_layout.addItem(self.h2_layout)
        self.v_layout.addItem(self.h3_layout)
        self.v_layout.addItem(self.h4_layout)
        self.v_layout.addItem(self.h5_layout)
        self.v_layout.addItem(self.h6_layout)
        self.setLayout(self.v_layout)

    def h1_layoutUI(self):
        self.label=QLabel()
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

        self.label.setFixedHeight(200)

        self.h1_layout.addWidget(self.label)

    def h2_layoutUI(self):
        self.push_clear=QPushButton("Clear")
        self.push_clear.setStyleSheet("QPushButton::pressed"
                                      "{"
                                      "background-color: red"
                                      "}")
        self.push_del=QPushButton("Del")
        self.push_del.setStyleSheet("QPushButton::pressed"
                                      "{"
                                      "background-color: red"
                                      "}")

        self.push_clear.clicked.connect(self.action_clear)
        self.push_del.clicked.connect(self.action_del)

        self.h2_layout.addWidget(self.push_clear)
        self.h2_layout.addWidget(self.push_del)


    def h3_layoutUI(self):
        self.push_1=QPushButton("1")
        self.push_1.setStyleSheet("QPushButton::pressed"
                                      "{"
                                      "background-color: red"
                                      "}")
        self.push_2=QPushButton("2")
        self.push_2.setStyleSheet("QPushButton::pressed"
                                      "{"
                                      "background-color: red"
                                      "}")
        self.push_3=QPushButton("3")
        self.push_3.setStyleSheet("QPushButton::pressed"
                                      "{"
                                      "background-color: red"
                                      "}")
        self.push_mul=QPushButton("*")
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
        self.push_4=QPushButton("4")
        self.push_4.setStyleSheet("QPushButton::pressed"
                                      "{"
                                      "background-color: red"
                                      "}")
        self.push_5=QPushButton("5")
        self.push_5.setStyleSheet("QPushButton::pressed"
                                      "{"
                                      "background-color: red"
                                      "}")
        self.push_6=QPushButton("6")
        self.push_6.setStyleSheet("QPushButton::pressed"
                                      "{"
                                      "background-color: red"
                                      "}")
        self.push_minus=QPushButton("-")
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
        self.push_7=QPushButton("7")
        self.push_7.setStyleSheet("QPushButton::pressed"
                                      "{"
                                      "background-color: red"
                                      "}")
        self.push_8=QPushButton("8")
        self.push_8.setStyleSheet("QPushButton::pressed"
                                      "{"
                                      "background-color: red"
                                      "}")
        self.push_9=QPushButton("9")
        self.push_9.setStyleSheet("QPushButton::pressed"
                                      "{"
                                      "background-color: red"
                                      "}")
        self.push_plus=QPushButton("+")
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
        self.push_0=QPushButton("0")
        self.push_0.setStyleSheet("QPushButton::pressed"
                                      "{"
                                      "background-color: red"
                                      "}")
        self.push_point=QPushButton(".")
        self.push_point.setStyleSheet("QPushButton::pressed"
                                      "{"
                                      "background-color: red"
                                      "}")
        self.push_div=QPushButton("/")
        self.push_div.setStyleSheet("QPushButton::pressed"
                                      "{"
                                      "background-color: red"
                                      "}")
        self.push_equal=QPushButton("=")
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

    def call_layout(self):
        return self.g_layout


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow() #開始創建視窗
    window.show()
    sys.exit(app.exec_())




