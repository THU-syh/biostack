# -*- coding: utf-8 -*-
"""
    Title   : Implementation of GUI
    @author : Sachin Nandakumar
"""

'''
    Form implementation generated from reading ui file 'trial-one.ui'

    Created by: PyQt5 UI code generator 5.11.3

    WARNING! All changes made in this file will be lost!
'''
 
import os
import sys
import time
import glob
import pandas as pd
import numpy as np
from statistics import mean
from PyQt5 import QtCore, QtGui, QtWidgets
from model_reliance import Model_Reliance
from model_info import Model_Info
from create_pdf import Create_PDF
from param_range import Parameter_Range
from generate_ice import Generate_ICE
from counter_factuals import Counterfactuals
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import warnings
import operator
warnings.filterwarnings("ignore")

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans']



class Worker(QtCore.QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and 
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args

    @QtCore.pyqtSlot()
    def run(self):
        print('inside run')
        '''
        Multi-threading
        
        Initialise the runner function with passed args, kwargs.
        '''
        print('Thread Started')
        self.fn(*self.args)
        print('Thread Finished')


class Ui_MainWindow(object):
    
    def __init__(self):
        self.model = Model_Reliance()
        self.param_range = Parameter_Range()
        self.generate_ice = Generate_ICE()
        self.counterfactuals = Counterfactuals()
        self.data = self.model.model_reliance('gbt')
        self.threadpool = QtCore.QThreadPool()
        self.threadpool.setMaxThreadCount(2)
        
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
        files = glob.glob('img/*')
        for f in files:
            if not 'counterfactual_gbt_52' in f:
                os.remove(f)
    
    def setupUi(self, MainWindow):
        '''
        
            This function creates the whole layout of the GUI.
            Format is as generated using pyuic.py from pyqt designer. 
            Modifications are then done based on requirements.
        
        '''
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1126, 834)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setStyleSheet("QTabWidget::pane { /* The tab widget frame */\n"
        "    border-top: 2px solid #C2C7CB;\n"
        "    position: absolute;\n"
        "    top: -0.5em;\n"
        "}\n"
        "\n"
        "QTabWidget::tab-bar {\n"
        "    alignment: center;\n"
        "}\n"
        "\n"
        "/* Style the tab using the tab sub-control. Note that\n"
        "    it reads QTabBar _not_ QTabWidget */\n"
        "QTabBar::tab {\n"
        "    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
        "                                stop: 0 #E1E1E1, stop: 0.4 #DDDDDD,\n"
        "                                stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3);\n"
        "    border: 2px solid #C4C4C3;\n"
        "    border-bottom-color: #C2C7CB; /* same as the pane color */\n"
        "    border-top-left-radius: 4px;\n"
        "    border-top-right-radius: 4px;\n"
        "    min-width: 8ex;\n"
        "    padding: 2px;\n"
        "}\n"
        "\n"
        "QTabBar::tab:selected, QTabBar::tab:hover {\n"
        "    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
        "                                stop: 0 #fafafa, stop: 0.4 #f4f4f4,\n"
        "                                stop: 0.5 #e7e7e7, stop: 1.0 #fafafa);\n"
        "}\n"
        "\n"
        "QTabBar::tab:selected {\n"
        "    border-color: #9B9B9B;\n"
        "    border-bottom-color: #C2C7CB; /* same as pane color */\n"
        "}")
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar1 = NavigationToolbar(self.canvas, self.canvas)
        self.figure1 = plt.figure()
        self.canvas1 = FigureCanvas(self.figure1)
        self.toolbar2 = NavigationToolbar(self.canvas1, self.canvas1)
        self.figure2 = plt.figure()
        self.canvas2 = FigureCanvas(self.figure2)
        self.toolbar3 = NavigationToolbar(self.canvas2, self.canvas2)
        
        
        self.tab.setObjectName("tab")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.tab)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 201, 31))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.comboBox = QtWidgets.QComboBox(self.verticalLayoutWidget)
        self.comboBox.setStyleSheet("QComboBox {\n"
        "    border: 1px solid gray;\n"
        "    border-radius: 3px;\n"
        "    padding: 1px 18px 1px 3px;\n"
        "    min-width: 6em;\n"
        "}\n"
        "\n"
        "QComboBox:editable {\n"
        "    background: white;\n"
        "}\n"
        "\n"
        "QComboBox:!editable, QComboBox::drop-down:editable {\n"
        "     background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
        "                                 stop: 0 #E1E1E1, stop: 0.4 #DDDDDD,\n"
        "                                 stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3);\n"
        "}\n"
        "\n"
        "/* QComboBox gets the \"on\" state when the popup is open */\n"
        "QComboBox:!editable:on, QComboBox::drop-down:editable:on {\n"
        "    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
        "                                stop: 0 #D3D3D3, stop: 0.4 #D8D8D8,\n"
        "                                stop: 0.5 #DDDDDD, stop: 1.0 #E1E1E1);\n"
        "}\n"
        "\n"
        "QComboBox:on { /* shift the text when the popup opens */\n"
        "    padding-top: 3px;\n"
        "    padding-left: 4px;\n"
        "}\n"
        "\n"
        "QComboBox::drop-down {\n"
        "    subcontrol-origin: padding;\n"
        "    subcontrol-position: top right;\n"
        "    width: 15px;\n"
        "\n"
        "    border-left-width: 1px;\n"
        "    border-left-color: darkgray;\n"
        "    border-left-style: solid; /* just a single line */\n"
        "    border-top-right-radius: 3px; /* same radius as the QComboBox */\n"
        "    border-bottom-right-radius: 3px;\n"
        "}\n"
        "\n"
        "QComboBox::down-arrow {\n"
        "    image: url(/usr/share/icons/crystalsvg/16x16/actions/1downarrow.png);\n"
        "}\n"
        "\n"
        "QComboBox::down-arrow:on { /* shift the arrow when popup is open */\n"
        "    top: 1px;\n"
        "    left: 1px;\n"
        "}")
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.verticalLayout_2.addWidget(self.comboBox)
        
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.tab)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(10, 50, 201, 671))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.textEdit = QtWidgets.QTextEdit(self.verticalLayoutWidget_3)
        self.textEdit.setStyleSheet("background-color: #e5e5e5;\n"
        "color: #000;\n"
        "border-radius: 5px;\n"
        "border-style: outset;\n"
        "height: 25px;")
        self.textEdit.setObjectName("textEdit")
        self.textEdit.setReadOnly(True)
        self.verticalLayout_4.addWidget(self.textEdit)
        self.verticalLayoutWidget_5 = QtWidgets.QWidget(self.tab)
        self.verticalLayoutWidget_5.setGeometry(QtCore.QRect(220, 10, 881, 711))
        self.verticalLayoutWidget_5.setObjectName("verticalLayoutWidget_5")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_5)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.verticalLayout_6.addWidget(self.toolbar1)
        self.verticalLayout_6.addWidget(self.canvas)
        self.tabWidget.addTab(self.tab, "")
        
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.tab_2)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(10, 10, 201, 31))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.comboBox_2 = QtWidgets.QComboBox(self.verticalLayoutWidget_2)
        self.comboBox_2.setStyleSheet("QComboBox {\n"
        "    border: 1px solid gray;\n"
        "    border-radius: 3px;\n"
        "    padding: 1px 18px 1px 3px;\n"
        "    min-width: 6em;\n"
        "}\n"
        "\n"
        "QComboBox:editable {\n"
        "    background: white;\n"
        "}\n"
        "\n"
        "QComboBox:!editable, QComboBox::drop-down:editable {\n"
        "     background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
        "                                 stop: 0 #E1E1E1, stop: 0.4 #DDDDDD,\n"
        "                                 stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3);\n"
        "}\n"
        "\n"
        "/* QComboBox gets the \"on\" state when the popup is open */\n"
        "QComboBox:!editable:on, QComboBox::drop-down:editable:on {\n"
        "    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
        "                                stop: 0 #D3D3D3, stop: 0.4 #D8D8D8,\n"
        "                                stop: 0.5 #DDDDDD, stop: 1.0 #E1E1E1);\n"
        "}\n"
        "\n"
        "QComboBox:on { /* shift the text when the popup opens */\n"
        "    padding-top: 3px;\n"
        "    padding-left: 4px;\n"
        "}\n"
        "\n"
        "QComboBox::drop-down {\n"
        "    subcontrol-origin: padding;\n"
        "    subcontrol-position: top right;\n"
        "    width: 15px;\n"
        "\n"
        "    border-left-width: 1px;\n"
        "    border-left-color: darkgray;\n"
        "    border-left-style: solid; /* just a single line */\n"
        "    border-top-right-radius: 3px; /* same radius as the QComboBox */\n"
        "    border-bottom-right-radius: 3px;\n"
        "}\n"
        "\n"
        "QComboBox::down-arrow {\n"
        "    image: url(/usr/share/icons/crystalsvg/16x16/actions/1downarrow.png);\n"
        "}\n"
        "\n"
        "QComboBox::down-arrow:on { /* shift the arrow when popup is open */\n"
        "    top: 1px;\n"
        "    left: 1px;\n"
        "}")
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.verticalLayout_3.addWidget(self.comboBox_2)
        self.verticalLayoutWidget_4 = QtWidgets.QWidget(self.tab_2)
        self.verticalLayoutWidget_4.setGeometry(QtCore.QRect(10, 50, 201, 671))
        self.verticalLayoutWidget_4.setObjectName("verticalLayoutWidget_4")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_4)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.textEdit_2 = QtWidgets.QTextEdit(self.verticalLayoutWidget_4)
        self.textEdit_2.setStyleSheet("background-color: #e5e5e5;\n"
        "color: #000;\n"
        "border-radius: 5px;\n"
        "border-style: outset;\n"
        "height: 25px;")
        self.textEdit_2.setObjectName("textEdit_2")
        self.textEdit_2.setReadOnly(True)
        self.verticalLayout_5.addWidget(self.textEdit_2)
        self.verticalLayoutWidget_9 = QtWidgets.QWidget(self.tab_2)
        self.verticalLayoutWidget_9.setGeometry(QtCore.QRect(220, 50, 881, 671))
        self.verticalLayoutWidget_9.setObjectName("verticalLayoutWidget_9")
        self.horizontalLayoutWidget_4 = QtWidgets.QWidget(self.tab_2)
        self.horizontalLayoutWidget_4.setGeometry(QtCore.QRect(240, 10, 91, 31))
        self.horizontalLayoutWidget_4.setObjectName("horizontalLayoutWidget_4")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_4)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label = QtWidgets.QLabel(self.horizontalLayoutWidget_4)
        self.label.setObjectName("label")
        self.horizontalLayout_4.addWidget(self.label)
        self.horizontalLayoutWidget_7 = QtWidgets.QWidget(self.tab_2)
        self.horizontalLayoutWidget_7.setGeometry(QtCore.QRect(340, 10, 121, 31))
        self.horizontalLayoutWidget_7.setObjectName("horizontalLayoutWidget_7")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_7)
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.comboBox_5 = QtWidgets.QComboBox(self.horizontalLayoutWidget_7)
        self.comboBox_5.setStyleSheet("QComboBox {\n"
        "    border: 1px solid gray;\n"
        "    border-radius: 3px;\n"
        "    padding: 1px 18px 1px 3px;\n"
        "    min-width: 6em;\n"
        "}\n"
        "\n"
        "QComboBox:editable {\n"
        "    background: white;\n"
        "}\n"
        "\n"
        "QComboBox:!editable, QComboBox::drop-down:editable {\n"
        "     background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
        "                                 stop: 0 #E1E1E1, stop: 0.4 #DDDDDD,\n"
        "                                 stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3);\n"
        "}\n"
        "\n"
        "/* QComboBox gets the \"on\" state when the popup is open */\n"
        "QComboBox:!editable:on, QComboBox::drop-down:editable:on {\n"
        "    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
        "                                stop: 0 #D3D3D3, stop: 0.4 #D8D8D8,\n"
        "                                stop: 0.5 #DDDDDD, stop: 1.0 #E1E1E1);\n"
        "}\n"
        "\n"
        "QComboBox:on { /* shift the text when the popup opens */\n"
        "    padding-top: 3px;\n"
        "    padding-left: 4px;\n"
        "}\n"
        "\n"
        "QComboBox::drop-down {\n"
        "    subcontrol-origin: padding;\n"
        "    subcontrol-position: top right;\n"
        "    width: 15px;\n"
        "\n"
        "    border-left-width: 1px;\n"
        "    border-left-color: darkgray;\n"
        "    border-left-style: solid; /* just a single line */\n"
        "    border-top-right-radius: 3px; /* same radius as the QComboBox */\n"
        "    border-bottom-right-radius: 3px;\n"
        "}\n"
        "\n"
        "QComboBox::down-arrow {\n"
        "    image: url(/usr/share/icons/crystalsvg/16x16/actions/1downarrow.png);\n"
        "}\n"
        "\n"
        "QComboBox::down-arrow:on { /* shift the arrow when popup is open */\n"
        "    top: 1px;\n"
        "    left: 1px;\n"
        "}")
        self.comboBox_5.setObjectName("comboBox_5")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.horizontalLayout_7.addWidget(self.comboBox_5)
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.tab_2)
        self.plainTextEdit.setGeometry(QtCore.QRect(850, 40, 241, 41))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.plainTextEdit.setFont(font)
        self.plainTextEdit.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.plainTextEdit.setStyleSheet("background-color: #e5e5e5;\n"
        "color: #000;\n"
        "border-radius: 5px;\n"
        "border-style: outset;\n"
        "height: 25px;")
        self.plainTextEdit.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.plainTextEdit.setFrameShadow(QtWidgets.QFrame.Raised)
        self.plainTextEdit.setLineWidth(0)
        self.plainTextEdit.setReadOnly(True)
        self.plainTextEdit.setBackgroundVisible(False)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.verticalLayoutWidget_9 = QtWidgets.QWidget(self.tab_2)
        self.verticalLayoutWidget_9.setGeometry(QtCore.QRect(220, 50, 881, 671))
        self.verticalLayoutWidget_9.setObjectName("verticalLayoutWidget_9")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_9)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.verticalLayout_7.addWidget(self.toolbar3)
        self.verticalLayout_7.addWidget(self.canvas2)
        self.radioButton_2 = QtWidgets.QRadioButton(self.tab_2)
        self.radioButton_2.setGeometry(QtCore.QRect(510, 20, 82, 17))
        self.radioButton_2.setObjectName("radioButton_2")
        self.radioButton_4 = QtWidgets.QRadioButton(self.tab_2)
        self.radioButton_4.setGeometry(QtCore.QRect(610, 20, 82, 17))
        self.radioButton_4.setObjectName("radioButton_4")
        self.horizontalLayoutWidget_9 = QtWidgets.QWidget(self.tab_2)
        self.horizontalLayoutWidget_9.setGeometry(QtCore.QRect(730, 10, 91, 31))
        self.horizontalLayoutWidget_9.setObjectName("horizontalLayoutWidget_9")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_9)
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.pushButton_3 = QtWidgets.QPushButton(self.horizontalLayoutWidget_9)
        self.pushButton_3.setStyleSheet("background-color: #64a3e3;\n"
        "color: #ffffff;\n"
        "border-radius: 5px;\n"
        "border-style: none;\n"
        "height: 25px;")
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout_9.addWidget(self.pushButton_3)
        self.tabWidget.addTab(self.tab_2, "")
        
        
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.verticalLayoutWidget_6 = QtWidgets.QWidget(self.tab_3)
        self.verticalLayoutWidget_6.setGeometry(QtCore.QRect(10, 10, 201, 31))
        self.verticalLayoutWidget_6.setObjectName("verticalLayoutWidget_6")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_6)
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.comboBox_3 = QtWidgets.QComboBox(self.verticalLayoutWidget_6)
        self.comboBox_3.setStyleSheet("QComboBox {\n"
        "    border: 1px solid gray;\n"
        "    border-radius: 3px;\n"
        "    padding: 1px 18px 1px 3px;\n"
        "    min-width: 6em;\n"
        "}\n"
        "\n"
        "QComboBox:editable {\n"
        "    background: white;\n"
        "}\n"
        "\n"
        "QComboBox:!editable, QComboBox::drop-down:editable {\n"
        "     background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
        "                                 stop: 0 #E1E1E1, stop: 0.4 #DDDDDD,\n"
        "                                 stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3);\n"
        "}\n"
        "\n"
        "/* QComboBox gets the \"on\" state when the popup is open */\n"
        "QComboBox:!editable:on, QComboBox::drop-down:editable:on {\n"
        "    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
        "                                stop: 0 #D3D3D3, stop: 0.4 #D8D8D8,\n"
        "                                stop: 0.5 #DDDDDD, stop: 1.0 #E1E1E1);\n"
        "}\n"
        "\n"
        "QComboBox:on { /* shift the text when the popup opens */\n"
        "    padding-top: 3px;\n"
        "    padding-left: 4px;\n"
        "}\n"
        "\n"
        "QComboBox::drop-down {\n"
        "    subcontrol-origin: padding;\n"
        "    subcontrol-position: top right;\n"
        "    width: 15px;\n"
        "\n"
        "    border-left-width: 1px;\n"
        "    border-left-color: darkgray;\n"
        "    border-left-style: solid; /* just a single line */\n"
        "    border-top-right-radius: 3px; /* same radius as the QComboBox */\n"
        "    border-bottom-right-radius: 3px;\n"
        "}\n"
        "\n"
        "QComboBox::down-arrow {\n"
        "    image: url(/usr/share/icons/crystalsvg/16x16/actions/1downarrow.png);\n"
        "}\n"
        "\n"
        "QComboBox::down-arrow:on { /* shift the arrow when popup is open */\n"
        "    top: 1px;\n"
        "    left: 1px;\n"
        "}")
        
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.verticalLayout_8.addWidget(self.comboBox_3)
        self.textEdit_3 = QtWidgets.QTextEdit(self.tab_3)
        self.textEdit_3.setGeometry(QtCore.QRect(10, 50, 199, 669))
        self.textEdit_3.setStyleSheet("background-color: #e5e5e5;\n"
        "color: #000;\n"
        "border-radius: 5px;\n"
        "border-style: outset;\n"
        "height: 25px;")
        self.textEdit_3.setObjectName("textEdit_3")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.tab_3)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(390, 10, 101, 31))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lineEdit = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.lineEdit.setStyleSheet("background-color: #fff;\n"
        "   color: #000;\n"
        "   font-style: italic;\n"
        "   font-weight: bold;")
        self.lineEdit.setInputMask("")
        self.lineEdit.setText("")
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout.addWidget(self.lineEdit)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.tab_3)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(500, 10, 81, 31))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.pushButton.setStyleSheet("background-color: #64a3e3;\n"
        "color: #ffffff;\n"
        "border-radius: 5px;\n"
        "border-style: none;\n"
        "height: 25px;")
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.pushButton)
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.tab_3)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(820, 10, 121, 31))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.comboBox_6 = QtWidgets.QComboBox(self.horizontalLayoutWidget_3)
        self.comboBox_6.setStyleSheet("QComboBox {\n"
        "    border: 1px solid gray;\n"
        "    border-radius: 3px;\n"
        "    padding: 1px 18px 1px 3px;\n"
        "    min-width: 6em;\n"
        "}\n"
        "\n"
        "QComboBox:editable {\n"
        "    background: white;\n"
        "}\n"
        "\n"
        "QComboBox:!editable, QComboBox::drop-down:editable {\n"
        "     background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
        "                                 stop: 0 #E1E1E1, stop: 0.4 #DDDDDD,\n"
        "                                 stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3);\n"
        "}\n"
        "\n"
        "/* QComboBox gets the \"on\" state when the popup is open */\n"
        "QComboBox:!editable:on, QComboBox::drop-down:editable:on {\n"
        "    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
        "                                stop: 0 #D3D3D3, stop: 0.4 #D8D8D8,\n"
        "                                stop: 0.5 #DDDDDD, stop: 1.0 #E1E1E1);\n"
        "}\n"
        "\n"
        "QComboBox:on { /* shift the text when the popup opens */\n"
        "    padding-top: 3px;\n"
        "    padding-left: 4px;\n"
        "}\n"
        "\n"
        "QComboBox::drop-down {\n"
        "    subcontrol-origin: padding;\n"
        "    subcontrol-position: top right;\n"
        "    width: 15px;\n"
        "\n"
        "    border-left-width: 1px;\n"
        "    border-left-color: darkgray;\n"
        "    border-left-style: solid; /* just a single line */\n"
        "    border-top-right-radius: 3px; /* same radius as the QComboBox */\n"
        "    border-bottom-right-radius: 3px;\n"
        "}\n"
        "\n"
        "QComboBox::down-arrow {\n"
        "    image: url(/usr/share/icons/crystalsvg/16x16/actions/1downarrow.png);\n"
        "}\n"
        "\n"
        "QComboBox::down-arrow:on { /* shift the arrow when popup is open */\n"
        "    top: 1px;\n"
        "    left: 1px;\n"
        "}")
        self.comboBox_6.setObjectName("comboBox_6")
        self.horizontalLayout_3.addWidget(self.comboBox_6)
        self.verticalLayoutWidget_10 = QtWidgets.QWidget(self.tab_3)
        self.verticalLayoutWidget_10.setGeometry(QtCore.QRect(220, 70, 881, 651))
        self.verticalLayoutWidget_10.setObjectName("verticalLayoutWidget_10")
        
        self.horizontalLayoutWidget_8 = QtWidgets.QWidget(self.tab_3)
        self.horizontalLayoutWidget_8.setGeometry(QtCore.QRect(730, 10, 121, 31))
        self.horizontalLayoutWidget_8.setObjectName("horizontalLayoutWidget_8")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_8)
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_3 = QtWidgets.QLabel(self.horizontalLayoutWidget_8)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_8.addWidget(self.label_3)
       
        self.horizontalLayoutWidget_10 = QtWidgets.QWidget(self.tab_3)
        self.horizontalLayoutWidget_10.setGeometry(QtCore.QRect(220, 10, 161, 31))
        self.horizontalLayoutWidget_10.setObjectName("horizontalLayoutWidget_10")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_10)
        self.horizontalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_4 = QtWidgets.QLabel(self.horizontalLayoutWidget_10)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_10.addWidget(self.label_4)
        
        self.label_5 = QtWidgets.QLabel(self.tab_3)
        self.label_5.setGeometry(QtCore.QRect(226, 62, 861, 651))
        self.label_5.setText("")
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.tab_3)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(1000, 10, 91, 31))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_2 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_2.setStyleSheet("background-color: #64a3e3;\n"
        "color: #ffffff;\n"
        "border-radius: 5px;\n"
        "border-style: none;\n"
        "height: 25px;")
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2)
        self.tabWidget.addTab(self.tab_3, "")
        
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.textEdit_4 = QtWidgets.QTextEdit(self.tab_4)
        self.textEdit_4.setGeometry(QtCore.QRect(10, 50, 199, 341))
        self.textEdit_4.setStyleSheet("background-color: #e5e5e5;\n"
        "color: #000;\n"
        "border-radius: 5px;\n"
        "border-style: outset;\n"
        "height: 25px;")
        self.textEdit_4.setObjectName("textEdit_4")
        self.textEdit_4.setReadOnly(True)
        self.verticalLayoutWidget_7 = QtWidgets.QWidget(self.tab_4)
        self.verticalLayoutWidget_7.setGeometry(QtCore.QRect(10, 10, 201, 31))
        self.verticalLayoutWidget_7.setObjectName("verticalLayoutWidget_7")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_7)
        self.verticalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.comboBox_4 = QtWidgets.QComboBox(self.verticalLayoutWidget_7)
        self.comboBox_4.setStyleSheet("QComboBox {\n"
        "    border: 1px solid gray;\n"
        "    border-radius: 3px;\n"
        "    padding: 1px 18px 1px 3px;\n"
        "    min-width: 6em;\n"
        "}\n"
        "\n"
        "QComboBox:editable {\n"
        "    background: white;\n"
        "}\n"
        "\n"
        "QComboBox:!editable, QComboBox::drop-down:editable {\n"
        "     background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
        "                                 stop: 0 #E1E1E1, stop: 0.4 #DDDDDD,\n"
        "                                 stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3);\n"
        "}\n"
        "\n"
        "/* QComboBox gets the \"on\" state when the popup is open */\n"
        "QComboBox:!editable:on, QComboBox::drop-down:editable:on {\n"
        "    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
        "                                stop: 0 #D3D3D3, stop: 0.4 #D8D8D8,\n"
        "                                stop: 0.5 #DDDDDD, stop: 1.0 #E1E1E1);\n"
        "}\n"
        "\n"
        "QComboBox:on { /* shift the text when the popup opens */\n"
        "    padding-top: 3px;\n"
        "    padding-left: 4px;\n"
        "}\n"
        "\n"
        "QComboBox::drop-down {\n"
        "    subcontrol-origin: padding;\n"
        "    subcontrol-position: top right;\n"
        "    width: 15px;\n"
        "\n"
        "    border-left-width: 1px;\n"
        "    border-left-color: darkgray;\n"
        "    border-left-style: solid; /* just a single line */\n"
        "    border-top-right-radius: 3px; /* same radius as the QComboBox */\n"
        "    border-bottom-right-radius: 3px;\n"
        "}\n"
        "\n"
        "QComboBox::down-arrow {\n"
        "    image: url(/usr/share/icons/crystalsvg/16x16/actions/1downarrow.png);\n"
        "}\n"
        "\n"
        "QComboBox::down-arrow:on { /* shift the arrow when popup is open */\n"
        "    top: 1px;\n"
        "    left: 1px;\n"
        "}")
        self.comboBox_4.setObjectName("comboBox_4")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.verticalLayout_9.addWidget(self.comboBox_4)

        self.horizontalLayoutWidget_5 = QtWidgets.QWidget(self.tab_4)
        self.horizontalLayoutWidget_5.setGeometry(QtCore.QRect(770, 10, 121, 31))
        self.horizontalLayoutWidget_5.setObjectName("horizontalLayoutWidget_5")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_5)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.radioButton = QtWidgets.QRadioButton(self.horizontalLayoutWidget_5)
        self.radioButton.setObjectName("radioButton")
        self.horizontalLayout_5.addWidget(self.radioButton)
        self.horizontalLayoutWidget_6 = QtWidgets.QWidget(self.tab_4)
        self.horizontalLayoutWidget_6.setGeometry(QtCore.QRect(920, 10, 121, 31))
        self.horizontalLayoutWidget_6.setObjectName("horizontalLayoutWidget_6")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_6)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.radioButton_3 = QtWidgets.QRadioButton(self.horizontalLayoutWidget_6)
        self.radioButton_3.setObjectName("radioButton_3")
        self.horizontalLayout_6.addWidget(self.radioButton_3)
        
        self.verticalLayoutWidget_8 = QtWidgets.QWidget(self.tab_4)
        self.verticalLayoutWidget_8.setGeometry(QtCore.QRect(220, 50, 881, 661))
        self.verticalLayoutWidget_8.setObjectName("verticalLayoutWidget_8")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_8)
        self.verticalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.verticalLayout_10.addWidget(self.toolbar2)
        self.verticalLayout_10.addWidget(self.canvas1)
        self.textEdit_5 = QtWidgets.QTextEdit(self.tab_4)
        self.textEdit_5.setGeometry(QtCore.QRect(10, 400, 199, 311))
        self.textEdit_5.setStyleSheet("background-color: #e5e5e5;\n"
        "color: #000;\n"
        "border-radius: 5px;\n"
        "border-style: outset;\n"
        "height: 25px;")
        self.textEdit_5.setObjectName("textEdit_5")
        self.textEdit_5.setReadOnly(True)
        
        self.tabWidget.addTab(self.tab_4, "")
        self.verticalLayout.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1126, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        self.toolBar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionDownload = QtWidgets.QAction(MainWindow)
        self.actionDownload.setObjectName("actionDownload")
#        self.actionDownload.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.actionDownload.setIcon(QtGui.QIcon("commons/download1.png"))
        self.actionHelp = QtWidgets.QAction(MainWindow)
        self.actionHelp.setObjectName("actionHelp")
        self.actionHelp.setIcon(QtGui.QIcon("commons/help.png"))
        self.toolBar.addAction(self.actionDownload)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionHelp)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "G-MARC"))
        self.comboBox.setItemText(0, _translate("MainWindow", "GBT"))
        self.comboBox.setItemText(1, _translate("MainWindow", "SVM"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Global Feature Importance"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "GBT"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "SVM"))
        self.label.setText(_translate("MainWindow", "Choose parameter"))
        
        sorted_data = sorted(self.data.items(), key=operator.itemgetter(1))
        cat_features = ["side","sex_f","sex_m","type_BF","type_SW"]
        param_count = 0
        for param in reversed(sorted_data):
            if param[0] not in cat_features:
                self.comboBox_5.setItemText(param_count, _translate("MainWindow", param[0]))
                param_count +=1
        
        self.radioButton_2.setText(_translate("MainWindow", "Normal ICE"))
        self.radioButton_2.setChecked(True)
        self.radioButton_4.setText(_translate("MainWindow", "Centered ICE"))
        self.pushButton_3.setText(_translate("MainWindow", "Refresh"))
        
        self.plainTextEdit.setPlainText(_translate("MainWindow", "*Cannot perform ICE plot for categorical features"))
        self.plainTextEdit.setReadOnly(True)
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "ICE plots for feature relationship"))
        
        self.comboBox_3.setItemText(0, _translate("MainWindow", "GBT"))
        self.comboBox_3.setItemText(1, _translate("MainWindow", "SVM"))
        self.label_4.setText(_translate("MainWindow", "Enter number in range [0-100]"))
        self.lineEdit.setPlaceholderText(_translate("MainWindow", "enter number in range [0 - 100]"))
        self.lineEdit.setText(_translate("MainWindow", "52"))
        self.pushButton.setText(_translate("MainWindow", "Submit"))
        self.pushButton_2.setText(_translate("MainWindow", "Refresh"))
        self.label_3.setText(_translate("MainWindow", "Choose Features"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Density plots for counterfactuals"))
        self.comboBox_4.setItemText(0, _translate("MainWindow", "GBT"))
        self.comboBox_4.setItemText(1, _translate("MainWindow", "SVM"))
        self.radioButton.setText(_translate("MainWindow", "Ruptured"))
        self.radioButton.setChecked(True)
        self.radioButton_3.setText(_translate("MainWindow", "Unruptured"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "RIPPER method for parameter value range"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionDownload.setText(_translate("MainWindow", "Download Report"))
        self.actionHelp.setText(_translate("MainWindow", "Help"))
        
        self.current_tab = self.tabWidget.currentIndex()
        self.current_model = ''
        self.current_status = ''
        self.ICE_type = ''
        self.kde_feat_dict = {'gbt_52':['o_area_2', 'n_max', 'gamma', 'ei', 'delta_ab', 'beta']}
        
        self.thread_status = 0
        self.tabWidget.currentChanged.connect(self.on_tabwidget_changed)
        self.radioButton_2.clicked.connect(self.radioButton2_clicked)
        self.radioButton_4.clicked.connect(self.radioButton4_clicked)
        self.radioButton.clicked.connect(self.radioButton_clicked)
        self.radioButton_3.clicked.connect(self.radioButton3_clicked)
        self.pushButton.clicked.connect(self.pushButton_clicked)
        self.pushButton_2.clicked.connect(self.task_3)
        self.pushButton_3.clicked.connect(self.task_2)
        self.actionDownload.triggered.connect(self.download_action_func) 
        self.actionHelp.triggered.connect(self.helpme)
        
        param_count = 0
        print(self.kde_feat_dict)
        if 'gbt_52' in self.kde_feat_dict:
            for param in self.kde_feat_dict['gbt_52']:
                self.comboBox_6.addItem("")
                self.comboBox_6.setItemText(param_count, _translate("MainWindow", param))
                param_count +=1
                
        print('current tab: ',self.current_tab)
        print(self.current_model)
        
        if self.radioButton.isChecked():
            self.current_status = 1
        else:
            self.current_status = 0
            
        if self.radioButton_2.isChecked():
            self.ICE_type = 1
        else:
            self.ICE_type = 0
        
        self.find_tasks()
        
    
# ===================================================================================
#   FUNCTIONS THAT DEFINES THE CALL FOR EACH TASK/TAB CHANGE
# ===================================================================================
        
    def find_tasks(self):
        '''
        This function redirects to the tasks corresponding to which tab is clicked/selected
        '''
        
        if self.current_tab == 0:
            self.task_1()
        if self.current_tab == 1:
            self.task_2()
        if self.current_tab == 2:
            self.task_3()
        if self.current_tab == 3:
            self.task_4()
    
    def task_1(self):
        '''
        This function runs if the tab 1 is selected
        
            . It reads name of the model from the dropdown and calls model_reliance() method
            . Connects to the method of on event change of the model dropdowns
        
        '''
        print('in task_1')
        self.current_model = self.comboBox.currentText().lower()
        self.draw_model_reliance()
        self.display_model_details(self.textEdit)
        print('current_model: ',self.current_model)
        self.comboBox.currentTextChanged.connect(self.on_combobox_changed1)
        
    def task_2(self):
        '''
        This function runs if the tab 2 is selected
        
            . It reads name of the model and feature from the corresponding dropdowns and calls display_ICE_plots() methods
            . Connects to the methods of  event change of the model and feature dropdowns
        
        '''
        print('in task_2')
        self.current_feature = self.comboBox_5.currentText()
        self.current_model = self.comboBox_2.currentText().lower()
        self.display_ICE_plots()
        self.display_model_details(self.textEdit_2)
        print('current_model: ',self.current_model)
        self.comboBox_5.currentTextChanged.connect(self.on_combobox_changed5)
        self.comboBox_2.currentTextChanged.connect(self.on_combobox_changed2)
        
    def task_3(self):
        '''
        This function runs if the tab 3 is selected
        
            . It reads name of the model and feature from the corresponding dropdowns and calls display_counterfactuals() methods
            . Connects to the methods of  event change of the model and feature dropdowns
        
        '''
        print('in task_3')
        self.current_model = self.comboBox_3.currentText().lower()
        self.current_feature = self.comboBox_6.currentText()
        self.display_model_details(self.textEdit_3)
        print('current_model: ',self.current_model)
        print('inside task_3')
        self.comboBox_3.currentTextChanged.connect(self.on_combobox_changed3)
        self.comboBox_6.currentTextChanged.connect(self.on_combobox_changed6)
        self.display_counterfactuals(self.lineEdit.text())
        
    def task_4(self):
        '''
        This function runs if the tab 4 is selected
        
            . It reads name of the model and feature from the corresponding dropdowns and calls print_parameter_range() methods
            . Connects to the methods of  event change of the model and feature dropdowns
        
        '''
        print('in task_4')
        self.current_model = self.comboBox_4.currentText().lower()
        self.display_model_details(self.textEdit_4, self.textEdit_5)
        self.print_parameter_range()
        print('current_model: ',self.current_model)
        self.comboBox_4.currentTextChanged.connect(self.on_combobox_changed4)
        
# ===================================================================================
#   FUNCTIONS RELATED TO MENU BAR OPTIONS: DOWNLOAD REPORT & HELP
# ===================================================================================
        
    def download_action_func(self):
        '''
        This function runs when user clicks Download Report button
            . Get the name of the file to be saved as
            . Start the thread by passing the file name as parameter
        '''
        name = QtWidgets.QFileDialog.getSaveFileName()
        worker = Worker(self.file_save, name)
        self.threadpool.start(worker)
    
    def file_save(self, name):
        '''
        This function runs if the thread for downloading the report starts 
        '''
        print('Name:', name[0])
        if name[0]:
            pdf = Create_PDF()
            pdf.html2pdf(name[0])
            
    def helpme(self):
        '''
        This function runs if the Help button is selected from the menu
        '''
        url = QtCore.QUrl('help.txt')
        if not QtGui.QDesktopServices.openUrl(url):
            QtGui.QMessageBox.warning(self, 'Open Url', 'Could not open url')
            
# ===================================================================================
#   FUNCTIONS ON EVENT CHANGES
# ===================================================================================

    # =========================================
    #   On change of tabs
    # =========================================
            
    def on_tabwidget_changed(self, value):
        print("tab changed", value)
        self.current_tab = value
        self.find_tasks()
        
    # =========================================
    #   On change of dropdowns
    # =========================================

    #tab 1
    def on_combobox_changed1(self, value):
        '''
        This function runs if the value of model dropdown changes in tab1
            . Get the model name and call draw_model_reliance()
        '''
        print("combobox changed", value)
        self.current_model = value.lower()
        print('current_model: ',self.current_model)
        self.data = self.model.model_reliance(self.current_model)
        self.draw_model_reliance()
        self.display_model_details(self.textEdit)
     
    #tab 2
    def on_combobox_changed2(self, value):
        '''
        This function runs if the value of model dropdown changes in tab2
            . Get the model name
                . Call model_reliance to get important features
                . Add the features to the 'features dropdown' in tab2 sorted based on feature importance
                . With the selected model and feature call display_ICE_plots() method
        
        '''
        print("combobox changed", value)
        self.current_model = value.lower()
        print('current_model: ',self.current_model)
        
        self.data = self.model.model_reliance(self.current_model)
        sorted_data = sorted(self.data.items(), key=operator.itemgetter(1))
        param_count = 0
        cat_features = ["side","sex_f","sex_m","type_BF","type_SW"]
        for param in reversed(sorted_data):
            if param[0] not in cat_features:
                _translate = QtCore.QCoreApplication.translate
                self.comboBox_5.setItemText(param_count, _translate("MainWindow", param[0]))
                param_count +=1
            
        self.display_ICE_plots()
        self.display_model_details(self.textEdit_2)
        
    #tab 2
    def on_combobox_changed5(self, value):
        '''
        This function runs if the value of feature dropdown changes in tab2
            . Call display_ICE_plots()
        '''
        print("combobox changed", value)
        self.current_feature = value
        self.display_ICE_plots()
        print('current_feature: ',self.current_feature)
   
    #tab 3     
    def on_combobox_changed3(self, value):
        '''
        This function runs if the value of model dropdown changes in tab3
            . Call pushButton_clicked()
        '''
        print("combobox changed", value)
        self.current_model = value.lower()
        self.display_model_details(self.textEdit_3)
        self.pushButton_clicked()
    
    #tab 3
    def on_combobox_changed6(self, value):
        '''
        This function runs if the value of counterfactual feature dropdown changes in tab3
            . Call display_counterfactuals()
        '''
        print("combobox changed", value)
        self.current_feature = value
        self.display_counterfactuals(self.lineEdit.text())
        print('current_feature: ',self.current_feature)
     
    #tab 4
    def on_combobox_changed4(self, value):
        '''
        This function runs if the value of model dropdown changes in tab4
            . Call print_parameter_range()
        '''
        print("combobox changed", value)
        self.current_model = value.lower()
        self.print_parameter_range()
        print('current_model: ',self.current_model)
        self.display_model_details(self.textEdit_4, self.textEdit_5)
    
        
    # =========================================
    #   On change of Submit Button
    # =========================================       
    
    def pushButton_clicked(self):
        '''
        This function runs if the Submit button in Tab 3 is clicked
            . Fetch the value of instance number from the text box and call display_counterfactuals() method by starting a new thread
            . In the meanwhile loading a gif
        '''
        self.thread_status = 1
        
        worker = Worker(self.display_counterfactuals,  self.lineEdit.text())
        self.threadpool.start(worker)
        
        img = 'commons/load.gif'
        movie = QtGui.QMovie(img)
        self.label_5.setMovie(movie)
        movie.start()
                
        self.thread_status = 0
        print(self.thread_status)
        self.comboBox_3.blockSignals(False)
        
    # =========================================
    #   On change of Radio Buttons
    # ========================================= 

    #tab2        
    def radioButton2_clicked(self, value):
        '''
        This function runs if 'Normal ICE' in Tab 2 is clicked
        '''
        print(">>>>> radio Button clicked: ", value)
        self.radioButton_2.setChecked(True)
        self.radioButton_4.setChecked(False)
        self.ICE_type = 1
        self.display_ICE_plots()
    
    #tab2 
    def radioButton4_clicked(self, value):
        '''
        This function runs if 'Centered ICE' in Tab 2 is clicked
        '''
        print(">>>>> radio Button3 clicked: ", value)
        self.radioButton_4.setChecked(True)
        self.radioButton_2.setChecked(False)
        self.ICE_type = 0
        self.display_ICE_plots()
        
    #tab4
    def radioButton_clicked(self, value):
        '''
        This function runs if 'Ruptured' in Tab 4 is clicked
        '''
        print(">>>>> radio Button clicked: ", value)
        self.radioButton.setChecked(True)
        self.radioButton_3.setChecked(False)
        self.current_status = 1
        self.display_model_details(self.textEdit_4, self.textEdit_5)
        self.print_parameter_range()
    
    #tab4
    def radioButton3_clicked(self, value):
        '''
        This function runs if 'Unruptured' in Tab 4 is clicked
        '''
        print(">>>>> radio Button3 clicked: ", value)
        self.radioButton_3.setChecked(True)
        self.radioButton.setChecked(False)
        self.current_status = 0
        self.display_model_details(self.textEdit_4, self.textEdit_5)
        self.print_parameter_range()
        
        
# ===================================================================================
#   FUNCTIONS ON DISPLAYING PLOTS?FIGURES ON GUI
# ===================================================================================
        
        
    # =========================================
    #   DISPLAYING MODEL INFORMATION - ALL TASKS
    # ========================================= 
    
    def display_model_details(self, textEdit, textEdit_2=None):
        '''
            This function prints the details about the model selected from the dropdown under all tabs
            It also prints information specific to task 4 in tab4
        '''
        m_info = Model_Info()
        if textEdit_2 is not None: # for tab4
            details, instances = m_info.get_info(self.current_model, self.current_status)
        else: # for rest of the tabs
            details, instances = m_info.get_info(self.current_model)
        textEdit.setHtml(details)
        if textEdit_2 is not None:
            print('inside textEdit_2')
            print(instances)
            textEdit_2.setHtml(instances)
        
    # =========================================
    #   Task 1: MODEL RELIANCE
    # ========================================= 
    
    # tab1
    def draw_model_reliance(self):
        '''
            This function deals with plotting the features with their importance score
            obtained from Model_Reliance.
        '''
        plt.clf()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.figure.suptitle('Feature Importance - Model Reliance', fontsize=20)
        ax.set_xlabel('Importance', fontsize=18)
        ax.set_ylabel('Features', fontsize=18)
        ind = np.arange(len(list(self.data.values())))
        # create a bar plot
        ax.barh(ind, list(self.data.values()), tick_label=list(self.data), label='features')
        self.figure.savefig('img/model_reliance_{}.jpg'.format(self.current_model))
        # draw the bar plot on the specified area in tab1
        self.canvas.draw()
        plt.clf()
    
    
    # =========================================
    #   Task 2: ICE PLOTS
    # ========================================= 
    
    #tab2
    def display_ICE_plots(self):
        ''' 
            This function generates regular ICE as well as centered-ice plots
            And, the features are dynamically populated in the feature dropdown which is sorted based on model reliance 
        ''' 
        plt.clf()
        self.figure2.clear()
        test, test_label, ice_feature_col, z_temp = self.generate_ice.generate_ICE(self.current_model, self.current_feature)
        self.figure2.suptitle('Individual Conditional Expectation', fontsize=20)
        j = 0
        rug = 0 #For adding Rugs
        k = len(ice_feature_col)
        pdp_temp = [0]*len(ice_feature_col) #For having a PDP line
        ax = self.figure2.add_subplot(111)
        ax.plot([], color='tab:blue', label ='non-ruptured')  
        ax.plot([], color='tab:orange',label = 'ruptured')
        
        #Create Temporary dataframe for generating ICE plots
        for i in range(int(len(test)/len(ice_feature_col))):
            temp5 = pd.concat([z_temp[j:k],test[j:k]],axis=1)
            label_temp = float(test_label[j:k].mean()[0]) #For splitting lines into two labels
            temp5.columns = ('{}'.format(self.current_feature),'probability')
            temp5_1 = temp5.sort_values(by=['{}'.format(self.current_feature)])
            rug_temp = 0
            center_val = 0
            
            #ICE plots
            if self.ICE_type == 1:
                if label_temp < 0.5:
                    ax.plot(temp5_1['{}'.format(self.current_feature)],temp5_1['probability'],color = 'tab:blue',lw = 0.7,label='_nolegend_')
                else:
                    ax.plot(temp5_1['{}'.format(self.current_feature)],temp5_1['probability'],color = 'tab:orange', lw = 0.7,label='_nolegend_')
                pdp_arr = list(temp5_1['probability'])
                pdp_temp = [x + y for x, y in zip(pdp_temp , pdp_arr)]
                rug_temp = float(temp5_1['{}'.format(self.current_feature)].min())
                
            #Centered ICE plots
            else:            
                center_val = temp5_1.iloc[0,1] #Centered according to the smallest value 
                temp6 = pd.DataFrame(temp5_1['probability'] - center_val)
                if label_temp < 0.5:
                    ax.plot(temp5_1['{}'.format(self.current_feature)],temp6,color = 'tab:blue',lw = 0.7,label='_nolegend_')
                else:
                    ax.plot(temp5_1['{}'.format(self.current_feature)],temp6,color = 'tab:orange', lw = 0.7,label='_nolegend_')
                rug_temp = float(temp6.min())
            
            j+=len(ice_feature_col)
            k+=len(ice_feature_col)
            
            
            if rug_temp < rug:
                rug = rug_temp
                
        if self.ICE_type == 1:        
            pdp_temp[:] = [g / 100 for g in pdp_temp]   
            ax.plot(temp5_1['{}'.format(self.current_feature)],pdp_temp,color = 'tab:purple',lw = 5,label='_nolegend_')
                
                
        '''
            Add rugs dynamically according to the least value on y-axis(probability) predicted by each
            feature. The three conditions take care of positioning of the rugs so the plots don't get messy.
        '''
        if rug < 0: 
            ax.plot(temp5_1['{}'.format(self.current_feature)], [rug*1.1]*len(temp5_1), '|', color='k')
        elif rug == 0:
            rug = -0.001
            ax.plot(temp5_1['{}'.format(self.current_feature)], [rug]*len(temp5_1), '|', color='k')
        else:
            ax.plot(temp5_1['{}'.format(self.current_feature)], [rug*0.1]*len(temp5_1), '|', color='k')
        
        ax.set_xlabel('{}'.format(self.current_feature), fontsize=18)
        ax.set_ylabel('Predicted rupture probability', fontsize=18)
        ax.legend(loc = "upper right")
        if self.ICE_type == 1:
            self.figure2.savefig('img/iceplot_{}_{}_{}.jpg'.format(self.current_model, self.current_feature, self.ICE_type), dpi=100) #Remove if you don't want to save fig 
        else:
            self.figure2.savefig('img/iceplot_{}_{}_{}.jpg'.format(self.current_model, self.current_feature, self.ICE_type), dpi=100)
        # draw the ICE plot on the specified area in tab2
        self.canvas2.draw()
        plt.clf()
    
    
    # =========================================
    #   Task 3: DENSITY PLOTS OF COUNTERFACTUALS 
    # ========================================= 
    
    #tab3
    def display_counterfactuals(self, instance):
        '''
            This function displays the density plot which helps visualize the counterfactuals 
            of all possible features of the instance.
        '''
        param_count = 0
        file_found = False
        
        # search if the file name already exist. 
        if '{}_{}'.format(self.current_model, instance) in self.kde_feat_dict:
            file_found = True
                
        if file_found:
            print('inside display_counter_file_found')
            if self.current_feature == '':          # instance found to have no counterfactual features
                print('inside')
                img = 'commons/oops2.gif'           # display not available
                movie = QtGui.QMovie(img)
                self.label_5.setMovie(movie)
                movie.start()
            else:                                   # instance have counterfactual features 
                val = '{}_{}'.format(self.current_model,instance)
                _translate = QtCore.QCoreApplication.translate
                for param in self.kde_feat_dict[val]:
                    self.comboBox_6.addItem("")     # populate values to the dropdown dynamically
                    self.comboBox_6.setItemText(param_count, _translate("MainWindow", param))
                    param_count +=1
                img = 'img/counterfactual_{}_{}_{}.jpg'.format(self.current_model,instance, self.current_feature)
                pixmap = QtGui.QPixmap(img)
                self.label_5.setPixmap(pixmap)       # display plot as image
        else:
            self.comboBox_6.clear()
            # get list of counterfactual features of selected instance
            feature_list = self.counterfactuals.get_best_counterfactuals(int(instance), self.current_model)
            val = '{}_{}'.format(self.current_model, instance)
            self.kde_feat_dict[val] = feature_list
            # block any onchange event happening in the feature dropdown
            self.comboBox_6.blockSignals(True)
            _translate = QtCore.QCoreApplication.translate
            for param in self.kde_feat_dict[val]:
                print(param)
                self.comboBox_6.addItem("")          # populate values to dropdown dynamically
                self.comboBox_6.setItemText(param_count, _translate("MainWindow", param))
                param_count +=1
            self.current_feature = self.comboBox_6.currentText()
            # re-assuring the values in the dropdown are removed
            print('img/counterfactual_{}_{}_{}.jpg'.format(self.current_model,instance, self.current_feature))
            for each in [self.comboBox_6.itemText(i) for i in range(self.comboBox_6.count())]:
                if each == '':
                    self.comboBox_6.removeItem(self.comboBox_6.findText(each))
            time.sleep(3)
            if self.current_feature == '':           # instance found to have no counterfactual features
                print('inside')
                img = 'commons/oops2.gif'            # display not available
                movie = QtGui.QMovie(img)
                self.label_5.setMovie(movie)
                movie.start()
            else:
                img = 'img/counterfactual_{}_{}_{}.jpg'.format(self.current_model,instance, self.current_feature)
                pixmap = QtGui.QPixmap(img)
                self.label_5.setPixmap(pixmap)       # display plot as image
                self.comboBox_6.blockSignals(False)  # unblock onchange events happening in the feature dropdown
                
                
    # =========================================
    #   Task 4: BOX PLOTS FOR VALUE RANGES OF PARAMETERS
    # ========================================= 
    
    # tab4
    def print_parameter_range(self):
        '''
            This function displays boxplot for each rule that maximally satisifies instances and that 
            defines the value range of features for which the rule holds.
        '''
        plt.clf()
        self.figure1.clear()
        
        #get the ruleset and the value ranges
        data = self.param_range.define_ruleset(self.current_model, self.current_status)
        print(data)
        ax = self.figure1.add_subplot(111)
        self.figure1.suptitle('Range of Parameters', fontsize=20)
        ax.set_xlabel('Parameters', fontsize=18)
        ax.set_ylabel('Range of values', fontsize=18)
        pos, color_count = 1, 0
        x_ticks = []
        colors = ['blue', 'lightgreen', 'red', 'cyan', 'yellow', 'orange']
        for k, v in data.items():
            bp = ax.boxplot(v, positions = list(range(pos, pos+len(v))), widths = 0.6, patch_artist=True)
            for box in bp['boxes']:
                # change outline color
                box.set(color='black', linewidth=2)
                # change fill color
                box.set(facecolor = colors[color_count] )
            color_count += 1
                
            x_ticks.append(mean(list(range(pos, pos+len(v)))))
            pos += len(v) + 2
        
        ax.set_xticklabels(data.keys())
        for i in range(0, len(x_ticks)):
            if not x_ticks[i] == x_ticks[-1]:
                ax.axvline(x=mean([x_ticks[i],x_ticks[i+1]]))
        ax.set_xticks(x_ticks)
        self.figure1.savefig('img/box_plot_{}_{}.jpg'.format(self.current_model, str(self.current_status)))
        # draw the box plot on the specified area in tab3
        self.canvas1.draw()
        plt.clf()
        
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.lastWindowClosed.connect(app.quit)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    app.exec_()

