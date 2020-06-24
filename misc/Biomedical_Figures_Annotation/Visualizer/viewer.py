#! /usr/bin/env python3

from PyQt5 import QtWidgets, QtGui
from PyQt5 import QtCore

import sys
import random
import pandas as pd
import os
import numpy as np
from scipy.misc import imresize

try:
    from keras.preprocessing import image
    from keras.models import Model, Sequential
    from keras.applications.resnet50 import preprocess_input
    from keras.applications.vgg19 import VGG19
except ImportError:
    print('Import Error')

from sklearn.externals import joblib
from sklearn.svm import SVC
from Utils.utils import (GraphicsLineScene,
                         GraphicsBarScene,
                         LineFigures,
                         BarFigures,
                         Overlay)


class Viewer(QtWidgets.QMainWindow):
    """The main window of the annotator
    """
    def __init__(self):
        super(Viewer, self).__init__()
        # File Extensions
        self.imageExt  = '.jpg'
        self.joblibExt = '.pkl'
        self.csvExt    = '.csv'

        self.pathIds   = {}
        self.pathCrr   = {}

        # Init the ratio of widget scene
        self.ratioOption = QtCore.Qt.IgnoreAspectRatio

        # Initialize the figures
        self.lineFigure = QtGui.QImage()
        self.barFigure  = QtGui.QImage()

        # Hold the path for each figure frame
        self.barFigurePath  = None
        self.lineFigurePath = None

        # Set title
        self.applicationTitle = 'BioTool Annotator v1.0'
        self.setWindowTitle(self.applicationTitle)

        # Instantiate the window
        self.initUI()

    def initUI(self):
        """Initialize the UI
        """

        # Instantiate toolbar
        self.toolbar = self.addToolBar('Tools')
        self.toolbar.setMovable(False)

        # Add the tool buttons
        iconDir = os.path.join(os.path.dirname(sys.argv[0]), 'icons')
        loadAction = QtWidgets.QAction(QtGui.QIcon(os.path.join(iconDir, 'open.png')), '&Tools', self)
        loadAction.setShortcuts(['Ctrl+O'])
        loadAction.triggered.connect(self.loadPredictions)
        self.toolbar.addAction(loadAction)
        loadAction.setToolTip('Select folder')

        # Open the csv file action
        csvAction = QtWidgets.QAction(QtGui.QIcon(os.path.join(iconDir, 'csv.png')), '&Tools', self)
        csvAction.setShortcuts(['Ctrl+C'])
        csvAction.triggered.connect(self.openCsv)
        self.toolbar.addAction(csvAction)
        csvAction.setToolTip('Open Csv')

        saveAction = QtWidgets.QAction(QtGui.QIcon(os.path.join(iconDir, 'save.png')), '&Tools', self)
        saveAction.triggered.connect(self.saveData)
        saveAction.setToolTip('Save File')
        saveAction.setShortcuts(['Ctrl+S'])
        self.toolbar.addAction(saveAction)

        # Save file to csv
        trainAction = QtWidgets.QAction(QtGui.QIcon(os.path.join(iconDir, 'learning.png')), '&Tools', self)
        trainAction.triggered.connect(self.trainModel)
        trainAction.setToolTip('Train Model')
        trainAction.setShortcuts(['Ctrl+L'])
        self.toolbar.addAction(trainAction)

        helpAction = QtWidgets.QAction(QtGui.QIcon(os.path.join(iconDir, 'help19.png')), '&Tools', self)
        helpAction.triggered.connect(self.displayHelpMessage)
        helpAction.setToolTip('Help')
        helpAction.setShortcuts(['Ctrl+H'])
        self.toolbar.addAction(helpAction)

        # Close the application
        exitAction = QtWidgets.QAction(QtGui.QIcon(os.path.join(iconDir, 'exit.png')), '&Tools', self)
        exitAction.setShortcuts(['Esc'])
        exitAction.triggered.connect(self.close)
        self.toolbar.addAction(exitAction)
        exitAction.setToolTip('Exit')

        self.defaultStatusBar = 'Ready'
        self.statusBar().showMessage(self.defaultStatusBar)

        # Enable mouse move events
        self.setMouseTracking(True)
        self.toolbar.setMouseTracking(True)

        # Init docked widgets
        self.initDocks()

        # Open main window in full screen
        self.showFullScreen()

        # Show
        self.show()

        # Display the help message
        self.displayHelpMessage()

    def initDocks(self):
        """Initialize the docks inside the Main window
        """
        # Define the grid of widgets
        gridLayout = QtWidgets.QGridLayout()
        gridLayout.setOriginCorner(QtCore.Qt.TopLeftCorner)

        # Set QWidget object as main window in order to develop the appropriate functions
        widget = QtWidgets.QWidget(self)
        widget.setLayout(gridLayout)
        self.setCentralWidget(widget)

        # Set the text font
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)

        # Add figure widget scenes
        self.lineFigureScene = GraphicsLineScene(self)
        self.barFigureScene  = GraphicsBarScene(self)

        # Init view windows
        self.displayLineFigure = QtWidgets.QGraphicsView()
        self.displayBarFigure  = QtWidgets.QGraphicsView()

        self.displayLineFigure.setScene(self.lineFigureScene)
        self.displayBarFigure.setScene(self.barFigureScene)

        # Initialize the classification scenes
        self.lineFigures = LineFigures(self)
        self.barFigures  = BarFigures(self)
        self.displayLineFigures = QtWidgets.QGraphicsView(self.lineFigures)
        self.displayBarFigures  = QtWidgets.QGraphicsView(self.barFigures)

        # Set item index method
        self.lineFigures.setItemIndexMethod(QtWidgets.QGraphicsScene.BspTreeIndex)
        self.barFigures.setItemIndexMethod(QtWidgets.QGraphicsScene.BspTreeIndex)

        # Define text widgets
        lineText = QtWidgets.QLabel()
        lineText.setFont(font)
        lineText.setText('Line Figures Classification')
        #
        barText = QtWidgets.QLabel()
        barText.setFont(font)
        barText.setText('Bar Figures Classification')

        # Add widgets to grid layout
        gridLayout.addWidget(lineText, 1, 0, 1, -1, QtCore.Qt.AlignHCenter)
        gridLayout.addWidget(barText, 3, 0, 1, -1, QtCore.Qt.AlignHCenter)
        gridLayout.addWidget(self.displayLineFigure, 2, 0, QtCore.Qt.AlignLeft)
        gridLayout.addWidget(self.displayBarFigure, 4, 0, QtCore.Qt.AlignLeft)
        gridLayout.addWidget(self.displayLineFigures, 2, 1, 1, -1, QtCore.Qt.AlignLeft)
        gridLayout.addWidget(self.displayBarFigures, 4, 1, 1, -1, QtCore.Qt.AlignLeft)

        gridLayout.setHorizontalSpacing(70)
        gridLayout.setVerticalSpacing(15)

        self.screenWidth  = QtWidgets.QDesktopWidget().width()
        self.screenHeight = QtWidgets.QDesktopWidget().height()

        # Create slots to update slider initial position
        self.displayBarFigures.horizontalScrollBar().rangeChanged.connect(                               self.barFigures.changeSliderPos)
        self.displayLineFigures.horizontalScrollBar().rangeChanged.connect(                               self.lineFigures.changeSliderPos)

        # Overlay loading widget
        self.overlay = Overlay(self)
        self.overlay.hide()

    def resizeEvent(self, event):
        """Resize overlay according to widget size
        """
        event.accept()
        self.overlay.resize(event.size())
        # Move gif to the center of the widget
        self.overlay.move(self.rect().center() - self.overlay.rect().center())

    def openCsv(self):
        dir_path     = os.path.dirname(os.path.realpath(__file__))
        message      = 'Select csv file'
        folderDialog = QtWidgets.QFileDialog(self, message, dir_path)
        folderDialog.setFileMode(QtWidgets.QFileDialog.AnyFile)
        folderDialog.setNameFilter('CSV files (*.csv)')
        folderDialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        fileName   = [] # Returns a list of the directory

        # Plot the window to select the csv file
        if folderDialog.exec_():
            fileName = folderDialog.selectedFiles()
            if self.csvExt in str(fileName):
                self.loadCsv(str(fileName[0]))
            else:
                message = 'Only csv files'
                self.messageBox(message)
                return

        self.selectFigures()

    def loadPredictions(self):
        """Load the joblib or csv file which contains the dictionary of
        predictions
        """

        dir_path     = os.path.dirname(os.path.realpath(__file__))
        message      = 'Select folder'
        folderDialog = QtWidgets.QFileDialog(self, message, dir_path)
        folderDialog.setFileMode(QtWidgets.QFileDialog.Directory)
        folderDialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        fileName   = [] # Returns a list of the directory

        # Plot the window to select the csv file
        if folderDialog.exec_():
            fileName = folderDialog.selectedFiles()
            # Debug
            #fileName = ['/media/dimitris/TOSHIBA EXT/Image_Document_Classification/PMC-Dataset']
            print(fileName)
            if os.path.isdir(str(fileName[0])):
                self.loadFolder(str(fileName[0]))
            else:
                message = 'Only csv files'
                self.messageBox(message)
                return

        self.selectFigures()

    def loadFolder(self, path):
        # Iterate through folders and get jpg files
        paths = []
        for root, dirs, files in os.walk(path):
            for file_ in files:
                if self.imageExt in file_:
                    paths.append(os.path.join(root, file_))

        self.pathCrr = {path:0 for path in paths}
        self.pathIds = {path:2 for path in paths}

    def loadCsv(self, file):
        data = pd.read_csv(file, header=None, names=['path', 'id', 'cid'])
        self.pathIds = data.set_index('path').to_dict()['id']
        self.pathCrr = data.set_index('path').to_dict()['cid']

    def nextLineFigure(self):
        for path, cid in self.pathCrr.items():
            if cid == 0 and (self.pathIds[path] == 0 or self.pathIds[path] == 2):
                if path == self.barFigurePath and self.barFigure is not None:
                    continue
                self.plotFigures(path)
                break

    def nextBarFigure(self):
        for path, cid in self.pathCrr.items():
            if cid == 0 and (self.pathIds[path] == 1 or self.pathIds[path] == 2):
                if path == self.lineFigurePath and self.lineFigurePath is not None:
                    continue
                self.plotFigures(path)
                break

    def selectFigures(self):
        """Go through figures from the loaded file
        """
        # Shuffle the dictionary
        dict_items = list(self.pathCrr.items())
        random.shuffle(dict_items)
        self.pathCrr = {}
        self.pathCrr = {key:value for (key,value) in dict_items}

        self.nextBarFigure()
        self.nextLineFigure()


    def getWidgetPos(self, widget):
        return widget.x(), widget.y()

    def getWidgetDims(self, widget):
        return widget.width(), widget.height()

    def checkFigureSize(self, figure):
        """Resize low resolution figures in order to plot them properly
        """
        # scale factor
        f = 3
        width, height = self.getWidgetDims(figure)

        if (width < 100) and (height < 100):
            return figure.scaled(width*f, height*f, self.ratioOption, QtCore.Qt.SmoothTransformation)
        elif width<100:
            return figure.scaled(width*f, self.height, self.ratioOption, QtCore.Qt.SmoothTransformation)

        elif height<100:
            return figure.scaled(self.width, height*f, self.ratioOption, QtCore.Qt.SmoothTransformation)
        else:
            return figure.scaled(self.width, self.height, self.ratioOption, QtCore.Qt.SmoothTransformation)

    def plotFigures(self, path):
        """Method which plots the figures in the scenes
        """
        self.width  = 500
        self.height = 400

        if self.pathIds[path] == 0:
            self.lineFigurePath = path
            self.lineFigure.load(path)
            self.lineFigure = self.checkFigureSize(self.lineFigure)
            self.lineFigureScene.addPixmap(QtGui.QPixmap.fromImage(                                     self.lineFigure))
            x, y = self.getWidgetPos(self.displayLineFigure)
            w, h = self.getWidgetDims(self.lineFigure)
            self.displayLineFigure.setGeometry(QtCore.QRect(x,y,w,h))
            self.displayLineFigure.fitInView(self.displayLineFigure.sceneRect()                             , self.ratioOption)
        elif self.pathIds[path] == 1:
            self.barFigurePath = path
            self.barFigure.load(path)
            self.barFigureScene.addPixmap(QtGui.QPixmap.fromImage(self.barFigure))
            self.barFigure = self.checkFigureSize(self.barFigure)

            x, y = self.getWidgetPos(self.displayBarFigure)
            w, h = self.getWidgetDims(self.barFigure)
            self.displayBarFigure.setGeometry(QtCore.QRect(x,y,w,h))
            self.displayBarFigure.fitInView(self.barFigureScene.sceneRect(), self.ratioOption)
        else:
            if self.barFigurePath is None:
                self.barFigurePath = path
                self.barFigure.load(path)
                self.barFigure = self.checkFigureSize(self.barFigure)
                self.barFigureScene.addPixmap(QtGui.QPixmap.fromImage(                          self.barFigure))
                x, y = self.getWidgetPos(self.displayBarFigure)
                w, h = self.getWidgetDims(self.barFigure)
                self.displayBarFigure.setGeometry(QtCore.QRect(x,y,w,h))
                self.displayBarFigure.fitInView(self.barFigureScene.sceneRect()                            , self.ratioOption)
            elif self.lineFigurePath is None:
                self.lineFigurePath = path
                self.lineFigure.load(path)
                self.lineFigure = self.checkFigureSize(self.lineFigure)
                self.lineFigureScene.addPixmap(QtGui.QPixmap.fromImage(                           self.lineFigure))
                x, y = self.getWidgetPos(self.displayLineFigure)
                w, h = self.getWidgetDims(self.lineFigure)
                self.displayLineFigure.setGeometry(QtCore.QRect(x,y,w,h))
                self.displayLineFigure.fitInView(self.lineFigureScene.sceneRect                            (), self.ratioOption)

    def saveData(self):
        """Saves the data into csv after correction
        """
        pdIds = pd.DataFrame.from_dict(self.pathIds, orient='index')
        pdCrr = pd.DataFrame.from_dict(self.pathCrr, orient='index', columns=['cid'])
        mergedData = pd.concat([pdIds, pdCrr['cid']], axis=1, ignore_index=False)

        # Create the save dialog box
        name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File',
        '', 'csv files (*.csv)', 'csv file (*.csv)')

        if not name:
            return
        # Check the extension when saving
        if self.csvExt in name:
            mergedData.to_csv(name, header=False, index=True)
        else:
            message = 'Error saving file {}.'.format(name)
            self.messageBox(message)

    def trainModel(self):
        img_rows = 800
        img_cols = 600
        channels = 3
        imgFeats = np.array([])
        yLabels  = np.array([])
        allPaths = [] # Keep the paths of labeled figures

        # Change status to training
        message = 'Training...'
        self.statusBar().showMessage(message)

        # Instantiate model
        model = VGG19(include_top=False, input_shape=(img_rows, img_cols,channels), pooling=None, weights='imagenet')

        # Instantiate Loading Logo
        self.overlay = Overlay(self)
        self.overlay.show()

        for path, cid in self.pathCrr.items():
            if cid == 1:
                figure  = self.loadImage(path, sampleSize=(img_rows, img_cols))
                yLabels = np.append(yLabels, np.float(self.pathIds[path]))
                y_pred  = model.predict(figure, verbose=1)
                allPaths.append(path)

                # check dimensions
                if imgFeats.size == 0:
                    imgFeats = y_pred
                else:
                    imgFeats = np.concatenate((imgFeats, np.expand_dims(y_pred, axis=0)), axis=0)

        # Reshape features
        # print(imgFeats.shape)
        imgFeats = np.reshape(imgFeats, (imgFeats.shape[0], -1))
        svm = self.svmClassifier(imgFeats, yLabels)
        self.saveSvmModel(svm)

        # Update the class ids
        for index, path in enumerate(allPaths):
            self.pathIds[path] = yLabels[index]

        self.overlay.hide() # Hide Logo
        self.statusBar().showMessage(self.defaultStatusBar)
        self.selectFigures()

    def saveSvmModel(self, model):
        """Needs to be reimplemented with proper gui
        """
                # Create the save dialog box
        name, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Model Parameters',
        '', 'pkl files (*.pkl)', 'pkl file (*.pkl)')

        if not name:
            return
        # Check the extension when saving
        if self.joblibExt in name:
            joblib.dump(model, name)
        else:
            message = 'Error saving file {}.'.format(name)
            self.messageBox(message)

    def svmClassifier(self, feats, labels, coef=12.):
        return SVC(C=coef).fit(feats, labels)

    def loadImage(self, imagePath, show=False, scale=True, sampleSize=(800,600)):
        img = image.load_img(imagePath)
        img = image.img_to_array(img)

        # Scale image
        if scale:
            img = imresize(img, size=sampleSize, interp='cubic')
            print(img.shape)
            img = np.clip(img, 0, 255)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        return img

    def messageBox(self, message):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(message)
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()

    def closeEvent(self, event):
        event.accept()

    def displayHelpMessage(self):
        """Help message box
        """
        message = self.applicationTitle + '\n\n'
        message += 'INSTRUCTIONS\n'
        message += ' - If you do not have any csv file with annotated figures, select the folder which contains the set of biomedical academic files and the tool extracts the \'jpg\' figures only\n'
        message += ' - Annotate some figures by clicking right click and select the label to annotate\n'
        message += ' - Click the training button to get some predictions in order to classify the rest figures automatically\n\n'
        message += 'CONTROLS\n'
        message += ' - Select folder [ctrl+O]\n'
        message += ' - Open csv file with annotations [ctrl+c]\n'
        message += ' - Save the annotations to csv file [ctrl+s]\n'
        message += ' - Train model [ctrl+L]\n'
        message += ' - Help [ctr+H]\n'
        message += ' - Exit viewer [esc]'

        QtWidgets.QMessageBox.about(self, 'HELP', message)
        self.update()

    # Destructor
    def __del__(self):
        return

if __name__ == '__main__':
    application = QtWidgets.QApplication(sys.argv)
    view        = Viewer()
    sys.exit(application.exec_())
