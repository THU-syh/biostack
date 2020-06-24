from PyQt5 import QtWidgets, QtGui
from PyQt5 import QtCore
import sys
import math

class GraphicsLineScene(QtWidgets.QGraphicsScene):
    """class which holds the qgraphics scene widget about line graphs along with 
    its own methods
    """
    def __init__(self, parent=None):
        super(GraphicsLineScene, self).__init__(parent)
        self.view = parent

    def contextMenuEvent(self, event):
        if event.reason() == event.Mouse and self.view.lineFigurePath is not None:
            event.accept()
            menu       = QtWidgets.QMenu()
            okAction   = QtWidgets.QAction('Ok', self)
            barAction  = QtWidgets.QAction('Bar Chart', self)
            lineAction = QtWidgets.QAction('Line Chart', self)            
            voidAction = QtWidgets.QAction('Unlabeled Chart', self)

            okAction.triggered.connect(self.okAction)
            barAction.triggered.connect(self.barAction)
            voidAction.triggered.connect(self.voidAction)
            lineAction.triggered.connect(self.lineAction)
            
            lineAction.setEnabled(False)
            
            if self.view.pathIds[self.view.lineFigurePath] == 2:
                lineAction.setEnabled(True)

            menu.addAction(okAction)
            menu.addAction(barAction)
            menu.addAction(voidAction)
            menu.addAction(lineAction)
            menu.exec_(event.screenPos())

    def okAction(self):
        self.view.pathCrr[self.view.lineFigurePath] = 1
        self.view.lineFigures.createItem(self.view.lineFigurePath, self.view.ratioOption)
        self.view.lineFigurePath = None
        self.view.lineFigureScene.clear()
        self.view.nextLineFigure()

    def lineAction(self):
        self.view.pathCrr[self.view.lineFigurePath] = 1
        self.view.pathIds[self.view.lineFigurePath] = 0
        self.view.lineFigures.createItem(self.view.lineFigurePath, self.view.ratioOption)
        self.view.lineFigurePath = None
        self.view.lineFigureScene.clear()
        self.view.nextLineFigure()

    def barAction(self):
        self.view.pathCrr[self.view.lineFigurePath] = 1
        self.view.pathIds[self.view.lineFigurePath] = 1
        self.view.barFigures.createItem(self.view.lineFigurePath, self.view.ratioOption)       
        self.view.lineFigurePath = None
        self.view.lineFigureScene.clear()
        self.view.nextLineFigure()

    def voidAction(self):
        self.view.pathCrr[self.view.lineFigurePath] = 1
        self.view.pathIds[self.view.lineFigurePath] = 2
        self.view.lineFigurePath = None
        self.view.lineFigureScene.clear()
        self.view.nextLineFigure()


class GraphicsBarScene(QtWidgets.QGraphicsScene):
    """GraphicsBarScene class
    """
    def __init__(self, parent=None):
        super(GraphicsBarScene, self).__init__(parent)
        self.view = parent

    def contextMenuEvent(self, event):
        if event.reason() == event.Mouse and self.view.barFigurePath is not None:
            event.accept()
            menu = QtWidgets.QMenu()
            okAction   = QtWidgets.QAction('Ok', self)
            barAction  = QtWidgets.QAction('Bar Chart', self)
            lineAction = QtWidgets.QAction('Line Chart', self)            
            voidAction = QtWidgets.QAction('Unlabeled Chart', self)

            okAction.triggered.connect(self.okAction)
            lineAction.triggered.connect(self.lineAction)
            voidAction.triggered.connect(self.voidAction)
            barAction.triggered.connect(self.barAction)
            
            barAction.setEnabled(False)

            # if the figure is unlabeled, enable the bar action
            if self.view.pathIds[self.view.barFigurePath] == 2:
                barAction.setEnabled(True)

            menu.addAction(okAction)
            menu.addAction(barAction)
            menu.addAction(voidAction)
            menu.addAction(lineAction)
            menu.exec_(event.screenPos()) 

    def okAction(self):
        self.view.pathCrr[self.view.barFigurePath] = 1
        self.view.barFigures.createItem(self.view.barFigurePath, self.view.ratioOption)
        self.view.barFigurePath = None
        self.view.barFigureScene.clear()
        self.view.nextBarFigure()

    def lineAction(self):
        self.view.pathCrr[self.view.barFigurePath] = 1
        self.view.pathIds[self.view.barFigurePath] = 0
        self.view.lineFigures.createItem(self.view.barFigurePath, self.view.ratioOption)       
        self.view.barFigurePath = None
        self.view.barFigureScene.clear()
        self.view.nextBarFigure()

    def barAction(self):
        self.view.pathCrr[self.view.barFigurePath] = 1
        self.view.pathIds[self.view.barFigurePath] = 1
        self.view.barFigures.createItem(self.view.barFigurePath, self.view.ratioOption)
        self.view.barFigurePath = None
        self.view.barFigureScene.clear()
        self.view.nextBarFigure()   

    def voidAction(self):
        self.view.pathCrr[self.view.barFigurePath] = 1
        self.view.pathIds[self.view.barFigurePath] = 2
        self.view.barFigurePath = None
        self.view.barFigureScene.clear()
        self.view.nextBarFigure()


class LineFigures(QtWidgets.QGraphicsScene):
    """docstring for LineFigures
    """
    def __init__(self, parent=None):
        super(LineFigures, self).__init__(parent)
        self.view        = parent
        self.figuresList = []
        self.color       = (182,182,182)
        self.brushWidth  = 3
        
        # The fixed size of single figure scene
        self.scaleX = 280
        self.scaleY = 400

        '''
        self.scrollbar = QtWidgets.QScrollBar()
        self.setHorizontalScrollBar(self.scrollbar)
        '''

    def createItem(self, figurePath, ratio=QtCore.Qt.IgnoreAspectRatio):
        self.figureItem = QtWidgets.QGraphicsPixmapItem()
        self.figure = QtGui.QPixmap(figurePath)
        self.ratio  = ratio
        self.scale()
        self.paint()
        self.figureItem.setPixmap(self.figure)
        self.figuresList.append(self.figureItem)
        self.offset = len(self.figuresList)               

        x, y = self.view.getWidgetPos(self.view.displayLineFigures)
        w, h = self.view.getWidgetDims(self.figure)
        
        if  self.offset == 1:
            # Set scene geometry
            self.view.displayLineFigures.setGeometry(QtCore.QRect(x,y,w,h))
            self.view.lineFigures.addItem(self.figureItem)
        else:
            self.arrangeScene(x, y, w, h)
            self.view.lineFigures.addItem(self.figureItem)
            self.figureItem.setPos(-(self.offset-1)*w,0)
    
    def arrangeScene(self, x, y, w, h):
        if x+self.offset*w > self.view.screenWidth:
            self.view.displayLineFigures.translate(w+2*self.brushWidth,0)
        else:
            self.view.displayLineFigures.setGeometry(QtCore.QRect(x,y,self.offset*w+2*self.brushWidth,h))

    def scale(self):
        self.figure = self.figure.scaled(self.scaleX, self.scaleY, 
                            self.ratio, 
                            QtCore.Qt.SmoothTransformation) 
    
    def paint(self):
        # Add bounding frame        
        picture      = QtGui.QPixmap(self.figure.width()+2*self.brushWidth-1,                         self.figure.height()+2*self.brushWidth-1)

        # Create Painter
        qp  = QtGui.QPainter(picture)
        pen = QtGui.QPen(QtGui.QColor(*self.color))
        pen.setWidth(self.brushWidth)
        qp.setPen(pen)
        qp.drawRect(0, 0, self.figure.width()+self.brushWidth, 
                        self.figure.height()+self.brushWidth)
        qp.drawPixmap(self.brushWidth-1,self.brushWidth-1, self.figure)
        self.figure = picture
        qp.end()

    @QtCore.pyqtSlot(int, int)
    def changeSliderPos(self, min, max):
        self.view.displayLineFigures.horizontalScrollBar().setSliderPosition(min)

class BarFigures(QtWidgets.QGraphicsScene):
    """docstring for BarFigures scene
    """
    def __init__(self, parent=None):
        super(BarFigures, self).__init__(parent)
        self.view        = parent
        self.figuresList = []

        # Painter Options
        self.color      = (182,182,182)
        self.brushWidth = 3

        # The fixed size of single figure scene
        self.scaleX     = 280
        self.scaleY     = 400       

    def createItem(self, figurePath, ratio=QtCore.Qt.IgnoreAspectRatio):
        self.figureItem = QtWidgets.QGraphicsPixmapItem()
        self.figure = QtGui.QPixmap(figurePath)
        self.ratio  = ratio
        self.scale()
        self.paint()
        self.figureItem.setPixmap(self.figure)
        self.figuresList.append(self.figureItem)
        self.offset = len(self.figuresList)

        x, y = self.view.getWidgetPos(self.view.displayBarFigures)
        w, h = self.view.getWidgetDims(self.figure)
        
        if self.offset == 1:
            # Set scene geometry equal to single figure
            self.view.displayBarFigures.setGeometry(QtCore.QRect(x,y,w,h))
            self.view.barFigures.addItem(self.figureItem)
        else:
            self.arrangeScene(x, y, w, h)
            self.view.barFigures.addItem(self.figureItem)
            self.figureItem.setPos(-(self.offset-1)*w,0)

    def arrangeScene(self, x, y, w, h):
        if  x+self.offset*w > self.view.screenWidth:
            self.view.displayBarFigures.translate(w+2*self.brushWidth,0)
        else:
            self.view.displayBarFigures.setGeometry(QtCore.QRect(x, y,                  self.offset*w+2*self.brushWidth, h))

    def scale(self):
        self.figure = self.figure.scaled(self.scaleX, self.scaleY, 
                            self.ratio, 
                            QtCore.Qt.SmoothTransformation) 
    
    def paint(self):
        # Add bounding frame        
        picture = QtGui.QPixmap(self.figure.width()+2*self.brushWidth-1,                    self.figure.height()+2*self.brushWidth-1)

        # Create Painter
        qp  = QtGui.QPainter(picture)
        pen = QtGui.QPen(QtGui.QColor(*self.color))
        pen.setWidth(self.brushWidth)
        qp.setPen(pen)
        qp.drawRect(0, 0, self.figure.width()+self.brushWidth, 
                        self.figure.height()+self.brushWidth)
        qp.drawPixmap(self.brushWidth-1,self.brushWidth-1, self.figure)
        self.figure = picture
        qp.end()

    @QtCore.pyqtSlot(int, int)
    def changeSliderPos(self, min, max):
        self.view.displayBarFigures.horizontalScrollBar().setSliderPosition(min)


class Overlay(QtWidgets.QWidget):
    """Overlay widget for loading gif while training 
    """
    def __init__(self, parent = None):
        super(Overlay, self).__init__(parent)
        palette = QtGui.QPalette()
        palette.setColor(palette.Background, QtCore.Qt.transparent)
        self.setPalette(palette)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground);
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents);
       
    def paintEvent(self, event):
   
        painter = QtGui.QPainter()
        painter.begin(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtGui.QPen(QtCore.Qt.NoPen))
       
        for i in range(6):
            if (self.counter / 5) % 6 == i:
                painter.setBrush(QtGui.QBrush(QtGui.QColor(127 + (self.counter % 5)*32, 127, 127)))
            else:
                painter.setBrush(QtGui.QBrush(QtGui.QColor(127, 127, 127)))
                painter.drawEllipse(
                self.width()/2 + 30 * math.cos(2 * math.pi * i / 6.0) - 10,
                self.height()/2 + 30 * math.sin(2 * math.pi * i / 6.0) - 10,
                20, 20)
       
        painter.end()
   
    def showEvent(self, event):
        # Set time interval
        self.timer   = self.startTimer(90)
        self.counter = 0
   
    def timerEvent(self, event):
   
        self.counter += 1
        self.update()

if __name__ == '__main__':
    raise Exception('This module is not executable')
    sys.exit(-1)