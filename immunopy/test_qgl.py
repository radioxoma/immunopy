#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 2014-06-07

@author: Eugene Dvoretsky
"""

import sys
import time
from scipy import misc
from PySide import QtGui
from ipui import GLFrame


img = misc.imread('image/hdab256.tif')

cycles = 100.0
counter = 0
start_time = time.time()


class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.widget = GLFrame()
        self.widget.setData(img)
        self.setCentralWidget(self.widget)

        # global counter
        # while counter < cycles:
        #     counter += 1
        #     self.widget.setData(img)
        # print('FPS: %f') % (cycles / (time.time() - start_time))


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
