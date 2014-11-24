#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 2014-06-07

@author: Eugene Dvoretsky
"""

import os
import sys
import time
from scipy import misc
from PySide import QtGui

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curdir + '/../immunopy')

from ipui import GLFrame


img = misc.imread('../immunopy/image/hdab256.tif')

cycles = 100
counter = 0
start_time = time.time()


class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.resize(300, 300)
        self.widget = GLFrame()
        self.widget.setData(img)
        self.setCentralWidget(self.widget)

#         global counter
#         while counter < cycles:
#             counter += 1
#             self.widget.setData(img)
#         print('FPS: %f') % (cycles / (time.time() - start_time))


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
