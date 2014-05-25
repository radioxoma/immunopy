#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 2014-05-25

@author: radioxoma
"""

import sys
from PySide import QtCore
from PySide import QtGui
import ipui
import MMCorePy

DEVICE = ['Camera', 'DemoCamera', 'DCam']
# DEVICE = ['Camera', 'OpenCVgrabber', 'OpenCVgrabber']
# DEVICE = ['Camera', "BaumerOptronic", "BaumerOptronic"]


class MainWindow(QtGui.QMainWindow):
    """Main GUI window.
    """
    def __init__(self):
        super(MainWindow, self).__init__()
        self.MControl = ipui.MicroscopeControl()
        self.AControl = ipui.AnalysisControl()
        self.GLWiget = ipui.GLFrame(width=640, height=480)
        self.GLWiget.setFixedSize(640, 480)  # Temporary
        self.setCentralWidget(self.GLWiget)
        self.setWindowTitle('Immunopy')
        
        self.dock = QtGui.QDockWidget('Ground control', parent=self)
        self.toolbox = QtGui.QToolBox()
        self.toolbox.addItem(self.MControl, 'Microscope control')
        self.toolbox.addItem(self.AControl, 'Analysis control')
        
        self.dock.setWidget(self.toolbox)
        self.addDockWidget(
            QtCore.Qt.DockWidgetArea(QtCore.Qt.LeftDockWidgetArea), self.dock)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
