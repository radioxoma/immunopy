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
import iptools
import MMCorePy

MM_CONFIGURATION_NAME = "MMConfig_dcam.cfg"
MM_CIRCULAR_BUFFER = 100
DEF_OBJECTIVE = '10'


class MainWindow(QtGui.QMainWindow):
    """Main GUI window.
    """
    def __init__(self):
        super(MainWindow, self).__init__()
        self.mmc = MMCorePy.CMMCore()
        self.mmc.enableStderrLog(False)
        self.mmc.enableDebugLog(False)
        self.mmc.loadSystemConfiguration(MM_CONFIGURATION_NAME)
        self.mmc.setCircularBufferMemoryFootprint(MM_CIRCULAR_BUFFER)
        
        self.CMicro = iptools.CalibMicro(DEF_OBJECTIVE)
        self.MControl = ipui.MicroscopeControl(parent=self)
        self.AControl = ipui.AnalysisControl(parent=self)
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
