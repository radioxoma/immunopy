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
import MMCorePyExperimental as MMCorePy

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
        
        self.workThread = QtCore.QThread()
        self.VProc = ipui.VideoProcessor(mmcore=self.mmc, parent=self)
        self.VProc.moveToThread(self.workThread)
        self.workThread.start()
        
        self.MControl = ipui.MicroscopeControl(parent=self)
        self.AControl = ipui.AnalysisControl(parent=self)
        self.GLWiget = ipui.GLFrame()
        self.GLWiget.setMinimumSize(640, 480)
        self.setCentralWidget(self.GLWiget)
        self.setWindowTitle('Immunopy')
        
        self.dock = QtGui.QDockWidget('Ground control', parent=self)
        self.dockcontainer = QtGui.QWidget()
        self.dockvbox = QtGui.QVBoxLayout()
        self.dockcontainer.setLayout(self.dockvbox)
        self.dockvbox.addWidget(self.MControl)
        self.dockvbox.addWidget(self.AControl)
        self.dock.setWidget(self.dockcontainer)
        self.addDockWidget(
            QtCore.Qt.DockWidgetArea(QtCore.Qt.LeftDockWidgetArea), self.dock)
        
        self.VProc.newframe.connect(self.updateFrame)
        self.VProc.histogramready.connect(self.MControl.setHistogram)
        self.CMicro.scale_changed.connect(self.VProc.setScale)
        
        self.AControl.vtype.valueChanged.connect(self.VProc.setVtype)
        self.AControl.sizemax.valueChanged.connect(self.VProc.setMaxSize)
        self.AControl.sizemin.valueChanged.connect(self.VProc.setMinSize)
        self.AControl.peak_dist.valueChanged.connect(self.VProc.setPeakDistance)
        self.AControl.shift_th.valueChanged.connect(self.VProc.setThresholdShift)

    @QtCore.Slot()
    def updateFrame(self):
        self.GLWiget.setData(self.VProc.out)
    
    @QtCore.Slot()
    def shutdown(self):
        """Switch off all stuff and exit."""
        if self.workThread.isRunning():
            self.workThread.quit()
            self.workThread.wait()
        self.mmc.reset()
        print('Shutdown safely...')

    def closeEvent(self, event):
        self.shutdown()
        event.accept()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
