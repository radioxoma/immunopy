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
        
        self.WorkThread = QtCore.QThread()
        self.WorkTimer = QtCore.QTimer(None)
        self.WorkTimer.setInterval(20)
        self.VProc = ipui.VideoProcessor(mmcore=self.mmc, parent=self)
        # Both must be in same thread, otherwise signals may be missing.
        self.VProc.moveToThread(self.WorkThread)
        self.WorkTimer.moveToThread(self.WorkThread)
        
        self.MControl = ipui.MicroscopeControl(parent=self)
        self.AControl = ipui.AnalysisControl(parent=self)
        self.GLWiget = ipui.GLFrame()
        self.GLWiget.setMinimumSize(640, 480)
        #self.GLWiget.setFixedSize(640, 480)  # Temporary
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
        
        self.WorkThread.started.connect(self.WorkTimer.start)
        self.WorkThread.started.connect(self.VProc.start_acquisition)
        self.WorkThread.finished.connect(self.WorkTimer.stop)
        self.WorkThread.finished.connect(self.VProc.stop_acquisition)
        self.WorkTimer.timeout.connect(self.VProc.process_frame)
        
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
        self.WorkThread.quit()
        if self.WorkThread.isRunning():
            self.WorkThread.wait()
        self.mmc.reset()

    def closeEvent(self, event):
        print('Shutdown safely...')
        self.shutdown()
        event.accept()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
