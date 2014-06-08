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
        
        self.WorkThread = QtCore.QThread()
        self.WorkTimer = QtCore.QTimer(None)
        self.WorkTimer.setInterval(20)
        self.VProc = ipui.VideoProcessor(mmcore=self.mmc, parent=self)
        self.VProc.moveToThread(self.WorkThread)
#         self.WorkTimer.moveToThread(self.WorkThread)
        
        self.MControl = ipui.MicroscopeControl(parent=self)
        self.AControl = ipui.AnalysisControl(parent=self)
        self.GLWiget = ipui.GLFrame()
        self.GLWiget.setMinimumSize(640, 480)
        #self.GLWiget.setFixedSize(640, 480)  # Temporary
        self.setCentralWidget(self.GLWiget)
        self.setWindowTitle('Immunopy')
        
        self.dock = QtGui.QDockWidget('Ground control', parent=self)
        self.toolbox = QtGui.QToolBox()
        self.toolbox.addItem(self.MControl, 'Microscope control')
        self.toolbox.addItem(self.AControl, 'Analysis control')
        
        self.dock.setWidget(self.toolbox)
        self.addDockWidget(
            QtCore.Qt.DockWidgetArea(QtCore.Qt.LeftDockWidgetArea), self.dock)
        
        self.WorkThread.started.connect(self.WorkTimer.start)
        self.WorkThread.started.connect(self.VProc.start_acquisition)
        self.WorkThread.finished.connect(self.WorkTimer.stop)
        self.WorkThread.finished.connect(self.VProc.stop_acquisition)
        self.WorkTimer.timeout.connect(self.VProc.process_frame)
        
        self.VProc.newframe.connect(self.updateFrame)
        self.CMicro.scale_changed.connect(self.VProc.setScale)


    @QtCore.Slot()
    def updateFrame(self):
        self.GLWiget.setData(self.VProc.rgb)
    
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
