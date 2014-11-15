#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 2014-05-25

@author: Eugene Dvoretsky

Immunopy - an IHC image real time analyzer.
"""

import sys
import os
from PySide import QtCore
from PySide import QtGui

MM_CIRCULAR_BUFFER = 100
DEF_OBJECTIVE = '20'
FAKE_CAMERA = True
MM_DEBUG = False


class MainWindow(QtGui.QMainWindow):
    """Main GUI window.
    """
    def __init__(self):
        super(MainWindow, self).__init__()
        self.mmc = MMCorePy.CMMCore()
        self.mmc.enableStderrLog(MM_DEBUG)
        self.mmc.enableDebugLog(MM_DEBUG)
        self.mmc.loadSystemConfiguration(MM_CONFIGURATION_NAME)
        self.mmc.setCircularBufferMemoryFootprint(MM_CIRCULAR_BUFFER)
        self.MPModel = mmanager.MicromanagerPropertyModel(
            self.mmc, deviceLabel=self.mmc.getCameraDevice())
        self.MPBrowser = mmanager.MicromanagerPropertyBrowser(self.MPModel)

        self.CMicro = iptools.CalibMicro(objective=DEF_OBJECTIVE)

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
        
        self.statBrowser = statdata.StatisticsBrowser(self.VProc.getModel())
        self.dockStat = QtGui.QDockWidget('Statistics', parent=self)
        self.dockStat.setWidget(self.statBrowser)
        self.addDockWidget(
            QtCore.Qt.DockWidgetArea(QtCore.Qt.BottomDockWidgetArea), self.dockStat)
        
        self.VProc.newframe.connect(self.updateFrame)
        self.VProc.histogramready.connect(self.MControl.setHistogram)
        self.CMicro.scale_changed.connect(self.VProc.setScale)
        
        self.MControl.sbx_adjust_r.valueChanged.connect(self.VProc.setRShift)
        self.MControl.sbx_adjust_g.valueChanged.connect(self.VProc.setGShift)
        self.MControl.sbx_adjust_b.valueChanged.connect(self.VProc.setBShift)
        
        self.AControl.vtype.valueChanged.connect(self.VProc.setVtype)
        self.AControl.sizemax.valueChanged.connect(self.VProc.setMaxSize)
        self.AControl.sizemin.valueChanged.connect(self.VProc.setMinSize)
        self.AControl.peak_dist.valueChanged.connect(self.VProc.setPeakDistance)
        self.AControl.dab_th_shift.valueChanged.connect(self.VProc.setDabThresholdShift)
        self.AControl.hem_th_shift.valueChanged.connect(self.VProc.setHemThresholdShift)
        
        self.statBrowser.wantAssay.connect(self.VProc.pushAssay)
        self.VProc.modelGotAssay.connect(self.statBrowser.ready)

        self.createMenus()

    def createMenus(self):
        """Create parent-less menu (MacOS likes it).

        Microscope("Configure device..."),
        View ("Ground control", "Statistics"),
        Help ("About...")
        """
        menuBar = QtGui.QMenuBar(parent=None)
        self.setMenuBar(menuBar)
        microscopeMenu = menuBar.addMenu('&Microscope')
        microscopeMenu.addAction(self.MPBrowser.showPropertyBrowserAction)
        viewMenu = menuBar.addMenu('&View')
        viewMenu.addAction(self.dock.toggleViewAction())
        viewMenu.addAction(self.dockStat.toggleViewAction())
        menuBar.addSeparator()  # Motif, CDE likes it.
        helpMenu = menuBar.addMenu('&Help')
        helpAction = helpMenu.addAction('&About...', self.showHelp)

    def showHelp(self):
        desc = """
        <p>This program perform real time image analysis of IHC-stained assays.</p>
        <p>Developed by Eugene Dvoretsky, Vitebsk State Medical University, 2014</p>
        <p><a href="mailto:radioxoma@gmail.com?subject=Immunopy">radioxoma@gmail.com</a></p>
        """
        QtGui.QMessageBox.about(self, "About Immunopy", desc)

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
    splash = QtGui.QSplashScreen(QtGui.QPixmap("image/slide.png"),
        QtCore.Qt.WindowStaysOnTopHint)
    splash.show()
    splash.showMessage("Loading statistics module...", color=QtCore.Qt.gray)
    app.processEvents()
    import statdata
    splash.showMessage("Loading imaging toolset...", color=QtCore.Qt.gray)
    app.processEvents()
    import ipui
    import iptools
    splash.showMessage("Loading Micro-manager...", color=QtCore.Qt.gray)
    app.processEvents()
    import mmanager
    module_dir = os.path.dirname(__file__)
    if FAKE_CAMERA:
        import MMCorePyFake as MMCorePy
        MM_CONFIGURATION_NAME = os.path.join(module_dir, "camera_demo.cfg")
    else:
        import MMCorePy
        MM_CONFIGURATION_NAME = os.path.join(module_dir, "camera_baumer.cfg")
    window = MainWindow()
    window.show()
    splash.finish(window)
    app.exec_()
