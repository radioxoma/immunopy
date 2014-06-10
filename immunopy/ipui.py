#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 2014-05-16

@author: radioxoma

TODO:
* Pixel buffer
* Correct image rotation
* Correct GL
"""

import sys
import time
import numpy as np
from PySide import QtCore
from PySide import QtGui
from PySide import QtOpenGL
from OpenGL.GL import *
from OpenGL import ERROR_ON_COPY
import iptools
import lut

ERROR_ON_COPY = True  # Raise exception on array copy or casting
# http://pyopengl.sourceforge.net/documentation/opengl_diffs.html


class AdjustBar(QtGui.QWidget):
    """Slider and spinbox widget.
    
    The spinboxes always contain real property value.
    BUG: precision sometimes is not enough.
    """
    def __init__(self, mmcore, prop, parent=None):
        super(AdjustBar, self).__init__(parent)
        self.parent = parent
        self.mmc = mmcore
        self.prop = prop
        self.mult = 1000.0
        self.camname = self.mmc.getCameraDevice()
        self.minlim=self.mmc.getPropertyLowerLimit(self.camname, prop)
        self.maxlim=self.mmc.getPropertyUpperLimit(self.camname, prop)
        
        self.vbox = QtGui.QVBoxLayout(self.parent)
        self.form = QtGui.QFormLayout()
        # self.hbox.setAlignment(QtCore.Qt.AlignTop)
        self.slid = QtGui.QSlider(QtCore.Qt.Horizontal)
        if iptools.get_prop_dtype(self.mmc, self.camname, prop) is int:
            self.spin = QtGui.QSpinBox()
            self.slid.setRange(int(self.minlim), int(self.maxlim))
            self.spin.setRange(int(self.minlim), int(self.maxlim))
            self.spin.setValue(int(self.mmc.getProperty(self.camname, prop)))
            self.slid.valueChanged.connect(self.spin.setValue)
            self.spin.valueChanged.connect(self.slid.setValue)
            self.slid.valueChanged.connect(self.setDevProperty)
        else:
            self.spin = QtGui.QDoubleSpinBox()
            self.spin.setSingleStep(0.01)
            # Stretch slider
            self.slid.setRange(self.minlim,
                (self.maxlim - self.minlim) * self.mult + self.minlim)
            self.spin.setRange(self.minlim, self.maxlim)
            # Prevent comma on Linux.
            self.spin.setValue(float(
                self.mmc.getProperty(self.camname, prop).replace(',', '.')))
            self.slid.valueChanged.connect(self.setAsDouble)
            self.spin.valueChanged.connect(self.setAsInt)
        
        self.form.addRow(QtGui.QLabel(prop), self.spin)
        self.setLayout(self.vbox)
        self.vbox.addLayout(self.form)                    
        self.vbox.addWidget(self.slid)
        
        
    @QtCore.Slot(float)
    def setAsInt(self, value):
        target = (value - self.minlim) * self.mult + self.minlim
        if self.slid.value() != target:
            self.slid.setValue(target)
            self.setDevProperty(value)
    
    @QtCore.Slot(int)
    def setAsDouble(self, value):
        current = round(self.spin.value(), 2)
        target = round((value - self.minlim) / self.mult + self.minlim, 2)
        if current != target:
            self.spin.setValue(target)
    
    @QtCore.Slot()
    def setDevProperty(self, value):
        self.mmc.setProperty(self.camname, self.prop, str(value))


class MicroscopeControl(QtGui.QWidget):
    """Control microscope devices.
    """
    def __init__(self, parent=None):
        super(MicroscopeControl, self).__init__(parent)
        self.parent = parent
        self.vbox = QtGui.QVBoxLayout(self.parent)
        self.vbox.setAlignment(QtCore.Qt.AlignTop)
        self.setLayout(self.vbox)
        
        self.btn_strt = QtGui.QPushButton('Start')
        self.btn_stop = QtGui.QPushButton('Stop')
        self.titl_magn = QtGui.QLabel('Objective magnification')
        self.comb_magn = QtGui.QComboBox()
        self.vbox.addWidget(self.btn_strt)
        self.vbox.addWidget(self.btn_stop)
        self.vbox.addWidget(self.titl_magn)
        self.vbox.addWidget(self.comb_magn)
        
        # Set appropriate camera control widgets.
        camname = self.parent.mmc.getCameraDevice()
        needed_prop = set(self.parent.mmc.getDevicePropertyNames(camname)) & set(('Exposure', 'Gain'))
        for prop in needed_prop:
            if not self.parent.mmc.isPropertyReadOnly(camname, prop) & \
                self.parent.mmc.isPropertySequenceable(camname, prop):
                self.vbox.addWidget(AdjustBar(self.parent.mmc, prop, self))
        self.histview = QtGui.QLabel('Histogram')
        self.histview.setAlignment(QtCore.Qt.AlignCenter)
        self.histview.setMinimumSize(256, 50)
        self.vbox.addWidget(self.histview)
        
        # Get scales and set default.
        self.comb_magn.addItems(self.parent.CMicro.get_all_scalenames())
        self.comb_magn.setCurrentIndex(
            self.comb_magn.findText(self.parent.CMicro.scalename))
        self.comb_magn.currentIndexChanged.connect(self.change_scalename)
        
        self.btn_strt.pressed.connect(self.parent.WorkThread.start)
        self.btn_stop.pressed.connect(self.parent.WorkThread.quit)
        
    @QtCore.Slot(int)
    def change_scalename(self, index):
        self.parent.CMicro.scalename = str(self.comb_magn.currentText())
    
    @QtCore.Slot()
    def setHistogram(self):
        img = self.parent.VProc.hist
        image = QtGui.QImage(img, img.shape[1], img.shape[0], QtGui.QImage.Format_ARGB32)
        self.histview.setPixmap(QtGui.QPixmap(image))


class AnalysisControl(QtGui.QWidget):
    """Control image analysis workflow.
    
    Cell segmentation controls.
    """
    def __init__(self, parent=None):
        super(AnalysisControl, self).__init__(parent)
        self.parent = parent
#         self.vbox = QtGui.QVBoxLayout(self.parent)
#         self.vbox.setAlignment(QtCore.Qt.AlignTop)
#         self.setLayout(self.vbox)
        
        self.form = QtGui.QFormLayout()
#         self.form.setFormAlignment(QtCore.Qt.AlignRight)
        self.setLayout(self.form)
#         self.setLayoutDirection(QtCore.Qt.RightToLeft)
        
        self.vtype = QtGui.QSpinBox()
        self.vtype.setRange(-1, 3)
        self.vtype.setValue(self.parent.VProc.CProcessor.vtype)
        self.form.addRow(QtGui.QLabel('VizType'), self.vtype)
        
        self.sizemax = QtGui.QSpinBox()
        self.sizemax.setSuffix(' px')
        self.sizemax.setRange(0, 9999)
        self.sizemax.setValue(self.parent.VProc.CProcessor.max_size)
        self.form.addRow(QtGui.QLabel('Max size'), self.sizemax)
        
        self.sizemin = QtGui.QSpinBox()
        self.sizemin.setSuffix(' px')
        self.sizemin.setRange(0, 9999)
        self.sizemin.setValue(self.parent.VProc.CProcessor.min_size)
        self.form.addRow(QtGui.QLabel('Min size'), self.sizemin)
        
        self.peak_dist = QtGui.QSpinBox()
        self.peak_dist.setSuffix(' px')
        self.peak_dist.setRange(0, 9999)
        self.peak_dist.setValue(self.parent.VProc.CProcessor.peak_distance)
        self.form.addRow(QtGui.QLabel('Peak distance'), self.peak_dist)
        
        self.shift_th = QtGui.QSpinBox()
        self.shift_th.setSuffix(' %')
        self.shift_th.setRange(-100, 100)
        self.shift_th.setValue(self.parent.VProc.CProcessor.threshold_shift)
        self.form.addRow(QtGui.QLabel('Shift threshold'), self.shift_th)


class GLFrame(QtOpenGL.QGLWidget):
    """OpenGL based video output Qt widget.

    Put RGB image to texture and show it with OpenGL.
    """
    def __init__(self):
        super(GLFrame, self).__init__()
        self._tex_data = None
        self._texture_id = None
        self.rect = QtCore.QRectF(QtCore.QPointF(-1, -1), QtCore.QPointF(1, 1))
        
    def initializeGL(self):
        glClearColor(0.4, 0.1, 0.1, 1.0)
        glEnable(GL_TEXTURE_2D)

    def paintGL(self):
        """Replace old texture data and show it on screen.
        """
        if self._texture_id is not None:
            # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.drawTexture(self.rect, self._texture_id)
            # glBegin(GL_QUADS)
            # glTexCoord2f(0, 1); glVertex3f(-1, -1, -1)
            # glTexCoord2f(1, 1); glVertex3f(1, -1, -1)
            # glTexCoord2f(1, 0); glVertex3f(1, 1, -1)
            # glTexCoord2f(0, 0); glVertex3f(-1, 1, -1)
            # glEnd()

    def resizeGL(self, width, height):
        """Keep aspect ratio in viewport.
        """
        glViewport(0, 0, width, height)
        new_size = QtCore.QSize(self.baseSize())
        new_size.scale(QtCore.QSize(width, height), QtCore.Qt.KeepAspectRatio)
        self.resize(new_size)

    def setData(self, array):
        """Set numpy array as new texture to widget.
        """
        # self.makeCurrent()
        if self._tex_data is not None:
            if self._tex_data.shape == array.shape:
                self._tex_data = array
                # Prevent segfault: glTexSubImage would not accept None.
                glTexSubImage2D(
                    GL_TEXTURE_2D, 0, 0, 0,
                    self._tex_data.shape[1], self._tex_data.shape[0],
                    GL_RGB, GL_UNSIGNED_BYTE, self._tex_data)
            else:
                self.deleteTexture(self._texture_id)
                self.createTex(array)
                self.setBaseSize(array.shape[1], array.shape[0])
                winsize = self.size()
                self.resizeGL(winsize.width(), winsize.height())
        else:
            self.createTex(array)
            self.setBaseSize(array.shape[1], array.shape[0])
            winsize = self.size()
            self.resizeGL(winsize.width(), winsize.height())
        self.updateGL()

    def createTex(self, array):
        """Create texture object for given RGB array.
        """
        self.makeCurrent()
        self._tex_data = array
        # Prepare an empty texture.
        self._texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._texture_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        # Установим параметры "оборачивания" текстуры
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        # Linear filtering (?)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB,
            self._tex_data.shape[1], self._tex_data.shape[0],
            0, GL_RGB, GL_UNSIGNED_BYTE, self._tex_data)


class VideoProcessor(QtCore.QObject):
    """Get frames."""
    newframe = QtCore.Signal()
    histogramready = QtCore.Signal()
    
    def __init__(self, mmcore, parent=None):
        super(VideoProcessor, self).__init__()
        self.parent = parent
        self.mmc = mmcore
        self.CProcessor = iptools.CellProcessor(
            scale=parent.CMicro.scale, colormap=lut.random_jet())
#         self.CProcessor.vtype = -1
        self.HPlotter = iptools.HistogramPlotter(gradient=True)
        self.rgb32 = None
        self.rgb = None
        self.out = None

    @QtCore.Slot()
    def process_frame(self):
        start_time = time.time()
        if self.mmc.getRemainingImageCount() > 0:
            start_time = time.time()
            # self.rgb32 = mmc.popNextImage()
            self.rgb32 = self.mmc.getLastImage()
            self.rgb = iptools.rgb32asrgb(self.rgb32)
            self.hist = self.HPlotter.plot(self.rgb)
            self.histogramready.emit()
            # Return as is, if processing not needed.
            if self.CProcessor.vtype == -1:
                self.out = self.rgb
            else:
                self.out = self.CProcessor.process(self.rgb)
            self.newframe.emit()
        else:
            print('No frame')
        delta_time = time.time() - start_time
        if delta_time != 0:
            print('FPS: %f') % (1. / (time.time() - start_time))

    @QtCore.Slot()
    def start_acquisition(self):
        print('Initialize camera.')
        self.mmc.snapImage()  # Avoid Baumer bug
        self.mmc.startContinuousSequenceAcquisition(1)

    @QtCore.Slot()
    def stop_acquisition(self):
        self.mmc.stopSequenceAcquisition()
        print('Video acquisition terminated.')
        # self.emit(QtCore.SIGNAL('CamReleased'))  # taskDone() may be better
    
    @QtCore.Slot()
    def setVtype(self, value):
        self.CProcessor.vtype = value
    
    @QtCore.Slot()
    def setScale(self, value):
        self.CProcessor.scale = value
    
    @QtCore.Slot()
    def setThresholdShift(self, value):
        self.CProcessor.threshold_shift = value

    @QtCore.Slot()
    def setMinSize(self, value):
        self.CProcessor.min_size = value

    @QtCore.Slot()
    def setMaxSize(self, value):
        self.CProcessor.max_size = value

    @QtCore.Slot()
    def setPeakDistance(self, value):
        self.CProcessor.peak_distance = value


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = AdjustBar(minlim=0, maxlim=2000, dtype=int)
#     window = AdjustBar(minlim=-1.0, maxlim=1.0, dtype=float)
    window.show()
    app.exec_()
