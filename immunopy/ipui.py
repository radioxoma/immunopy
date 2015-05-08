#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 2014-05-16

@author: Eugene Dvoretsky

Immunopy GUI primitives.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import numpy as np
from PySide import QtCore
from PySide import QtGui
from PySide import QtOpenGL
from OpenGL.GL import *
from OpenGL import ERROR_ON_COPY

from . import iptools
from . import lut
from . import statdata

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
        if self.mmc.getPropertyType(self.camname, prop) == 3:  # 3 is int
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

        self.form.addRow(prop, self.spin)
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


class MicroscopeControl(QtGui.QGroupBox):
    """Control microscope devices.

    Aware hardcode for properties is better way.
    """
    willRunOnce = QtCore.Signal()
    willRunContinuously = QtCore.Signal()
    willStop = QtCore.Signal()
    needAutoWb = QtCore.Signal()

    def __init__(self, parent=None):
        super(MicroscopeControl, self).__init__(parent)
        self.parent = parent
        self.setTitle('Microscope control')

        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        self.form = QtGui.QFormLayout()
        self.in_vbox = QtGui.QVBoxLayout()
        self.horizontal = QtGui.QHBoxLayout()
        self.horizontal.setAlignment(QtCore.Qt.AlignLeft)
        self.vbox.addLayout(self.form)
        self.vbox.addLayout(self.in_vbox)
        self.vbox.addLayout(self.horizontal)

        self.streaming_btn = QtGui.QPushButton('&Start')
        self.form.addRow('Acquisition', self.streaming_btn)
        self.streaming_btn.pressed.connect(self.toggle_streaming)

        self.cont_cbx = QtGui.QCheckBox()
        self.form.addRow('Continuous', self.cont_cbx)

        # Get scales and set default.
        self.objective = QtGui.QComboBox()
        self.objective.addItems(self.parent.CMicro.get_all_scalenames())
        self.objective.setCurrentIndex(
            self.objective.findText(self.parent.CMicro.scalename))
        self.form.addRow('Objective', self.objective)
        self.objective.currentIndexChanged.connect(self.change_scalename)

        self.camname = self.parent.mmc.getCameraDevice()
        self.exposure = QtGui.QSpinBox()
        self.exposure.setSuffix(' ms')
        self.exposure.setRange(
            self.parent.mmc.getPropertyLowerLimit(self.camname, 'Exposure'),
            self.parent.mmc.getPropertyUpperLimit(self.camname, 'Exposure'))
        self.exposure.setValue(self.parent.mmc.getExposure())
        self.exposure.valueChanged.connect(self.parent.mmc.setExposure)
        self.form.addRow('Exposure', self.exposure)

        self.gain = QtGui.QDoubleSpinBox()
        self.gain.setSingleStep(0.1)
        self.gain.setRange(
            self.parent.mmc.getPropertyLowerLimit(self.camname, 'Gain'),
            self.parent.mmc.getPropertyUpperLimit(self.camname, 'Gain'))
        self.gain.setValue(float(self.parent.mmc.getProperty(self.camname, 'Gain')))
        self.gain.valueChanged.connect(self.set_gain)
        self.form.addRow('Gain', self.gain)

        self.binning = QtGui.QComboBox()
        self.binning.addItems(self.parent.mmc.getAllowedPropertyValues(self.camname, 'Binning'))
        self.binning.setCurrentIndex(
            self.binning.findText(self.parent.mmc.getProperty(self.camname, 'Binning')))
        self.binning.currentIndexChanged.connect(self.set_binning)
        self.form.addRow('Binning', self.binning)

        self.histview = QtGui.QLabel('Histogram')
        self.histview.setAlignment(QtCore.Qt.AlignCenter)
        self.histview.setFixedSize(256, 100)
        self.in_vbox.addWidget(self.histview)

        self.horizontal.addWidget(QtGui.QLabel('R'))
        self.sbx_adjust_r = QtGui.QDoubleSpinBox()
        self.sbx_adjust_r.setSingleStep(0.01)
        self.sbx_adjust_r.setRange(-2.0, 2.0)
        self.horizontal.addWidget(self.sbx_adjust_r)
        self.horizontal.addWidget(QtGui.QLabel('G'))
        self.sbx_adjust_g = QtGui.QDoubleSpinBox()
        self.sbx_adjust_g.setSingleStep(0.01)
        self.sbx_adjust_g.setRange(-2.0, 2.0)
        self.horizontal.addWidget(self.sbx_adjust_g)
        self.horizontal.addWidget(QtGui.QLabel('B'))
        self.sbx_adjust_b = QtGui.QDoubleSpinBox()
        self.sbx_adjust_b.setSingleStep(0.01)
        self.sbx_adjust_b.setRange(-2.0, 2.0)
        self.horizontal.addWidget(self.sbx_adjust_b)
        self.btn_autowb = QtGui.QPushButton('Auto')
        self.btn_autowb.setToolTip(
            "Please remove slice and click the button")
        self.btn_autowb.setStyleSheet("padding: 3px;")
        self.horizontal.addWidget(self.btn_autowb)
        self.btn_resetwb = QtGui.QPushButton('Reset')
        self.btn_resetwb.setToolTip("Reset channels shifts to zero")
        self.btn_resetwb.setStyleSheet("padding: 3px;")
        self.btn_resetwb.clicked.connect(self.resetWbControls)
        self.horizontal.addWidget(self.btn_resetwb)
        self.btn_autowb.clicked.connect(self.autowb)
        self.updateWbControls()

        self.willRunOnce.connect(self.parent.VProc.runOnce)
        self.willRunContinuously.connect(self.parent.VProc.runContinuous)
        self.willStop.connect(self.parent.VProc.stop)

    @QtCore.Slot()
    def toggle_streaming(self):
        if not self.parent.mmc.isSequenceRunning():
            if self.cont_cbx.checkState() == QtCore.Qt.Checked:
                self.willRunContinuously.emit()
                self.streaming_btn.setText('&Stop')
                self.cont_cbx.setEnabled(False)
                self.binning.setEnabled(False)
            else:
                self.willRunOnce.emit()
        else:
            self.willStop.emit()
            self.streaming_btn.setText('&Start')
            self.cont_cbx.setEnabled(True)
            self.binning.setEnabled(True)

    @QtCore.Slot(int)
    def change_scalename(self, index):
        self.parent.CMicro.scalename = str(self.objective.currentText())

    @QtCore.Slot(float)
    def set_gain(self, value):
        self.parent.mmc.setProperty(self.camname, 'Gain', str(value))

    @QtCore.Slot(int)
    def set_binning(self, index):
        value = self.binning.itemText(index)
        self.parent.mmc.setProperty(self.camname, 'Binning', str(value))

    @QtCore.Slot()
    def setHistogram(self):
        img = self.parent.VProc.hist
        image = QtGui.QImage(img, img.shape[1], img.shape[0], QtGui.QImage.Format_ARGB32)
        self.histview.setPixmap(QtGui.QPixmap(image))

    @QtCore.Slot()
    def updateWbControls(self):
        r, g, b = self.parent.VProc.get_white_point()
        self.sbx_adjust_r.setValue(r)
        self.sbx_adjust_g.setValue(g)
        self.sbx_adjust_b.setValue(b)

    def resetWbControls(self):
        self.sbx_adjust_r.setValue(1.0)
        self.sbx_adjust_g.setValue(1.0)
        self.sbx_adjust_b.setValue(1.0)
        self.btn_autowb.setEnabled(True)

    def autowb(self):
        self.btn_autowb.setEnabled(False)
        self.needAutoWb.emit()


class AnalysisControl(QtGui.QGroupBox):
    """Control image analysis workflow.

    Cell segmentation controls.
    """
    def __init__(self, parent=None):
        super(AnalysisControl, self).__init__(parent)
        self.parent = parent
        self.setTitle('Analysis control')
        self.vbox = QtGui.QVBoxLayout()
        self.vbox.setAlignment(QtCore.Qt.AlignTop)
        self.setLayout(self.vbox)
        self.in_vbox = QtGui.QVBoxLayout()

        self.form = QtGui.QFormLayout()
        self.vbox.addLayout(self.in_vbox)
        self.vbox.addLayout(self.form)

#         self.cont_cbx = QtGui.QCheckBox()
#         self.form.addRow('Analyze', self.cont_cbx)

        self.vtype = QtGui.QComboBox()
        self.vtype.addItems(self.parent.VProc.CProcessor.vtypes)
        self.vtype.setCurrentIndex(
            self.vtype.findText(self.parent.VProc.CProcessor.vtype))
        self.form.addRow('VizType', self.vtype)

        self.sizemax = QtGui.QSpinBox()
        self.sizemax.setSuffix(' px')
        self.sizemax.setRange(0, 9999999)
        self.sizemax.setValue(self.parent.VProc.CProcessor.max_size)
        self.form.addRow('Max size', self.sizemax)

        self.sizemin = QtGui.QSpinBox()
        self.sizemin.setSuffix(' px')
        self.sizemin.setRange(0, 9999)
        self.sizemin.setValue(self.parent.VProc.CProcessor.min_size)
        self.form.addRow('Min size', self.sizemin)

        self.peak_dist = QtGui.QSpinBox()
        self.peak_dist.setSuffix(' px')
        self.peak_dist.setRange(0, 9999)
        self.peak_dist.setValue(self.parent.VProc.CProcessor.peak_distance)
        self.form.addRow('Peak distance', self.peak_dist)

        self.dab_th_shift = QtGui.QSpinBox()
        self.dab_th_shift.setSuffix(' %')
        self.dab_th_shift.setRange(-100, 100)
        self.dab_th_shift.setValue(self.parent.VProc.CProcessor.th_dab_shift)
        self.form.addRow('DAB threshold shift', self.dab_th_shift)

        self.hem_th_shift = QtGui.QSpinBox()
        self.hem_th_shift.setSuffix(' %')
        self.hem_th_shift.setRange(-100, 100)
        self.hem_th_shift.setValue(self.parent.VProc.CProcessor.th_hem_shift)
        self.form.addRow('HEM threshold shift', self.hem_th_shift)


class GLFrame(QtOpenGL.QGLWidget):
    """OpenGL based video output Qt widget.

    Put RGB image to texture and show it with OpenGL.

    * Разрешение Viewport определяется размером окна
        * Соотношение сторон фиксированное 4:3 (зависит от `setBaseSize()` при установке текстуры)
    """
    def __init__(self):
        super(GLFrame, self).__init__()
        self._tex_data = None
        self._texture_id = None
        self.rect = QtCore.QRectF(-1, -1, 2, 2)  # x, y, w, h

    def initializeGL(self):
        glClearColor(0.85, 0.85, 0.85, 1.0)  # Like Qt background
        glEnable(GL_TEXTURE_2D)

    def paintGL(self):
        """Replace old texture data and show it on screen.
        """
        if self._texture_id is not None:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.drawTexture(self.rect, self._texture_id)
            # glBegin(GL_QUADS)
            # glTexCoord2f(0, 0); glVertex3f(-1.0, 1.0, 0.0);  # Top left (w,h,d)
            # glTexCoord2f(1, 0); glVertex3f( 1.0, 1.0, 0.0);  # Top right
            # glTexCoord2f(1, 1); glVertex3f( 1.0,-1.0, 0.0);  # Bottom right
            # glTexCoord2f(0, 1); glVertex3f(-1.0,-1.0, 0.0);  # Bottom left
            # glEnd()

    def resizeGL(self, width, height):
        """Keep aspect ratio in viewport.
        """
        widget_size = self.baseSize()
        widget_size.scale(width, height, QtCore.Qt.KeepAspectRatio)
        glViewport(0, 0, widget_size.width(), widget_size.height())
#         self.resize(widget_size)

    def setData(self, array):
        """Set numpy array as new texture to widget.
        """
        # self.makeCurrent()
        if self._tex_data is not None:
            if (self._tex_data.shape == array.shape and self._tex_data.dtype == array.dtype):
                self.update_texture(array)
            else:
                self.deleteTexture(self._texture_id)
                self.create_texture(array)
                self.update_widget_size()
        else:
            self.create_texture(array)
            self.update_widget_size()
        self.updateGL()

    def create_texture(self, array):
        """Create texture object for given RGB or grayscale uint8 array.
        """
        self.makeCurrent()
        # Update texture properties
        self._tex_data = array
        if len(self._tex_data.shape) == 3:
            self._tex_color = GL_RGB
        elif len(self._tex_data.shape) == 2:
            self._tex_color = GL_LUMINANCE

        if self._tex_data.dtype == np.uint8:
            self._tex_dtype = GL_UNSIGNED_BYTE
        elif self._tex_data.dtype == np.float32:
            self._tex_dtype = GL_FLOAT
        else:
            raise ValueError("{} dtype is not supported, "
                "use uint8 or float32 instead".format(array.dtype))

        # Prepare an empty texture
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
            0, self._tex_color, self._tex_dtype, self._tex_data)

    def update_texture(self, array):
        # Prevent segfault: glTexSubImage would not accept None.
        self.makeCurrent()
        self._tex_data = array
        glTexSubImage2D(
            GL_TEXTURE_2D, 0, 0, 0,
            self._tex_data.shape[1], self._tex_data.shape[0],
            self._tex_color, self._tex_dtype, self._tex_data)

    def update_widget_size(self):
        self.setBaseSize(self._tex_data.shape[1], self._tex_data.shape[0])
        winsize = self.size()
        self.resizeGL(winsize.width(), winsize.height())


class VideoWidget(QtGui.QWidget):
    """Video output with OpenGL and size controls.
    """
    def __init__(self, parent=None):
        super(VideoWidget, self).__init__()
        self.parent = parent
        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)
        self.bar = QtGui.QToolBar('ToolBar')

        self.scrollableView = QtGui.QScrollArea()
        self.glWidget = GLFrame()
        self.glWidget.setSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
        self.scrollableView.setWidget(self.glWidget)
        self.vbox.addWidget(self.bar)
        self.vbox.addWidget(self.scrollableView)

        self.frameResNatural = self.bar.addAction('1:1', self.resNatural)
#         self.frameResPlus = self.bar.addAction('+', self.resPlus)
#         self.frameResMinus = self.bar.addAction(u'−', self.resMinus)
        self.frameResFit = self.bar.addAction('Fit', self.resFit)
        self.resFit()  # Fit viewport on start

    @QtCore.Slot()
    def resNatural(self):
        self.scrollableView.setWidgetResizable(False)
        self.glWidget.resize(self.glWidget.baseSize())

    @QtCore.Slot()
    def resPlus(self):
        print("resPlus")

    @QtCore.Slot()
    def resMinus(self):
        print("resMinus")

    @QtCore.Slot()
    def resFit(self):
        """Fit in scrollableView size.
        """
        self.scrollableView.setWidgetResizable(True)
        widget_size = self.glWidget.baseSize()
        widget_size.scale(self.scrollableView.size(), QtCore.Qt.KeepAspectRatio)
        self.scrollableView.resize(widget_size)

    def setData(self, array):
        self.glWidget.setData(array)


class VideoProcessor(QtCore.QObject):
    """Get frames and process it. Should live in separate thread."""
    newframe = QtCore.Signal()
    histogramready = QtCore.Signal()
    modelGotAssay = QtCore.Signal()
    newwhitepoint = QtCore.Signal()

    def __init__(self, mmcore, parent=None):
        super(VideoProcessor, self).__init__()
        self.parent = parent
        self.mmc = mmcore
        self.CProcessor = iptools.CellProcessor(
            scale=parent.CMicro.scale, colormap=lut.random_jet(), mp=True)
        self.HPlotter = iptools.HistogramPlotter(gradient=True)
        self.__model = statdata.StatDataModel()
        self.rgb32 = None
        self.rgb = None
        self._wb_gain = [1.0, 1.0, 1.0]
        self.out = None

        self.workTimer = QtCore.QTimer(parent=self)
        self.workTimer.setInterval(20)
        self.workTimer.timeout.connect(self.process_frame)
        self.__singleshot = False  # Snap one image flag
        self.__lock = QtCore.QMutex()

    @QtCore.Slot()
    def set_white_point(self):
        rgb_gain = self.HPlotter.get_wp_gain(normalize=False)
        if rgb_gain is not None:
            self._wb_gain = rgb_gain
            print("New white point: {}".format(str(self._wb_gain)))
            self.newwhitepoint.emit()

    @QtCore.Slot()
    def get_white_point(self):
        return self._wb_gain

    @QtCore.Slot()
    def process_frame(self):
        """Retrieve frame from buffer and process it.
        """
        start_time = time.time()
        with QtCore.QMutexLocker(self.__lock):
            if self.__singleshot:
                self.rgb32 = self.mmc.getImage()
                self.__singleshot = False
            else:
                if self.mmc.getRemainingImageCount() > 0:
                    self.rgb32 = self.mmc.getLastImage()
                else:
                    print('No frame')
            if self.rgb32 is not None:
                rgb = iptools.rgb32asrgb(self.rgb32)
                # WB correction before histogram calculation
                self.rgb = iptools.correct_wb(rgb, self._wb_gain)
                self.hist = self.HPlotter.plot(self.rgb)
                self.histogramready.emit()
                self.out = self.CProcessor.process(self.rgb)
                self.newframe.emit()
                delta_time = time.time() - start_time
                if delta_time != 0:
                    print('FPS: %f' % (1. / (time.time() - start_time)))

    @QtCore.Slot()
    def runOnce(self):
        print('Take one picture.')
        if self.workTimer.isActive():
            raise RuntimeWarning('Timer must be stopped before runOnce()!')
        self.mmc.snapImage()
        self.__singleshot = True
        self.process_frame()

    @QtCore.Slot()
    def runContinuous(self):
        print('Start taking pictures continuously')
        if self.workTimer.isActive():
            raise RuntimeWarning('Timer must be stopped before runContinuous()!')
        self.mmc.snapImage()  # Avoid Baumer bug
        self.mmc.startContinuousSequenceAcquisition(1)
        self.workTimer.start()

    @QtCore.Slot()
    def pushAssay(self):
        """Safely save statistics and image to StatModel.
        """
        with QtCore.QMutexLocker(self.__lock):
            if self.__model.isSaveImage:
                self.__model.appendAssay(self.CProcessor.take_assay(),
                    image=self.rgb)
            else:
                self.__model.appendAssay(self.CProcessor.take_assay())
            self.modelGotAssay.emit()

    def getModel(self):
        return self.__model

    @QtCore.Slot()
    def stop(self):
        self.workTimer.stop()
        self.mmc.stopSequenceAcquisition()
        print('Video acquisition terminated.')

    @QtCore.Slot()
    def setVtype(self, value):
        print(value)
        print(self.CProcessor.vtypes[value])
        self.CProcessor.vtype = self.CProcessor.vtypes[value]

    @QtCore.Slot()
    def setScale(self, value):
        self.CProcessor.scale = value

    @QtCore.Slot()
    def setDabThresholdShift(self, value):
        self.CProcessor.th_dab_shift = value

    @QtCore.Slot()
    def setHemThresholdShift(self, value):
        self.CProcessor.th_hem_shift = value

    @QtCore.Slot()
    def setMinSize(self, value):
        self.CProcessor.min_size = value

    @QtCore.Slot()
    def setMaxSize(self, value):
        self.CProcessor.max_size = value

    @QtCore.Slot()
    def setPeakDistance(self, value):
        self.CProcessor.peak_distance = value

    @QtCore.Slot()
    def setRShift(self, value):
        self._wb_gain[0] = value

    @QtCore.Slot()
    def setGShift(self, value):
        self._wb_gain[1] = value

    @QtCore.Slot()
    def setBShift(self, value):
        self._wb_gain[2] = value


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = AdjustBar(minlim=0, maxlim=2000, dtype=int)
#     window = AdjustBar(minlim=-1.0, maxlim=1.0, dtype=float)
    window.show()
    app.exec_()
