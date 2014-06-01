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

ERROR_ON_COPY = True  # Raise exception on array copy or casting
# http://pyopengl.sourceforge.net/documentation/opengl_diffs.html


class AdjustBar(QtGui.QWidget):
    """Slider and spinbox widget.
    
    dtype - Data type of MM property value: int or float.
    BUG: precision is not enough.
    """
    def __init__(self, minlim, maxlim, dtype, parent=None):
        super(AdjustBar, self).__init__(parent)
        self.parent = parent
        self.mult = 1000.0
        self.maxlim = maxlim
        self.minlim = minlim
        
        self.hbox = QtGui.QHBoxLayout(self.parent)
        # self.hbox.setAlignment(QtCore.Qt.AlignTop)
        self.setLayout(self.hbox)
        self.slid = QtGui.QSlider(QtCore.Qt.Horizontal)
        if dtype is int:
            self.spin = QtGui.QSpinBox()
            self.slid.setRange(int(minlim), int(maxlim))
            self.spin.setRange(int(minlim), int(maxlim))
            self.slid.valueChanged.connect(self.spin.setValue)
            self.spin.valueChanged.connect(self.slid.setValue)
        else:
            self.spin = QtGui.QDoubleSpinBox()
            self.spin.setSingleStep(0.01)
            # Stretch slider
            self.slid.setRange(minlim, (self.maxlim - self.minlim) * self.mult + self.minlim)
            self.spin.setRange(minlim, maxlim)
            self.slid.valueChanged.connect(self.setAsDouble)
            self.spin.valueChanged.connect(self.setAsInt)
        self.hbox.addWidget(self.slid)
        self.hbox.addWidget(self.spin)
        
    @QtCore.Slot(float)
    def setAsInt(self, value):
        target = (value - self.minlim) * self.mult + self.minlim
        self.slid.setValue(target)
    
    @QtCore.Slot(int)
    def setAsDouble(self, value):
        target = (value - self.minlim) / self.mult + self.minlim
        self.spin.setValue(target)


class MicroscopeControl(QtGui.QWidget):
    """Control microscope devices.
    """
    def __init__(self, parent=None):
        super(MicroscopeControl, self).__init__(parent)
        self.parent = parent
        self.vbox = QtGui.QVBoxLayout(self.parent)
        self.vbox.setAlignment(QtCore.Qt.AlignTop)
        self.setLayout(self.vbox)
        self.titl_camn = QtGui.QLabel('Camera name')
        self.titl_magn = QtGui.QLabel('Objective magnification')
        self.titl_expo = QtGui.QLabel('Exposure, ms')
        self.titl_gain = QtGui.QLabel('Gain')
        self.adju_expo = AdjustBar(minlim=1, maxlim=2000, dtype=int, parent=self)
        self.adju_gain = AdjustBar(minlim=1.0, maxlim=4.0, dtype=float, parent=self)
        
        self.slid_expo = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slid_gain = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.spin_expo = QtGui.QSpinBox()
        self.spin_gain = QtGui.QSpinBox()
        self.comb_magn = QtGui.QComboBox()
#         self.comb_reso = QtGui.QComboBox()
        
        self.vbox.addWidget(self.titl_camn)
        self.vbox.addWidget(self.titl_magn)
        self.vbox.addWidget(self.comb_magn)
        
        self.vbox.addWidget(self.titl_expo)
        self.vbox.addWidget(self.adju_expo)
        self.vbox.addWidget(self.titl_gain)
        self.vbox.addWidget(self.adju_gain)
        
        # Set camera name
        self.titl_camn.setText(
            self.parent.mmc.getDeviceDescription(
                self.parent.mmc.getCameraDevice()))
        
        # Get scales and set default.
        self.comb_magn.addItems(self.parent.CMicro.get_all_scalenames())
        self.comb_magn.setCurrentIndex(
            self.comb_magn.findText(self.parent.CMicro.scalename))
        self.comb_magn.currentIndexChanged.connect(self.change_scalename)
    
    @QtCore.Slot(int)
    def change_scalename(self, index):
        self.parent.CMicro.scalename = str(self.comb_magn.currentText())
        
class AnalysisControl(QtGui.QWidget):
    """Control image analysis workflow.
    """
    def __init__(self, parent=None):
        super(AnalysisControl, self).__init__(parent)
        self.parent = parent
        self.vbox = QtGui.QVBoxLayout(self.parent)
        self.vbox.setAlignment(QtCore.Qt.AlignTop)
        self.setLayout(self.vbox)
        
        self.titl_none = QtGui.QLabel('Not implemented')
        self.vbox.addWidget(self.titl_none)


class GLFrame(QtOpenGL.QGLWidget):
    """Video output Qt widget based on OpenGL.
    
    Put image to texture and show it with OpenGL.
    """
    def __init__(self, width, height):
        super(GLFrame, self).__init__()
        self._tex_width, self._tex_height = width, height
        self._tex_data = None
    
    def initializeGL(self):
        """Create an empty texture with uninitialized data.
        """
#         print('initializeGL')
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        # Установим параметры "оборачивания" текстуры
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        # Linear filtering (?)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        # Prepare an empty texture.
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB,
            self._tex_width, self._tex_height,
            0, GL_RGB, GL_UNSIGNED_BYTE, self._tex_data)
        # glDeleteTextures(1, texture_id)

    def paintGL(self):
        """Replace old texture data and show it on screen.
        """
        # print('paintGL')
        # Place new texture data
        # Prevent segfault: glTexSubImage would not accept None.
        if self._tex_data is not None:
            glTexSubImage2D(
                GL_TEXTURE_2D, 0, 0, 0,
                self._tex_width, self._tex_height,
                GL_RGB, GL_UNSIGNED_BYTE, self._tex_data)

        glClearColor(0.4, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_TEXTURE_2D)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex3f(-1, -1, -1)
        glTexCoord2f(1, 1); glVertex3f(1, -1, -1)
        glTexCoord2f(1, 0); glVertex3f(1, 1, -1)
        glTexCoord2f(0, 0); glVertex3f(-1, 1, -1)
        glEnd()
        # glDisable(GL_TEXTURE_2D)

    def resizeGL(self, width, height):
#         print('resizeGL')
        self.view_width, self.view_height = width, height
        glViewport(0, 0, self.view_width, self.view_height)

    def setData(self, array):
        """Set numpy array as new texture to wiget.
        
        Numpy array must be same shape as previous texture.
        """
        assert((self._tex_height, self._tex_width) == array.shape[:2])
        self._tex_data = array
        self.updateGL()


class VideoProcessor(QtCore.QThread):
    """Get frames."""
    def __init__(self, mmcamera):
        super(VideoProcessor, self).__init__()
        self.mmcamera = mmcamera
        self.running = True
        self.rgb32 = None
        self.rgb = None
        
    def run(self):
        self.mmcamera.snapImage()  # Avoid Baumer bug
        self.mmcamera.startContinuousSequenceAcquisition(1)
        
        start_time = time.time()
        while self.running is True:
            if self.mmcamera.getRemainingImageCount() > 0:
                start_time = time.time()
                # self.rgb32 = mmcamera.popNextImage()
                self.rgb32 = self.mmcamera.getLastImage()
                self.rgb = self.rgb32.view(dtype=np.uint8).reshape(
                    self.rgb32.shape[0], self.rgb32.shape[1], 4)[..., 2:: -1]
                self.emit(QtCore.SIGNAL('NewFrame()'))
                print('GET frame')
            else:
                print('No frame')
            time.sleep(0.020)
            print('FPS: %f') % (1. / (time.time() - start_time))

        self.mmcamera.stopSequenceAcquisition()
        self.mmcamera.reset()
        print('Video acquisition terminated.')
        # self.emit(QtCore.SIGNAL('CamReleased'))  # taskDone() may be better
    
    def control(self, trigger):
        """Control thread run.
        """
        if trigger is True:
            self.running = True
            self.start()
        if trigger is False:
            self.running = False
            print('Terminated')


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = AdjustBar(minlim=0, maxlim=2000, dtype=int)
#     window = AdjustBar(minlim=-1.0, maxlim=1.0, dtype=float)
    window.show()
    app.exec_()
