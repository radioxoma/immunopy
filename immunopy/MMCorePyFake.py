#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 2014-02-25

@author: radioxoma
"""

import numpy as np
from scipy import ndimage
import cv2


class CMMCore(object):
    """Fake Micro-manager RGB32 camera.
    
    Can return RGB. BGR, BGRA, RGB32.
    """
    def __init__(self):
        super(CMMCore, self).__init__()
        self.RGB = ndimage.imread('image/hdab256.tif')
#         self.RGB = ndimage.imread('image/Ki6720x_blue_filter.tif')
#         self.RGB = ndimage.imread('image/2px_um.tif')
        self.BGR = cv2.cvtColor(self.RGB, cv2.COLOR_RGB2BGR)
        self.BGRA = np.dstack((self.BGR, np.zeros((self.BGR.shape[0], self.BGR.shape[1]), dtype=np.uint8)))
        self.RGB32 = self.BGRA.view(dtype=np.uint32)
    def startContinuousSequenceAcquisition(self, bool_):
        pass
    def loadDevice(self, *device):
        self.cam = ', '.join(device)
        print("Device '%s' loaded" % self.cam)
    def initializeDevice(self, devname):
        print("Device '%s' initialized" % devname)
    def setCameraDevice(self, devname):
        print("Device camera '%s' initialized" % devname)
    def setProperty(self, *props):
        print("Props '%s' setted" % ', '.join([str(k) for k in props]))
    def setCircularBufferMemoryFootprint(self, value):
        pass
    def setROI(self, x, y, w, h):
        print('fake setROI: %d %d %d %d') % (x, y, w, h)
    def enableStderrLog(self, bool_):
        pass
    def getBufferTotalCapacity(self):
        return 0.
    def getImageBufferSize(self):
        return 0.
    def getLastImage(self):
#         print('Last frame incoming!')
        return self.RGB32
    def getImageHeight(self):
        return self.RGB.shape[0]
    def getImageWidth(self):
        return self.RGB.shape[1]
    def stopSequenceAcquisition(self):
        pass
    def reset(self):
        print('MMAdapterFake: Fake cam `%s` reseted.' % self.cam)
