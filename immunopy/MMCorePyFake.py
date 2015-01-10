#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 2014-05-28

@author: Eugene Dvoretsky
"""

import numpy as np
from scipy import misc


try:
    import MMCorePy
    base = MMCorePy.CMMCore
    MM_INSTALLED = True
except ImportError:
    base = object
    MM_INSTALLED = False


class CMMCore(base):
    def __init__(self):
        super(CMMCore, self).__init__()
        # self.RGB = misc.imread('image/hdab256.tif')
        self.RGB = misc.imread('image/Ki6720x_blue_filter.tif')
#         self.RGB = misc.imread('image/2px_um.tif')
        self.BGR = self.RGB[:,:,::-1]
        self.BGRA = np.dstack(
            (self.BGR, np.zeros((self.BGR.shape[0], self.BGR.shape[1]),
            dtype=np.uint8)))
        self.RGB32 = self.BGRA.view(dtype=np.uint32)
        self.frame = self.RGB32
    def getImageHeight(self):
        return self.frame.shape[0]
    def getImageWidth(self):
        return self.frame.shape[1]
    def setROI(self, x, y, w, h):
        print("setROI: %d %d %d %d") % (x, y, w, h)
        if self.RGB32.shape[0] < (y + h) or self.RGB32.shape[1] < (x + w):
            raise ValueError(
                "ROI %d, %d, %dx%d is bigger than image" % (x, y, w, h))
        self.frame = self.RGB32[y:y+h, x:x+w].copy()
    def clearROI(self):
        self.frame = self.RGB32.copy()
    def getLastImage(self):
        return self.frame.copy()
    def getImage(self):
        return self.frame.copy()
    def popNextImage(self):
        return self.frame.copy()
    # If Micromanager isn't installed
    if not MM_INSTALLED:
        print("BAD NEWS")
        def loadSystemConfiguration(self, config_name):
            pass
        def getCameraDevice(self):
            return "Fake camera"
        def startContinuousSequenceAcquisition(self, bool_):
            pass
        def snapImage(self):
            pass
        def loadDevice(self, *device):
            self.input_video = ', '.join(device)
            print("Device '%s' loaded" % self.input_video)
        def initializeDevice(self, devname):
            print("Device '%s' initialized" % devname)
        def setCameraDevice(self, devname):
            print("Device camera '%s' initialized" % devname)
        def hasProperty(self, *props):
            pass
        def setProperty(self, *props):
            print("Props '%s' setted" % ', '.join([str(k) for k in props]))
        def setCircularBufferMemoryFootprint(self, value):
            pass
        def enableStderrLog(self, bool_):
            pass
        def enableDebugLog(self, bool_):
            pass
        def getBufferTotalCapacity(self):
            return 0.
        def getDevicePropertyNames(self, label):
            assert(label == "Fake camera")
            return ("Exposure", "Gain")
        def getImageBufferSize(self):
            return 0.
        def getRemainingImageCount(self):
            return 2
        def stopSequenceAcquisition(self):
            pass
        def reset(self):
            print("MMAdapterFake: Fake input_video `%s` reseted." % self.input_video)
