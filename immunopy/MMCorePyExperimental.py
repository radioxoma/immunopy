#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 2014-05-28

@author: radioxoma
"""

import numpy as np
from scipy import misc
import MMCorePy


class CMMCore(MMCorePy.CMMCore):
    """Fake Micro-manager RGB32 camera.
    """
    def __init__(self):
        super(CMMCore, self).__init__()
        self.RGB = misc.imread('image/hdab256.tif')
#         self.RGB = misc.imread('image/Ki6720x_blue_filter.tif')
#         self.RGB = misc.imread('image/2px_um.tif')
        self.BGR = self.RGB[:,:,::-1]
        self.BGRA = np.dstack((self.BGR, np.zeros((self.BGR.shape[0], self.BGR.shape[1]), dtype=np.uint8)))
        self.RGB32 = self.BGRA.view(dtype=np.uint32)
        self.frame = self.RGB32
    def setROI(self, x, y, w, h):
        print("setROI: %d %d %d %d") % (x, y, w, h)
        if self.RGB32.shape[0] < (y + h) or self.RGB32.shape[1] < (x + w):
            raise ValueError(
                "ROI %d, %d, %dx%d is bigger than image" % (x, y, w, h))
        self.frame = self.RGB32[y:y+h, x:x+w].copy()
    def getLastImage(self):
        return self.frame
    def popNextImage(self):
        return self.frame
