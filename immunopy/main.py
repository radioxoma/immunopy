#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 18 Jan. 2014 г.

@author: radioxoma
"""

import time
from multiprocessing import Pool
import numpy as np
from scipy import ndimage
import cv2
from skimage.color import separate_stains, hdx_from_rgb
from skimage.feature import peak_local_max
import MMCorePyFake as MMCorePy

import iptools
import lut


MAGNIFICATION = '10'

# DEVICE = ['Camera', 'DemoCamera', 'DCam']
# DEVICE = ['Camera', 'OpenCVgrabber', 'OpenCVgrabber']
DEVICE = ['Camera', 'BaumerOptronic', 'BaumerOptronic']


def set_threshold_shift(value):
    CProcessor.threshold_shift = value - 100


def set_peak_distance(value):
    CProcessor.peak_distance = value


def set_max_size(value):
    CProcessor.max_size = value


def set_min_size(value):
    CProcessor.min_size = value


def set_vtype(value):
    """Type of visualization method (overlay or labels)."""
    CProcessor.vtype = value


class CellProcessor(object):
    """Segment and visualize cell image."""
    def __init__(self, scale, pool=None):
        super(CellProcessor, self).__init__()
        self.threshold_shift = 20
        self.min_size = 10
        self.max_size = 3000
        self.vtype = 1

        self.peak_distance = 8
        self.scale = scale
        self.blur = 2
        self.pool = pool

    @property
    def vtype(self):
        return self._vtype
    @vtype.setter
    def vtype(self, value):
        self._vtype = value

    @property
    def threshold_shift(self):
        return self._threshold_shift
    @threshold_shift.setter
    def threshold_shift(self, value):
        self._threshold_shift = value

    @property
    def min_size(self):
        return self._min_size
    @min_size.setter
    def min_size(self, value):
        if value > 0:
            self._min_size = value

    @property
    def max_size(self):
        return self._max_size
    @max_size.setter
    def max_size(self, value):
        if value > 0:
            self._max_size = value

    @property
    def peak_distance(self):
        return self._peak_distance
    @peak_distance.setter
    def peak_distance(self, value):
        if value > 0:
            self._peak_distance = value

    def process(self, image):
        """Segmentation and statistical calculation.
        """

        rgb = image.copy()
        # Коррекция освещённости

        # Размытие
        meaned = cv2.blur(rgb, (self.blur, self.blur))

        # Масштаб
        scaled = iptools.rescale(meaned, self.scale)

        # Разделение красителей
        hdx = separate_stains(scaled, hdx_from_rgb)
        hem = hdx[:,:,0]
        dab = hdx[:,:,1]

        # MULTICORE -------------------------------------------------------------
        if self.pool is None:
            hemfiltered, hemfnum = worker(hem, self.threshold_shift, self.peak_distance, self.min_size, self.max_size)
            dabfiltered, dabfnum = worker(dab, self.threshold_shift + 10, self.peak_distance, self.min_size, self.max_size)
        else:
            hproc = self.pool.apply_async(worker, (hem, self.threshold_shift, self.peak_distance, self.min_size, self.max_size))
            dproc = self.pool.apply_async(worker, (dab, self.threshold_shift + 10, self.peak_distance, self.min_size, self.max_size))
            hemfiltered, hemfnum = hproc.get(timeout=5)
            dabfiltered, dabfnum = dproc.get(timeout=5)
        # MULTICORE END ---------------------------------------------------------

        # Stats
        stats = 'Num D%3.d/H%3.d, %.2f' % (dabfnum, hemfnum, float(dabfnum) / (hemfnum + dabfnum + 0.001) * 100)
        stats2 = 'Area fract %.2f' % (iptools.calc_stats(hemfiltered, dabfiltered) * 100)
        stats3 = 'Ar disj %.2f' % (iptools.calc_stats_binary(hemfiltered, dabfiltered) * 100)

        # Visualization
        if self.vtype == 0:
            overlay = scaled
        elif self.vtype == 1:
            overlay = iptools.overlay(scaled, dabfiltered, hemfiltered)
        elif self.vtype == 2:
            overlay = lut.apply_lut(dabfiltered, LUT)
        else:
            overlay = lut.apply_lut(hemfiltered, LUT)
        cv2.putText(overlay, stats, (2,25), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
        cv2.putText(overlay, stats2, (2,55), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
        cv2.putText(overlay, stats3, (2,85), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
        return overlay


def worker(stain, threshold_shift, peak_distance, min_size, max_size):
    """Process each stain.

    Return filtered objects and their count.
    Would not work with processes as class method.
    """
    stth = iptools.threshold_isodata(stain, shift=threshold_shift)
    stmask = stain > stth
    stmed = ndimage.filters.median_filter(stmask, size=2)
    stedt = cv2.distanceTransform(
        stmed.view(np.uint8), distanceType=cv2.cv.CV_DIST_L2, maskSize=3)
    st_max = peak_local_max(
        stedt, min_distance=peak_distance, exclude_border=False, indices=False)
    stlabels, stlnum = iptools.watershed_segmentation(stmed, stedt, st_max)
    stfiltered, stfnum = iptools.filter_objects(
        stlabels, stlnum, min_size, max_size)
    return stfiltered, stfnum


if __name__ == '__main__':
    CMicro = iptools.CalibMicro(MAGNIFICATION)
    SCALE = CMicro.um2px(1)
    LUT = lut.random_jet()
    POOL = Pool(processes=2)
    CProcessor = CellProcessor(scale=SCALE, pool=POOL)
    print('curscale %f') % CMicro.get_curr_scale()
    print('um2px %f') % SCALE

    mmc = MMCorePy.CMMCore()
    print('ImageBufferSize %f' % mmc.getImageBufferSize())  # Returns the size of the internal image buffer.
    print('BufferTotalCapacity %f' % mmc.getBufferTotalCapacity())
    mmc.setCircularBufferMemoryFootprint(100)
    mmc.enableStderrLog(False)
    mmc.enableDebugLog(False)
    mmc.loadDevice(*DEVICE)
    mmc.initializeDevice(DEVICE[0])
    mmc.setCameraDevice(DEVICE[0])
    # mmc.setProperty(DEVICE[0], 'Binning', '2')
    mmc.setProperty(DEVICE[0], 'PixelType', '32bitRGB')
    iptools.set_mmc_resolution(mmc, 500, 500)
    mmc.snapImage()  # Baumer bug workaround
    # mmc.initializeCircularBuffer()
    cv2.namedWindow('Overlay')
    cv2.namedWindow('Controls', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_AUTOSIZE)
    if mmc.hasProperty(DEVICE[0], 'Gain'):
        cv2.createTrackbar(
            'Gain', 'Controls',
            int(float(mmc.getProperty(DEVICE[0], 'Gain'))),
            int(mmc.getPropertyUpperLimit(DEVICE[0], 'Gain')),
            lambda value: mmc.setProperty(DEVICE[0], 'Gain', value))
    if mmc.hasProperty(DEVICE[0], 'Exposure'):
        cv2.createTrackbar(
            'Exposure', 'Controls',
            int(float(mmc.getProperty(DEVICE[0], 'Exposure'))),
            100,  # int(mmc.getPropertyUpperLimit(DEVICE[0], 'Exposure')),
            lambda value: mmc.setProperty(DEVICE[0], 'Exposure', int(value)))
    cv2.createTrackbar('SHIFT_THRESHOLD', 'Controls', CProcessor.threshold_shift + 100, 200, set_threshold_shift)
    cv2.createTrackbar('PEAK_DISTANCE', 'Controls', CProcessor.peak_distance, 100, set_peak_distance)
    cv2.createTrackbar('MAX_SIZE', 'Controls', CProcessor.max_size, 5000, set_max_size)
    cv2.createTrackbar('MIN_SIZE', 'Controls', CProcessor.min_size, 1000, set_min_size)
    cv2.createTrackbar('VMethod', 'Controls', CProcessor.vtype, 3, set_vtype)

    mmc.startContinuousSequenceAcquisition(1)
    while True:
        start_time = time.time()
        if mmc.getRemainingImageCount() > 0:
            rgb32 = mmc.getLastImage()
            cv2.imshow(
                'Overlay',
                CProcessor.process(iptools.rgb32asrgb(rgb32)[...,::-1]))
        if cv2.waitKey(5) >= 0:
            break
        print('FPS: %f') % (1. / (time.time() - start_time))
    mmc.reset()
    cv2.destroyAllWindows()
