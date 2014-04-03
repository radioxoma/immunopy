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

THRESHOLD_SHIFT = 20
PEAK_DISTANCE = 8
MIN_SIZE = 10
MAX_SIZE = 3000
VTYPE = 1


# DEVICE = ['Camera', 'DemoCamera', 'DCam']
# DEVICE = ['Camera', 'OpenCVgrabber', 'OpenCVgrabber']
DEVICE = ['Camera', 'BaumerOptronic', 'BaumerOptronic']


def set_threshold_shift(value):
    global THRESHOLD_SHIFT
    THRESHOLD_SHIFT = value - 100


def set_peak_distance(value):
    global PEAK_DISTANCE
    if value > 0:
        PEAK_DISTANCE = value


def set_max_size(value):
    global MAX_SIZE
    if value > 0:
        MAX_SIZE = value
        
        
def set_min_size(value):
    global MIN_SIZE
    if value > 0:
        MIN_SIZE = value


def set_vtype(value):
    """Type of visualization method (overlay or labels)."""
    global VTYPE
    VTYPE = value


def process(image, scale, threshold_shift, peak_distance, min_size, max_size):
    """Segmentation and statistical calculation.
    """
    def worker(stain, threshold_shift, peak_distance, min_size, max_size):
        """Process each stain.
        
        Return filtered objects and their count.
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
    
    rgb = image.copy()
    # Коррекция освещённости
    
    # Размытие
    BLUR = 2
    meaned = cv2.blur(rgb, (BLUR, BLUR))
    
    # Масштаб
    scaled = iptools.rescale(meaned, scale)
    
    # Разделение красителей
    hdx = separate_stains(scaled, hdx_from_rgb)
    hem = hdx[:,:,0]
    dab = hdx[:,:,1]
    # xna = hdx[:,:,2]
    
    # MULTICORE -------------------------------------------------------------
#     hproc = POOL.apply_async(worker, (hem, threshold_shift, peak_distance, min_size, max_size))
#     dproc = POOL.apply_async(worker, (dab, threshold_shift, peak_distance, min_size, max_size))
#     hemfiltered, hemfnum = hproc.get(timeout=5)
#     dabfiltered, dabfnum = dproc.get(timeout=5)
    hemfiltered, hemfnum = worker(hem, threshold_shift, peak_distance, min_size, max_size)
    dabfiltered, dabfnum = worker(dab, threshold_shift+10, peak_distance, min_size, max_size)
    # MULTICORE END ---------------------------------------------------------
    
    # Stats
    stats = 'Num D%3.d/H%3.d, %.2f' % (dabfnum, hemfnum, float(dabfnum) / (hemfnum + dabfnum + 0.001) * 100)
    stats2 = 'Are %.2f' % (iptools.calc_stats(hemfiltered, dabfiltered) * 100)
    stats3 = 'ArOR %.2f' % (iptools.calc_stats_binary(hemfiltered, dabfiltered) * 100)

    # Visualization
    if VTYPE == 0:
        overlay = scaled
    elif VTYPE == 1:
        overlay = iptools.overlay(scaled, dabfiltered, hemfiltered)
    elif VTYPE == 2:
        overlay = lut.apply_lut(dabfiltered, LUT)
    else:
        overlay = lut.apply_lut(hemfiltered, LUT)
    cv2.putText(overlay, stats, (2,25), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    cv2.putText(overlay, stats2, (2,55), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    cv2.putText(overlay, stats3, (2,85), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return overlay
    

if __name__ == '__main__':
    CMicro = iptools.CalibMicro(MAGNIFICATION)
    SCALE = CMicro.um2px(1)
    LUT = lut.random_jet()
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
    cv2.createTrackbar('SHIFT_THRESHOLD', 'Controls', 100+THRESHOLD_SHIFT, 200, set_threshold_shift)
    cv2.createTrackbar('PEAK_DISTANCE', 'Controls', 8, 100, set_peak_distance)
    cv2.createTrackbar('MAX_SIZE', 'Controls', MAX_SIZE, 5000, set_max_size)
    cv2.createTrackbar('MIN_SIZE', 'Controls', MIN_SIZE, 1000, set_min_size)
    cv2.createTrackbar('VMethod', 'Controls', VTYPE, 3, set_vtype)
    
    POOL = Pool(processes=2)
    mmc.startContinuousSequenceAcquisition(1)
    while True:
        start_time = time.time()
        if mmc.getRemainingImageCount() > 0:
            rgb32 = mmc.getLastImage()
            cv2.imshow('Overlay', process(
                iptools.rgb32asrgb(rgb32),
                SCALE,
                THRESHOLD_SHIFT,
                PEAK_DISTANCE,
                MIN_SIZE,
                MAX_SIZE)[...,::-1])
        if cv2.waitKey(5) >= 0:
            break
        print('FPS: %f') % (1. / (time.time() - start_time))
    mmc.reset()
    cv2.destroyAllWindows()
