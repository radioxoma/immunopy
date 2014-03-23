#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 18 Jan. 2014 г.

@author: radioxoma
"""

import sys
import time
from multiprocessing import Pool
import numpy as np
from scipy import ndimage
import cv2
from skimage.color import separate_stains, hdx_from_rgb
from skimage.feature import peak_local_max
import MMCorePyFake as MMCorePy

import iptools 
from ipdebug import show


MAGNIFICATION = '20'

BLUR = 2
THRESHOLD_SHIFT = 8
PEAK_DISTANCE = 8
MIN_SIZE = 15
MAX_SIZE = 3000


# DEVICE = ['Camera', 'DemoCamera', 'DCam']
# DEVICE = ['Camera', 'OpenCVgrabber', 'OpenCVgrabber']
DEVICE = ['Camera', 'BaumerOptronic', 'BaumerOptronic']


def set_exposure(value):
    mmc.setProperty(DEVICE[0], 'Exposure', int(value))


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


def worker(stain, THRESHOLD_SHIFT, MIN_SIZE, MAX_SIZE):
    """Process each stain.
    
    Return filtered objects and their count.
    """
    stth = iptools.threshold_isodata(stain, shift=THRESHOLD_SHIFT)
    stmask = stain > stth
    stmed = ndimage.filters.median_filter(stmask, size=2)
    stedt = cv2.distanceTransform(
        stmed.view(np.uint8), distanceType=cv2.cv.CV_DIST_L2, maskSize=3)
    st_max = peak_local_max(
        stedt, min_distance=PEAK_DISTANCE, exclude_border=False, indices=False)
    stlabels, stlnum = iptools.watershed_segmentation(stmed, stedt, st_max)
    stfiltered, stfnum = iptools.filter_objects(
        stlabels, num=stlnum, min_size=MIN_SIZE, max_size=MAX_SIZE)
    return stfiltered, stfnum


def process(image):
    rgb = image.copy()
    # Коррекция освещённости
    
    # Размытие
    meaned = cv2.blur(rgb, (BLUR, BLUR))
    
    # Масштаб
    scaled = iptools.rescale(meaned, scale=SCALE)
    
    # Разделение красителей
    hdx = separate_stains(scaled, hdx_from_rgb)
    hem = hdx[:,:,0]
    dab = hdx[:,:,1]
    # xna = hdx[:,:,2]
    
    # MULTICORE -------------------------------------------------------------
    hproc = POOL.apply_async(worker, (hem, THRESHOLD_SHIFT, MIN_SIZE, MAX_SIZE))
    dproc = POOL.apply_async(worker, (dab, THRESHOLD_SHIFT, MIN_SIZE, MAX_SIZE))
    hemfiltered, hemfnum = hproc.get(timeout=5)
    dabfiltered, dabfnum = dproc.get(timeout=5)
#     hemfiltered, hemfnum = worker(hem, THRESHOLD_SHIFT, MIN_SIZE, MAX_SIZE)
#     dabfiltered, dabfnum = worker(dab, THRESHOLD_SHIFT, MIN_SIZE, MAX_SIZE)
    # MULTICORE END ---------------------------------------------------------
    
    # Stats
    stats = 'Num H%dD%d, %.2f' % (hemfnum, dabfnum, float(dabfnum) / (hemfnum + dabfnum + 0.001) * 100)
    stats2 = 'Are %.2f' % (iptools.calc_stats(hemfiltered, dabfiltered) * 100)
    stats3 = 'ArOR %.2f' % (iptools.calc_stats_binary(hemfiltered, dabfiltered) * 100)

    # Visualization
    overlay = iptools.overlay(scaled, dabfiltered, hemfiltered)
    cv2.putText(overlay, stats, (2,25), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 200, 0), thickness=2)
    cv2.putText(overlay, stats2, (2,55), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 200, 0), thickness=2)
    cv2.putText(overlay, stats3, (2,85), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 200, 0), thickness=2)
    cv2.imshow('Overlay', overlay[...,::-1])
#     cv2.imshow('Video', scaled[...,::-1])
#     composite_rgb = np.dstack((hemfiltered, np.zeros_like(hemfiltered), dabfiltered)) # NB! BGR
#     cv2.imshow('Masks', composite_rgb.astype(np.float32))
    

if __name__ == '__main__':
    CMicro = iptools.CalibMicro(MAGNIFICATION)
    SCALE = CMicro.um2px(1)
    print('curscale %f') % CMicro.get_curr_scale()
    print('um2px %f') % SCALE
    
    mmc = MMCorePy.CMMCore()
    print('ImageBufferSize %f' % mmc.getImageBufferSize())  # Returns the size of the internal image buffer.
    print('BufferTotalCapacity %f' % mmc.getBufferTotalCapacity())
    mmc.setCircularBufferMemoryFootprint(60)
    mmc.enableStderrLog(False)
    mmc.enableDebugLog(False)
    mmc.loadDevice(*DEVICE)
    mmc.initializeDevice(DEVICE[0])
    mmc.setCameraDevice(DEVICE[0])
    # mmc.setProperty(DEVICE[0], 'Binning', '2')
    mmc.setProperty(DEVICE[0], 'PixelType', '32bitRGB')
    iptools.set_mmc_resolution(mmc, 1024, 768)
    mmc.snapImage()  # Baumer bug workaround
#     cv2.namedWindow('Video')
    cv2.namedWindow('Overlay')
    cv2.namedWindow('Controls')
    
    cv2.createTrackbar('EXPOSURE', 'Controls', 20, 100, set_exposure)
    cv2.createTrackbar('SHIFT_THRESHOLD', 'Controls', 108, 200, set_threshold_shift)
    cv2.createTrackbar('PEAK_DISTANCE', 'Controls', 8, 100, set_peak_distance)
    cv2.createTrackbar('MAX_SIZE', 'Controls', MAX_SIZE, 5000, set_max_size)
    cv2.createTrackbar('MIN_SIZE', 'Controls', MIN_SIZE, 1000, set_min_size)
    
    POOL = Pool(processes=2)
#   FPS single thread 1.35 > 1.8
    mmc.startContinuousSequenceAcquisition(1)
    while True:
        start_time = time.time()
        rgb32 = mmc.getLastImage()
        if rgb32 is not None:
            process(iptools.rgb32asrgb(rgb32))
        if cv2.waitKey(5) >= 0:
            break
        print('FPS: %f') % (1. / (time.time() - start_time))
    mmc.reset()
    cv2.destroyAllWindows()
