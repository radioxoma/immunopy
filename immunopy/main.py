#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 18 Jan. 2014 г.

@author: radioxoma
"""

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
MIN_SIZE = 150
MAX_SIZE = 2000


DEVICE = ['Camera', 'DemoCamera', 'DCam']
# DEVICE = ['Camera', 'OpenCVgrabber', 'OpenCVgrabber']
# DEVICE = ['Camera', "BaumerOptronic", "BaumerOptronic"]


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


def main(image):
    rgb = image.copy()
    # Коррекция освещённости
    
    # Размытие
    meaned = cv2.blur(rgb, (BLUR, BLUR))
    
    # Масштаб
    scaled = iptools.rescale(meaned, scale=2)
    
    # Разделение красителей
    hdx = separate_stains(scaled, hdx_from_rgb) # Отрицательные значения!
    hem = hdx[:,:,0]
    dab = hdx[:,:,1]
    xna = hdx[:,:,2]
    
    # threshold (still no correction)
    hemth = iptools.threshold_isodata(hem, shift=THRESHOLD_SHIFT)
    dabth = iptools.threshold_isodata(dab, shift=THRESHOLD_SHIFT)
    hemmask = hem > hemth
    dabmask = dab > dabth
    
    hemmed = ndimage.filters.median_filter(hemmask, size=2)
    dabmed = ndimage.filters.median_filter(dabmask, size=2)
    
    # Начало водораздела
    hemedt = cv2.distanceTransform(hemmed.view(np.uint8), distanceType=cv2.cv.CV_DIST_L2, maskSize=3)
    dabedt = cv2.distanceTransform(dabmed.view(np.uint8), distanceType=cv2.cv.CV_DIST_L2, maskSize=3)
#     hemedt = ndimage.morphology.distance_transform_edt(hemmed)
#     dabedt = ndimage.morphology.distance_transform_edt(dabmed)
    
    hem_maxi = peak_local_max(hemedt, min_distance=PEAK_DISTANCE, exclude_border=False, indices=False)
    dab_maxi = peak_local_max(dabedt, min_distance=PEAK_DISTANCE, exclude_border=False, indices=False)
    
    hemlabels, hemlnum = iptools.watershed_segmentation(hemmed, hemedt, hem_maxi)
    dablabels, dablnum = iptools.watershed_segmentation(dabmed, dabedt, dab_maxi)
    
    hemfiltered, hemfnum = iptools.filter_objects(hemlabels, num=hemlnum, min_size=MIN_SIZE, max_size=MAX_SIZE)
    dabfiltered, dabfnum = iptools.filter_objects(dablabels, num=dablnum, min_size=MIN_SIZE, max_size=MAX_SIZE)
    
    stats = 'H%d, D%d, %.2f' % (hemfnum, dabfnum, float(dabfnum) / (hemfnum + dabfnum + 0.001) * 100)
    stats2 = '%.2f %%' % (iptools.calc_stats(hemfiltered, dabfiltered) * 100)

    cv2.putText(rgb, stats, (2,25), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=255, thickness=2)
    cv2.putText(rgb, stats2, (2,60), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=255, thickness=2)
    
    cv2.imshow('Video', rgb[...,::-1])
    composite_rgb = np.dstack((hemlabels, np.zeros_like(dablabels), dablabels)) # NB! BGR
    cv2.imshow('Masks', composite_rgb.astype(np.float32))


if __name__ == '__main__':
    mmc = MMCorePy.CMMCore()
    print('ImageBufferSize %f' % mmc.getImageBufferSize())  # Returns the size of the internal image buffer.
    print('BufferTotalCapacity %f' % mmc.getBufferTotalCapacity())
    # mmc.setCircularBufferMemoryFootprint(200)
    mmc.enableStderrLog(False)
    mmc.loadDevice(*DEVICE)
    mmc.initializeDevice(DEVICE[0])
    mmc.setCameraDevice(DEVICE[0])
    mmc.setProperty(DEVICE[0], 'PixelType', '32bitRGB')
    # mmc.setROI(0, 0, 256, 256)

    cv2.namedWindow('Video')
    cv2.namedWindow('Masks')
    cv2.namedWindow('Controls')
    
    cv2.createTrackbar('EXPOSURE', 'Controls', 20, 100, set_exposure)
    cv2.createTrackbar('SHIFT_THRESHOLD', 'Controls', 108, 200, set_threshold_shift)
    cv2.createTrackbar('PEAK_DISTANCE', 'Controls', 8, 100, set_peak_distance)
    cv2.createTrackbar('MAX_SIZE', 'Controls', 2000, 5000, set_max_size)
    cv2.createTrackbar('MIN_SIZE', 'Controls', 150, 5000, set_min_size)

    mmc.startContinuousSequenceAcquisition(1)
    while True:
        rgb32 = mmc.getLastImage()
        if rgb32 is not None:
            # Efficient conversion without data copying.
            rgb = iptools.rgb32asrgb(rgb32)
        main(rgb)
        if cv2.waitKey(30) >= 0:
            break
    cv2.destroyAllWindows()
