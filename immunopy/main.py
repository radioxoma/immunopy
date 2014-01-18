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
import iptools 
from ipdebug import show

MAGNIFICATION = '20'
BLUR = 2
PEAK_DISTANCE = 8

if __name__ == '__main__':
    rgb = ndimage.imread('image/hdab256.tif')
    
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
    
    # threshold
    hemth = iptools.threshold_isodata(hem)
    dabth = iptools.threshold_isodata(dab)
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
    
    hemfiltered, hemfnum = iptools.filter_objects(hemlabels, hemlnum)
    dabfiltered, dabfnum = iptools.filter_objects(dablabels, dablnum)
    
#     show(dablabels)
#     show(dabfiltered)