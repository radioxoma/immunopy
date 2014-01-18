#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 18 янв. 2014 г.

@author: radioxoma
"""

import numpy as np
from scipy import ndimage
import cv2
from skimage.color import separate_stains, hdx_from_rgb
# from iptools import *

MAGNIFICATION = '20'

def process_dab(dab):
    meaned = cv2.boxFilter(to8bit, -1, (2,2)) # I hope this is right filter box size
    

    
    
    
    medmask = cv2.medianBlur(fgmask, 5) # Exactly!
    # Make opening to remove small particles?
    return medmask

if __name__ == '__main__':
    rgb = ndimage.imread('fname', flatten, mode)
    # Коррекция освещённости    
    # Масштаб
    rgbscaled = resize(rgb, um_perpx=2.0)
    
    # Разделение красителей
    hdx = separate_stains(rgbscaled, hdx_from_rgb) # Отрицательные значения!
    hem = hdx[:,:,0]
    dab = hdx[:,:,1]
    xna = hdx[:,:,2]
    
    # threshold
