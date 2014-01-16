#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 07 дек. 2013 г.
@author: radioxoma
"""

import numpy as np
from scipy import ndimage
import cv2

from skimage.color import separate_stains, hdx_from_rgb
# from skimage.exposure import rescale_intensity

import matplotlib.pyplot as plt


def show(*segments):
    """Show all segments in matplotlib window.
    """
    if type(segments) == np.ndarray:
        # Just show array
        if segments.ndim == 3:
            plt.imshow(segments, cmap='gray', interpolation='None')
        else:
            plt.imshow(segments, interpolation='None')
        plt.show()
    elif type(segments[0]) == np.ndarray:
        # Matrices in one line.
        for n, arr in enumerate(segments, 1):
            plt.subplot(1, len(segments), n)
            plt.axis('off')
            plt.imshow(arr, cmap='gray', interpolation='None')
        plt.show()
    else:
        # Matrix of matrices.
        for nline, line in enumerate(segments, 1):
            for narr, arr in enumerate(line, 1):
                # plt.subplot(len(segments), len(line), narr+nline)
                plt.subplot(len(segments), len(line), (nline-1) * len(line)+narr)
                plt.axis('off')
                plt.imshow(arr, interpolation='None')
        plt.show()


def cvshow(img):
    """Show BGR image.
    """
    cv2.namedWindow('cvshow')
    cv2.imshow('cvshow', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def process_dab(dab):
    to8bit = cv2.normalize(dab, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    meaned = cv2.boxFilter(to8bit, -1, (2,2)) # I hope this is right filter box size
    fgmask = cv2.threshold(meaned, thresh=0, maxval=255, type=cv2.THRESH_OTSU)[1]
    medmask = cv2.medianBlur(fgmask, 5) # Exactly!
    # Make opening to remove small particles?
    return medmask


if __name__ == '__main__':
    bgr = cv2.imread('clipboard/Clipboard.tif')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgbscaled = resize(rgb, um_perpx=2.0)
    # Коррекция освещённости
    # Разделение красителей
    hdx = separate_stains(rgbscaled, hdx_from_rgb) # Отрицательные значения!
    _hem = hdx[:,:,0]
    _dab = hdx[:,:,1]
    _xna = hdx[:,:,2]
    ############################################################################

    mask = process_dab(_hem)
    labels = imagej_watershed(mask)
    obj = process_particles(labels, mask)
    objmask = draw_masks(rgb, obj)
#     show(objmask)
    plt.imshow(labels, cmap='flag')
    plt.show()