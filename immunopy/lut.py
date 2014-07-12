#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 2014-04-02

@author: Eugene Dvoretsky

Look Up Table generation and applying.
"""

import numpy as np
from matplotlib.pyplot import cm


def random_jet(bgcolor=(255, 255, 255)):
    """Generate random RGB lut for nice visually object distinguishing.

    Return random uint8 cmap based on jet colors.
    """
    lut = (cm.jet(np.arange(256))[...,:3] * 256).round().astype(np.uint8)
    np.random.shuffle(lut)
    lut[0] = bgcolor
    return lut


def random(bgcolor=(255, 255, 255)):
    """Random RGB colormap."""
    lut = np.random.randint(
        0, 256, size=256 * 3).astype(np.uint8).reshape(256, 3)
    lut[0] = bgcolor
    return lut


def apply_lut(arr, lut):
    """Apply arbitrary look-up-table with numpy.

    See `lut.ipnb`.
    """
    # return np.take(lut, np.mod(arr, 256), axis=0)
    return np.take(lut, arr, axis=0, mode='wrap')
