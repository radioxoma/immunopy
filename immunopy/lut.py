#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 2014-04-02

@author: Eugene Dvoretsky

Look Up Table generation and applying.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib.pyplot import cm
import cv2


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


def offset_sat(arr, offset):
    """Saturated offset math.
    """
    return cv2.add(arr, offset)


def offset(array, offset, undefined=0):
    """Shift all histogram bins of an uint8 array.

    Add `offset` value (like +=) to all elements of `array`. Statistically it
    leads to all histogram bins shift. Emulates saturated math operations,
    except undefined values filled with `undefined` parameter value.

    offset : int, If > 0, bins will shift from right to left.

    undefined: int, Default value for unknown values.

    Examples:
    >>> arr = np.array([0, 1, 3, 4, 5, 252, 253, 254, 255], dtype=np.uint8)
    >>> offset(arr, offset=5, undefined=0)
    [ 5  6  8  9 10  0  0  0  0]  # Saturated values replaced with 0.
    >>> offset(arr, offset=-3, undefined=0)
    [  0   0   0   1   2 249 250 251 252]  # Truncated values replaced with 0.
    >>> offset(arr, offset=-3, undefined=255)
    [255 255   0   1   2 249 250 251 252]  # Replaced with 255.
    """
    if offset == 0:
        return array
    offset = -offset
    lut = np.arange(256, dtype=np.uint8)  # Some kind of look-up-table.
    lut = np.roll(lut, offset)
    if offset > 0:
        lut[:offset] = undefined
    else:
        lut[offset:] = undefined
    return np.take(lut, array, axis=0)
