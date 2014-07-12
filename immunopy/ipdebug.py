#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 2013-12-07

@author: Eugene Dvoretsky
"""

import numpy as np
import cv2
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
                plt.subplot(
                    len(segments), len(line), (nline - 1) * len(line) + narr)
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
