#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 2014-12-20

@author: radioxoma
"""

import numpy as np
from scipy import linalg
from skimage.util import dtype


# Vectors for images from Human Protein Atlas
hpa_rgb_from_hdx = np.array([[0.684, 0.696, 0.183],
                             [0.250, 0.500, 0.850],
                             [  0.0,   0.0,   0.0]])
hpa_rgb_from_hdx[2, :] = np.cross(hpa_rgb_from_hdx[0, :], hpa_rgb_from_hdx[1, :])
hpa_hdx_from_rgb = linalg.inv(hpa_rgb_from_hdx)


def fastdot(A, B):
    """Uses blas libraries directly to perform dot product.
    
    Numpy blas-optimized
    http://www.huyng.com/posts/faster-numpy-dot-product/
    """
    def _force_forder(x):
        """Converts arrays x to fortran order.
        
        Returns a tuple in the form (x, is_transposed).
        """
        if x.flags.c_contiguous:
            return (x.T, True)
        else:
            return (x, False)
    
    A, trans_a = _force_forder(A)
    B, trans_b = _force_forder(B)
    gemm_dot = linalg.get_blas_funcs("gemm", arrays=(A, B))
    
    # gemm is implemented to compute: C = alpha*AB  + beta*C
    return gemm_dot(alpha=1.0, a=A, b=B, trans_a=trans_a, trans_b=trans_b)


def color_deconvolution(rgb, conv_matrix):
    """Unmix stains for histogram analysis.
    :return: Image values in normal space (not optical density i.e. log space)
             and in range 0...1.
    :rtype: float array
    """
    rgb = dtype.img_as_float(rgb, force_copy=True)
    rgb += 1  # Faster then log1p 
    stains = np.dot(np.reshape(-np.log(rgb), (-1, 3)), conv_matrix)
    stains = np.exp(-stains)
    return np.reshape(stains, rgb.shape)
