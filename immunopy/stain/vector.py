#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
Created on 2015-01-19

@author: radioxoma
'''

import numpy as np
from scipy import linalg


# Vectors for images from Human Protein Atlas
hpa_rgb_from_hdx = np.array([[0.684, 0.696, 0.183],
                             [0.250, 0.500, 0.850],
                             [  0.0,   0.0,   0.0]])
hpa_rgb_from_hdx[2, :] = np.cross(hpa_rgb_from_hdx[0, :], hpa_rgb_from_hdx[1, :])
hpa_hdx_from_rgb = linalg.inv(hpa_rgb_from_hdx)

# Vectors for VOPAB
vopab_rgb_from_hdx = np.array([[0.76775604, 0.49961546, 0.40116712],
                               [0.73394805, 0.4267708,  0.52838147],
                               [0.0,        0.0,        0.0]])
vopab_rgb_from_hdx[2, :] = np.cross(vopab_rgb_from_hdx[0, :], vopab_rgb_from_hdx[1, :])
vopab_hdx_from_rgb = linalg.inv(vopab_rgb_from_hdx)
