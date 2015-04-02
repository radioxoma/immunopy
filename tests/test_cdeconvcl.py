#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division

import os
import sys
import unittest

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(curdir, '/../immunopy'))

import numpy as np
from skimage import color
from skimage.util import dtype
from immunopy.stain import cdeconvcl
from immunopy.stain import cdeconv


class TestGEMM(unittest.TestCase):
    MW = 3  # Crucial matrix value
    h = 10
    w = 5
    A = np.arange(start=1, stop=MW*h+1, dtype=np.float32).reshape(h, MW)

    def test_dot(self):
        B = np.arange(start=1, stop=self.MW*self.w+1, dtype=np.float32).reshape(self.MW, self.w)
        CD = cdeconvcl.ColorDeconvolution()
        dt_cl = CD.dot(self.A, B)
        dt_np = np.dot(self.A, B)
        np.testing.assert_allclose(dt_cl, dt_np,
            err_msg='FAILED TEST DOT')

    def test_dot_float(self):
        B = color.hed_from_rgb.astype(np.float32)
        CD = cdeconvcl.ColorDeconvolution()
        dt_cl = CD.dot(self.A, B)
        dt_np = np.dot(self.A, B)
        np.testing.assert_allclose(dt_cl, dt_np, rtol=1e-5,
            err_msg='FAILED TEST DOT FLOAT')


class TestDeconvolution(unittest.TestCase):
    h, w, d = 3, 4, 3

    def test_density(self):
        rgb = np.arange(self.h*self.w*self.d, dtype=np.float32).reshape(
            self.h, self.w, self.d)
        CD = cdeconvcl.ColorDeconvolution()
        dens_cl = CD.optical_density(rgb)
        dens_np = -np.log1p(rgb)
        np.testing.assert_allclose(dens_cl, dens_np,
            err_msg='Density test failed')

    def test_unmix_stain_like_skimage(self):
        rgb = np.arange(self.h*self.w*self.d, dtype=np.float32).reshape(
            self.h, self.w, self.d)
        stain = color.hdx_from_rgb.astype(np.float32)
        CD = cdeconvcl.ColorDeconvolution()
        dec_cl = CD.unmix_stains(rgb, stain)
        dec_np = np.reshape(np.dot(np.reshape(-np.log1p(rgb), (-1, 3)),
            stain), rgb.shape)
        np.testing.assert_allclose(dec_cl, dec_np, rtol=1e-5,
            err_msg='FAILED TEST UNMIX Like skimage')

    def test_unmix_stain_like_me(self):
        rgb = np.random.randint(
            0, 256, self.h*self.w*self.d).astype(np.uint8).reshape(self.h, self.w, self.d)
        stain = color.hdx_from_rgb.astype(np.float32)

        dec_np = cdeconv.color_deconvolution(rgb, stain)
        CD = cdeconvcl.ColorDeconvolution()
        dec_cl = CD.color_deconvolution(dtype.img_as_float(rgb), stain)
        np.testing.assert_allclose(dec_cl, dec_np, rtol=1e-5)

'''
def benchmark():
    img = np.random.randint(low=0, high=256, size=1536 * 2048 * 3)
    img = img.reshape(1536, 2048, 3).astype(np.uint8)
    # img = misc.imread("Ki6720x_blue_filter.tif")
    # imgi = misc.imread("rgbcolortest.png")[...,:3].
    # imgi = misc.imread("hdab.tif")
    # imgi = misc.imread("ir100px.tif")
    global img
    import timeit
    setcl = "from __main__ import img, ColorDeconvolution; from skimage import color; CDec = ColorDeconvolution()"
    cltimers = timeit.Timer('CDec.unmix_stains(img, color.hdx_from_rgb)', setup=setcl).repeat(2, 5)
    print(cltimers)
    setnp = "from __main__ import img; from skimage import color"
    nptimers = timeit.Timer('color.separate_stains(img, color.hdx_from_rgb)', setup=setnp).repeat(2, 5)
    print(nptimers)
    print("\nOpenCL faster than Numpy in %f times") % (min(nptimers) / min(cltimers))

def benchmark_density_aligning():
    img = np.arange(3453371, dtype=np.float32)
    global img
    import timeit
    # Test worse wariant: image size is prime number.
    setcl = "from __main__ import img, ColorDeconvolution; from skimage import color; CDec = ColorDeconvolution()"
    primenum_timers = timeit.Timer('CDec.optical_density(img)', setup=setcl).repeat(2, 10)
    # Test best wariant: image size is n*512.
    img = np.arange(6744*512, dtype=np.float32)
    timers = timeit.Timer('CDec.optical_density(img)', setup=setcl).repeat(2, 10)

    print(primenum_timers)
    print(timers)
    print("Precice aligning faster in %f times") % (min(primenum_timers) / min(timers))
'''

if __name__ == "__main__":
    unittest.main()
    # benchmark()
    # benchmark_density_aligning()
