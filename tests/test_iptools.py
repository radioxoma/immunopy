#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 2014-01-18

@author: Eugene Dvoretsky
"""

import os
import sys
import unittest

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curdir + '/../immunopy')

from scipy import misc
from skimage import filters, color
import iptools


class Test(unittest.TestCase):

    def testInstance(self):
        CMicro = iptools.CalibMicro(objective='20')
        self.assertTrue(0.22 < CMicro.scale < 0.23)

    def testScaleSwitching(self):
        CMicro = iptools.CalibMicro(objective='100')
        CMicro.scalename = '20'
        CMicro.scalename = '10'
        self.assertTrue(0.45 < CMicro.scale < 0.46)

    def testUm2px(self):
        CMicro = iptools.CalibMicro(objective='20')
        self.assertTrue(43 < CMicro.um2px(10) < 44)

    def testPx2um(self):
        CMicro = iptools.CalibMicro(objective='20')
        self.assertTrue(9.5 < CMicro.px2um(43) < 10.1)


class TestThresholdIsodata(unittest.TestCase):
    def setUp(self):
        self.rgb = misc.imread("../immunopy/image/hdab256.tif")
        self.hdx = color.separate_stains(self.rgb, color.hdx_from_rgb)
        self.hem = self.hdx[:,:,0]
        self.dab = self.hdx[:,:,1]

    def testIsodataThreshold(self):
        t = iptools.threshold_isodata(image=self.rgb[...,0])
        self.assertEqual(t, 113)

    def testIsodataThresholdNegativeFloat(self):
        t = iptools.threshold_isodata(image=self.hem)
        self.assertAlmostEqual(t, -0.84058, places=5)

    def testCompareSkimageAndIPThresholdIsodataValue(self):
        t1 = iptools.threshold_isodata(image=self.rgb[...,0])
        t2 = filters.threshold_isodata(image=self.rgb[...,0])
        self.assertEqual(t1, t2, msg='Different ISODATA algorithm')

    def testIsodataPositiveShift(self):
        """Input data > 0."""
        t1 = iptools.threshold_isodata(image=self.rgb[...,0], shift=10)
        self.assertEqual(t1, 137.7)

    def testIsodataNegativeShift(self):
        t1 = iptools.threshold_isodata(image=self.rgb[...,0], shift=-200)
        self.assertAlmostEqual(t1, -381)

    def testIsodataThresholdFloatShift(self):
        t = iptools.threshold_isodata(image=self.hem, shift=20)
        self.assertAlmostEqual(t, -0.72770, places=5)

    def testThreesholdMaxLimit(self):
        t = iptools.threshold_isodata(image=self.rgb[...,0], max_limit=40)
        self.assertAlmostEqual(t, 106.8)

    def testThreesholdLowLimit(self):
        t = iptools.threshold_isodata(image=self.rgb[...,0], min_limit=45)
        self.assertEqual(t, 119.15)

    def testMaxMinSwapException(self):
        with self.assertRaises(ValueError):
            iptools.threshold_isodata(image=self.rgb, min_limit=1, max_limit=0)


if __name__ == "__main__":
    unittest.main()
