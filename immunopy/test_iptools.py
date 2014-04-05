#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 18 Jan. 2014 г.

@author: radioxoma
"""

import unittest
from iptools import CalibMicro


class Test(unittest.TestCase):

    def testInstance(self):
        CMicro = CalibMicro('20')
        assert(0.22 < CMicro.curr_scale < 0.23)

    def testScaleSwitching(self):
        CMicro = CalibMicro('20')
        CMicro.curr_scale = '10'
        assert(0.45 < CMicro.curr_scale < 0.46)
        
    def testUm2px(self):
        CMicro = CalibMicro('20')
        assert(43 < CMicro.um2px(10) < 44)
        
    def testPx2um(self):
        CMicro = CalibMicro('20')
        assert(9.5 < CMicro.px2um(43) < 10.1)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testCalibration']
    unittest.main()
