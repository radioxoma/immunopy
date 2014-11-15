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

from iptools import CalibMicro


class Test(unittest.TestCase):

    def testInstance(self):
        CMicro = CalibMicro(objective='20')
        assert(0.22 < CMicro.scale < 0.23)

    def testScaleSwitching(self):
        CMicro = CalibMicro(objective='100')
        CMicro.scalename = '20'
        CMicro.scalename = '10'
        assert(0.45 < CMicro.scale < 0.46)
        
    def testUm2px(self):
        CMicro = CalibMicro(objective='20')
        assert(43 < CMicro.um2px(10) < 44)
        
    def testPx2um(self):
        CMicro = CalibMicro(objective='20')
        assert(9.5 < CMicro.px2um(43) < 10.1)

if __name__ == "__main__":
    unittest.main()
