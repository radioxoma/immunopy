#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 2014-01-18

@author: Eugene Dvoretsky

Cell segmentation algorithm demo with simple opencv-based GUI.
"""

import time
import cv2
import MMCorePyFake as MMCorePy
import iptools
import lut


MAGNIFICATION = '10'

# DEVICE = ['Camera', 'DemoCamera', 'DCam']
# DEVICE = ['Camera', 'OpenCVgrabber', 'OpenCVgrabber']
DEVICE = ['Camera', 'BaumerOptronic', 'BaumerOptronic']


def set_threshold_shift(value):
    CProcessor.threshold_shift = value - 100


def set_peak_distance(value):
    CProcessor.peak_distance = value


def set_max_size(value):
    CProcessor.max_size = value


def set_min_size(value):
    CProcessor.min_size = value


def set_vtype(value):
    """Type of visualization method (overlay or labels)."""
    CProcessor.vtype = value


if __name__ == '__main__':
    CMicro = iptools.CalibMicro(MAGNIFICATION)
    SCALE = CMicro.scale
    CProcessor = iptools.CellProcessor(scale=SCALE, colormap=lut.random_jet(), mp=True)
    print('curscale %f') % CMicro.scale
    print('um2px %f') % SCALE

    mmc = MMCorePy.CMMCore()
    print('ImageBufferSize %f' % mmc.getImageBufferSize())  # Returns the size of the internal image buffer.
    print('BufferTotalCapacity %f' % mmc.getBufferTotalCapacity())
    mmc.setCircularBufferMemoryFootprint(100)
    mmc.enableStderrLog(False)
    mmc.enableDebugLog(False)
    mmc.loadDevice(*DEVICE)
    mmc.initializeDevice(DEVICE[0])
    mmc.setCameraDevice(DEVICE[0])
    # mmc.setProperty(DEVICE[0], 'Binning', '2')
    mmc.setProperty(DEVICE[0], 'PixelType', '32bitRGB')
    iptools.setMmcResolution(mmc, 512, 512)
    mmc.snapImage()  # Baumer bug workaround
    cv2.namedWindow('Overlay')
    cv2.namedWindow('Controls', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_AUTOSIZE)
    if mmc.hasProperty(DEVICE[0], 'Gain'):
        cv2.createTrackbar(
            'Gain', 'Controls',
            int(float(mmc.getProperty(DEVICE[0], 'Gain'))),
            int(mmc.getPropertyUpperLimit(DEVICE[0], 'Gain')),
            lambda value: mmc.setProperty(DEVICE[0], 'Gain', value))
    if mmc.hasProperty(DEVICE[0], 'Exposure'):
        cv2.createTrackbar(
            'Exposure', 'Controls',
            int(float(mmc.getProperty(DEVICE[0], 'Exposure'))),
            100,  # int(mmc.getPropertyUpperLimit(DEVICE[0], 'Exposure')),
            lambda value: mmc.setProperty(DEVICE[0], 'Exposure', int(value)))
    cv2.createTrackbar('SHIFT_THRESHOLD', 'Controls', CProcessor.threshold_shift + 100, 200, set_threshold_shift)
    cv2.createTrackbar('PEAK_DISTANCE', 'Controls', CProcessor.peak_distance, 100, set_peak_distance)
    cv2.createTrackbar('MAX_SIZE', 'Controls', CProcessor.max_size, 5000, set_max_size)
    cv2.createTrackbar('MIN_SIZE', 'Controls', CProcessor.min_size, 1000, set_min_size)
    cv2.createTrackbar('VMethod', 'Controls', CProcessor.vtype, 4, set_vtype)
    mmc.startContinuousSequenceAcquisition(1)

    while True:
        start_time = time.time()
        if mmc.getRemainingImageCount() > 0:
            rgb32 = mmc.getLastImage()
            cv2.imshow(
                'Overlay',
                CProcessor.process(iptools.rgb32asrgb(rgb32)[...,::-1]))
        if cv2.waitKey(5) >= 0:
            break
        print('FPS: %f') % (1. / (time.time() - start_time))
    mmc.reset()
    cv2.destroyAllWindows()
