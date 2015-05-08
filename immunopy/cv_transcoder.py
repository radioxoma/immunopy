#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 2014-03-28

@author: Eugene Dvoretsky

Analyze input video and render distinguished colored cells with statistic
to another one.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2

from . import iptools
from . import lut


MAGNIFICATION = '10'
COUNTER = 0
H = 591
W = 1050


def CV_FOURCC(c1, c2, c3, c4):
    """Missed in cv2."""
    return (c1 & 255) + ((c2 & 255) << 8) + ((c3 & 255) << 16) + ((c4 & 255) << 24)


if __name__ == '__main__':
    CMicro = iptools.CalibMicro(objective=MAGNIFICATION)
    CProcessor = iptools.CellProcessor(
        scale=CMicro.scale, colormap=lut.random_jet(), mp=True)
    CProcessor.vtype = 4

    cv2.namedWindow('Video')
    input_video = cv2.VideoCapture('/home/radioxoma/analysis/Видео/10x_1280x1024_20_lags_perfect.avi')
    assert(input_video.isOpened())
    fps = int(input_video.get(cv2.cv.CV_CAP_PROP_FPS))
    allframes = int(input_video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    print('Source fps: %d, total %d frames') % (fps, allframes)
    # fourcc = cv2.cv.FOURCC('I', 'Y', 'U', 'V')
    codecArr = 'XVID'
    fourcc = CV_FOURCC(
        ord(codecArr[0]),
        ord(codecArr[1]),
        ord(codecArr[2]),
        ord(codecArr[3]))
    status, bgr = input_video.read()
#     frame = CProcessor.process(bgr[...,::-1])
    frame = CProcessor.process(bgr[:H,:W,::-1])
    height, width = frame.shape[:2]
    print(frame.shape)
    output_video = cv2.VideoWriter(
        filename='rendered.avi',
        fourcc=fourcc,  # '-1' Ask for an codec; '0' disables compressing.
        fps=10,
        frameSize=(width, height),
        isColor=True)
    assert(output_video.isOpened())
    if status is True:
        COUNTER += 1
        output_video.write(frame[...,::-1])
    while True:
        status, bgr = input_video.read()
        if status is True:
            COUNTER += 1
            print('Remaining %d/%d') % (allframes - COUNTER, allframes)
            frame = CProcessor.process(bgr[:H,:W,::-1])
#             cv2.imshow('Video', frame[...,::-1])
            output_video.write(frame[...,::-1])
#             if COUNTER > 50:
#                 break
        else:
            break
#         if cv2.waitKey(10) == 27:
#             break
    output_video.release()
    input_video.release()
    cv2.destroyAllWindows()
