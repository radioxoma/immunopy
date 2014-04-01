#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 2014-03-28

@author: radioxoma
"""

import cv2

import main
import iptools


MAGNIFICATION = '10'
CMicro = iptools.CalibMicro(MAGNIFICATION)
SCALE = CMicro.um2px(1)

THRESHOLD_SHIFT = 8
PEAK_DISTANCE = 8
MIN_SIZE = 15
MAX_SIZE = 3000


def CV_FOURCC(c1, c2, c3, c4):
    """Missed in cv2."""
    return (c1 & 255) + ((c2 & 255) << 8) + ((c3 & 255) << 16) + ((c4 & 255) << 24)


if __name__ == '__main__':
    input_video = cv2.VideoCapture('/home/radioxoma/analysis/Видео/10x_1280x1024_20_lags_perfect.avi')
    assert(input_video.isOpened())
    width = int(input_video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    fps = int(input_video.get(cv2.cv.CV_CAP_PROP_FPS))
    allframes = int(input_video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    print(fps, allframes)
    codecArr = 'XVID'
    fourcc = CV_FOURCC(
        ord(codecArr[0]),
        ord(codecArr[1]),
        ord(codecArr[2]),
        ord(codecArr[3]))
#     fourcc = cv2.cv.FOURCC('I', 'Y', 'U', 'V')
    output_video = cv2.VideoWriter(
        filename='rendered.avi',
        fourcc=fourcc,  # '-1' Ask for an codec; '0' disables compressing.
        fps=15,
        frameSize=(1170, 936),
        isColor=True)
    assert(output_video.isOpened())
    
#     cv2.namedWindow('Video')
    counter = 0
    while True:
        status, bgr = input_video.read()
        if status is True:
            counter += 1
            print('Remaining %d/%d') % (allframes - counter, allframes)
            frame = main.process(
                bgr[...,::-1],
                SCALE,
                THRESHOLD_SHIFT,
                PEAK_DISTANCE,
                MIN_SIZE,
                MAX_SIZE)
#             cv2.imshow('Video', frame[...,::-1])
            output_video.write(frame[...,::-1])
#             print(frame.shape)
#             if counter > 50:
#                 break
        else:
            break
#         if cv2.waitKey(10) == 27:
#             break
    
    output_video.release()
    input_video.release()
    cv2.destroyAllWindows()
