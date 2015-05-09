#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 2014-03-28

@author: Eugene Dvoretsky
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__description__ = """\
Analyze input video and render distinguished colored cells with statistic
to another one.

python2 -m immunopy.cv_transcoder '10' in.avi out.avi
"""

import argparse
import cv2

from . import iptools
from . import lut


MAGNIFICATION = '20'

H = 591
W = 1050


def CV_FOURCC(c1, c2, c3, c4):
    """Missed in cv2.

    fourcc = cv2.cv.FOURCC('I', 'Y', 'U', 'V')
    """
    return (c1 & 255) + ((c2 & 255) << 8) + ((c3 & 255) << 16) + (
        (c4 & 255) << 24)


def main():
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument("objective", type=str, help="Objective name")
    parser.add_argument("input", help="Input video")
    parser.add_argument("out", help="Output file")
    parser.add_argument("--gui", action='store_true', help="Show OpenCV window")
    # parser.add_argument("--codec", type=int, help="Output video codec")
    args = parser.parse_args()
    counter = 0

    # CMicro = iptools.CalibMicro(scale=args.scale)
    CMicro = iptools.CalibMicro(objective=args.objective)
    CProcessor = iptools.CellProcessor(
        scale=CMicro.scale, colormap=lut.random_jet(), mp=True)
    print("Scale is {:.2f} um/px".format(CMicro.scale))
    CProcessor.vtype = 2  # Not necessary now?

    if args.gui:
        cv2.namedWindow('Video')

    input_video = cv2.VideoCapture(args.input)
    assert(input_video.isOpened())
    fps = int(input_video.get(cv2.cv.CV_CAP_PROP_FPS))
    allframes = int(input_video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    print("Source FPS: {}, {} frames total".format(fps, allframes))
    codecArr = 'XVID'
    fourcc = CV_FOURCC(
        ord(codecArr[0]),
        ord(codecArr[1]),
        ord(codecArr[2]),
        ord(codecArr[3]))
    status, bgr = input_video.read()
    frame = CProcessor.process(bgr[:H,:W,::-1])
    print("Frame shape: {}".format(frame.shape))
    height, width = frame.shape[:2]
    output_video = cv2.VideoWriter(
        filename=args.out,
        fourcc=fourcc,  # '-1' Ask for an codec; '0' disables compressing.
        fps=10,
        frameSize=(width, height),
        isColor=True)
    assert(output_video.isOpened())
    if status is True:
        counter += 1
        output_video.write(frame[...,::-1])
    while True:
        status, bgr = input_video.read()
        if status is True:
            counter += 1
            print("Remaining {}/{}".format(allframes - counter, allframes))
            frame = CProcessor.process(bgr[:H,:W,::-1])
            if args.gui:
                cv2.imshow('Video', frame[...,::-1])
            output_video.write(frame[...,::-1])
#             if counter > 50:
#                 break
        else:
            break
        if args.gui and cv2.waitKey(10) == 27:
            break
    output_video.release()
    input_video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
