#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division

"""
Created on Wed Aug 06 19:37:30 2014

@author: radioxoma
"""

import os
import numpy as np
from skimage import color
import pyopencl as cl
import pyopencl.array as cla
from scipy import misc
import matplotlib.pyplot as plt

VERBOSE = False


class ColorDeconvolution(object):
    """Provide color deconvolution facilities with OpenCL.
    """
    def __init__(self):
        super(ColorDeconvolution, self).__init__()
        self.__basetype = np.float32
        curdir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(curdir, 'kernels.cl')) as f:
            kernels = f.read()
        # ctx = cl.create_some_context()
        self.ctx = cl.Context(
            cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU))
        if VERBOSE:
            print(self.ctx.get_info(cl.context_info.DEVICES))
        queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, kernels).build()
        # print(self.prg.get_info(cl.program_info.KERNEL_NAMES)) # Not in 1:2013.2

        # self.stain = color.hed_from_rgb.astype(self.__basetype)
        # self.stain_g = cla.to_device(queue, self.f_order(self.stain), self.mem_pool)
        # stain = np.arange(9, dtype=self.__basetype).reshape((3, 3))

        self.mem_pool = cl.tools.MemoryPool(cl.tools.ImmediateAllocator(queue))

    def check_contiguous(self, arr):
        """Change memory layout to C (row-major) order, cast to float32.

        It's *not* oposite of f_order.
        """
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            if VERBOSE:
                print('check_arr: ascontiguous %d elements - performance may suffer') % arr.size
        if arr.dtype is not np.float32:
            arr = arr.astype(np.float32)
            if VERBOSE:
                print('check_arr: casting to float32 %d elements - performance may suffer') % arr.size
        return arr

    def check_fortran(self, arr):
        """Change memory layout to FORTRAN (column-major) order, cast to float32.
        """
        if not arr.flags.f_contiguous:
            arr = np.asfortranarray(arr, dtype=np.float32)
            if VERBOSE:
                print('check_arr: as fortran %d elements - performance may suffer') % arr.size
        if arr.dtype is not np.float32:
            arr = arr.astype(np.float32)
            if VERBOSE:
                print('check_arr: casting to float32 %d elements - performance may suffer') % arr.size
        return arr

    def optical_density(self, rgb):
        queue = cl.CommandQueue(self.ctx)
        if rgb.dtype is not np.float32:
            rgb = rgb.astype(np.float32)
        img_g = cla.to_device(queue, rgb, self.mem_pool)
        self.prg.opticalDense(queue, (img_g.size, 1), None, img_g.data)
        return img_g.get()

    def dot(self, A, B):
        """Output must have same shape as A.

        Incoming RGB matrix "A" should be aligned
        """
        A = self.check_contiguous(A)
        B = self.check_contiguous(B)
        assert(A.flags.c_contiguous == B.flags.c_contiguous)
        queue = cl.CommandQueue(self.ctx)
        if A.dtype is not np.float32:
            A = A.astype(np.float32)
        if B.dtype is not np.float32:
            B = B.astype(np.float32)
        A_g = cla.to_device(queue, A, self.mem_pool)
        B_g = cla.to_device(queue, B, self.mem_pool)
        C_g = cla.empty(queue, (A.shape[0], B.shape[1]), dtype=A_g.dtype, order="C", allocator=self.mem_pool)
        self.prg.gemm_slow(queue, C_g.shape, None, C_g.data, A_g.data, B_g.data, np.int32(A.shape[1]), np.int32(B.shape[1]))
        return C_g.get()

    def unmix_stains(self, rgb, stain):
        """Take RGB IHC image and split it to stains like skimage version.
        """
        rgb = self.check_contiguous(rgb)
        stain = self.check_contiguous(stain)
        assert(rgb.flags.c_contiguous == stain.flags.c_contiguous)
        queue = cl.CommandQueue(self.ctx)
        rgb2d = rgb.reshape(-1, 3)  # 2D array with R,G,B columns from 3D
        rgb2d_g = cla.to_device(queue, rgb2d, allocator=self.mem_pool)
        stain_g = cla.to_device(queue, stain, allocator=self.mem_pool)
        out_g = cla.empty(queue, (rgb2d.shape[0], stain.shape[1]), dtype=rgb2d_g.dtype, order="C", allocator=self.mem_pool)

        # Process as flat array
        self.prg.opticalDense(queue, (rgb2d.size, 1), None, rgb2d_g.data)

        # In PyOpenCL arrays rgb2d_g.shape[0] is column count (usually 3 columns here).
        self.prg.gemm_slow(queue, out_g.shape, None, out_g.data, rgb2d_g.data, stain_g.data, np.int32(rgb2d.shape[1]), np.int32(stain.shape[1]))
        ### self.prg.gemm(queue, rgb2d_g.shape, None, out_g.data, rgb2d_g.data, stain_g.data, np.int32(rgb2d_g.shape[0]), np.int32(stain_g.shape[1]))
        # event =
        # event.wait()
        return out_g.get().reshape(rgb.shape) # Again 3D array

    def color_deconvolution(self, rgb, stain):
        """Return stains in normal (non-logarithmic) color space.
        """
        rgb = self.check_contiguous(rgb)
        stain = self.check_contiguous(stain)
        assert(rgb.flags.c_contiguous == stain.flags.c_contiguous)
        queue = cl.CommandQueue(self.ctx)
        rgb2d = rgb.reshape(-1, 3)  # 2D array with R,G,B columns from 3D
        rgb2d_g = cla.to_device(queue, rgb2d, allocator=self.mem_pool)
        stain_g = cla.to_device(queue, stain, allocator=self.mem_pool)
        out_g = cla.empty(queue, (rgb2d.shape[0], stain.shape[1]), dtype=rgb2d_g.dtype, order="C", allocator=self.mem_pool)
        # Process as flat array
        self.prg.opticalDense(queue, (rgb2d.size, 1), None, rgb2d_g.data)
        # In PyOpenCL arrays rgb2d_g.shape[0] is column count (usually 3 columns here).
        self.prg.gemm_slow(queue, out_g.shape, None, out_g.data, rgb2d_g.data, stain_g.data, np.int32(rgb2d.shape[1]), np.int32(stain.shape[1]))
        self.prg.toColorDense(queue, (out_g.size, 1), None, out_g.data)
        return out_g.get().reshape(rgb.shape) # Again 3D array

def f_order(arr):
    """Convert to FORTRAN (column-major) order, if still not."""
    if arr.flags.c_contiguous:
        print("Transposing array")
        return np.array(arr.T, copy=False, order='F')
    else:
        return np.array(arr, copy=False, order='F')
