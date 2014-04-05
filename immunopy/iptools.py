#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 16 Jan. 2014 г.

@author: radioxoma
"""

import numpy as np
from scipy import ndimage
from skimage.exposure import histogram
from skimage.morphology import watershed
from skimage.color import separate_stains, hdx_from_rgb
from skimage.feature import peak_local_max
import cv2

import lut


class CalibMicro(object):
    """Microscope objective calibration and size conversion.

    Instance it with default objective name (e.g. '20').

    TODO:
        * Use properties
        * Handle scales as class attributes
        * Use binning
        * Check 100 magnification
    """

    def __init__(self, objective_name):
        """Objective scales (um/px) from Leica Acquisition Suite *.cal.xml
        files for Leica DMI 2000 optical microscope.
        """
        super(CalibMicro, self).__init__()
        self.scales = {'5': 9.1428482142944335E-01,
                       '10': 4.5714241071472167E-01,
                       '20': 2.2857120535736084E-01,
                       '63': 7.2562287415035193E-02,
                       '100': 4.571E-02}
        # self.curr_scale = None
        self.binning = 1
        self.curr_scale = objective_name  # objective name
#         self.roi = (2048, 1536)

    @property
    def curr_scale(self):
        return self._curr_scale
    @curr_scale.setter
    def curr_scale(self, value):
        """Set microscope scale from available."""
        assert(isinstance(value, str))
        self._curr_scale = self.scales[value]

    def um2px(self, um, scale=None):
        """Convert um to pixel line."""
        return um / self.curr_scale

    def px2um(self, um, scale=None):
        """Convert pixel line to um."""
        return um * self.curr_scale

    def um2circle(self, diameter):
        """Диаметр (um) в площадь эквивалентного круга в px."""
        return np.pi * (self.um2px(diameter) / 2.0) ** 2

    def um2rect(self, diameter):
        """Диаметр (um) в площадь эквивалентного квадрата в px."""
        return self.um2px(diameter) ** 2

    # def set_all_scales(self):
    #     pass

    # def get_all_scales(self):
    #     pass


class CellProcessor(object):
    """Segment and visualize cell image."""
    def __init__(self, scale, colormap, pool=None):
        super(CellProcessor, self).__init__()
        self.threshold_shift = 20
        self.min_size = 10
        self.max_size = 3000
        self.vtype = 1

        self.peak_distance = 8
        self.scale = scale
        self.blur = 2
        self.colormap = colormap
        self.pool = pool

    @property
    def vtype(self):
        return self._vtype
    @vtype.setter
    def vtype(self, value):
        self._vtype = value

    @property
    def threshold_shift(self):
        return self._threshold_shift
    @threshold_shift.setter
    def threshold_shift(self, value):
        self._threshold_shift = value

    @property
    def min_size(self):
        return self._min_size
    @min_size.setter
    def min_size(self, value):
        if value > 0:
            self._min_size = value

    @property
    def max_size(self):
        return self._max_size
    @max_size.setter
    def max_size(self, value):
        if value > 0:
            self._max_size = value

    @property
    def peak_distance(self):
        return self._peak_distance
    @peak_distance.setter
    def peak_distance(self, value):
        if value > 0:
            self._peak_distance = value

    def process(self, image):
        """Segmentation and statistical calculation.
        """

        rgb = image.copy()
        # Коррекция освещённости

        # Размытие
        meaned = cv2.blur(rgb, (self.blur, self.blur))

        # Масштаб
        scaled = rescale(meaned, self.scale)

        # Разделение красителей
        hdx = separate_stains(scaled, hdx_from_rgb)
        hem = hdx[:,:,0]
        dab = hdx[:,:,1]

        # MULTICORE -------------------------------------------------------------
        if self.pool is None:
            hemfiltered, hemfnum = worker(hem, self.threshold_shift, self.peak_distance, self.min_size, self.max_size)
            dabfiltered, dabfnum = worker(dab, self.threshold_shift + 10, self.peak_distance, self.min_size, self.max_size)
        else:
            hproc = self.pool.apply_async(worker, (hem, self.threshold_shift, self.peak_distance, self.min_size, self.max_size))
            dproc = self.pool.apply_async(worker, (dab, self.threshold_shift + 10, self.peak_distance, self.min_size, self.max_size))
            hemfiltered, hemfnum = hproc.get(timeout=5)
            dabfiltered, dabfnum = dproc.get(timeout=5)
        # MULTICORE END ---------------------------------------------------------

        # Stats
        stats = 'Num D%3.d/H%3.d, %.2f' % (dabfnum, hemfnum, float(dabfnum) / (hemfnum + dabfnum + 0.001) * 100)
        stats2 = 'Area fract %.2f' % (calc_stats(hemfiltered, dabfiltered) * 100)
        stats3 = 'Ar disj %.2f' % (calc_stats_binary(hemfiltered, dabfiltered) * 100)

        # Visualization
        if self.vtype == 0:
            overlay = scaled
        elif self.vtype == 1:
            overlay = draw_overlay(scaled, dabfiltered, hemfiltered)
        elif self.vtype == 2:
            overlay = lut.apply_lut(dabfiltered, self.colormap)
        else:
            overlay = lut.apply_lut(hemfiltered, self.colormap)
        cv2.putText(overlay, stats, (2,25), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
        cv2.putText(overlay, stats2, (2,55), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
        cv2.putText(overlay, stats3, (2,85), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
        return overlay


def worker(stain, threshold_shift, peak_distance, min_size, max_size):
    """Process each stain.

    Return filtered objects and their count.
    Would not work with processes as class method.
    """
    stth = threshold_isodata(stain, shift=threshold_shift)
    stmask = stain > stth
    stmed = ndimage.filters.median_filter(stmask, size=2)
    stedt = cv2.distanceTransform(
        stmed.view(np.uint8), distanceType=cv2.cv.CV_DIST_L2, maskSize=3)
    st_max = peak_local_max(
        stedt, min_distance=peak_distance, exclude_border=False, indices=False)
    stlabels, stlnum = watershed_segmentation(stmed, stedt, st_max)
    stfiltered, stfnum = filter_objects(
        stlabels, stlnum, min_size, max_size)
    return stfiltered, stfnum


def rgb32asrgb(rgb32):
    """View RGB32 as RGB array (no copy).

    low memory address    ---->      high memory address
    | pixel | pixel | pixel | pixel | pixel | pixel |...
    |-------|-------|-------|-------|-------|-------|...
    |B|G|R|A|B|G|R|A|B|G|R|A|B|G|R|A|B|G|R|A|B|G|R|A|...
    http://avisynth.nl/index.php/RGB32
    """
    return rgb32.view(dtype=np.uint8).reshape(
        rgb32.shape[0], rgb32.shape[1], 4)[...,2::-1]


def get_central_rect(width, height, divisor=1):
    """Select central rectangle with reduced size.

    MMstudio-like.
       1: 2048 1536 (  0,   0, 2048, 1536)
       2: 1024  768 (512, 384, 1536, 1152)
       4:  512  384 (768, 576, 1280, 960)
       8:  256  192 (896, 672, 1152, 864)
      16:  128   96 (960, 720, 1088, 816)
      32:   64   48 (992, 744, 1056, 792)
      64:   32   24 (1008, 756, 1040, 780)
     128:   16   12 (1016, 762, 1032, 774)
     256:    8    6 (1020, 765, 1028, 771)
     512:    4    2 (1022, 767, 1026, 769)
    1024:    2    0 (1023, 768, 1025, 768)

    LeicaDFC 295 available resolutions
    2048 1536
    1600 1200
    1280 1024
    1024  768
     640  480
    """
    centerx = width / 2
    centery = height / 2
    roi = (
        centerx - centerx / divisor,
        centery - centery / divisor,
        width / divisor,
        height / divisor)
    return roi


def set_mmc_resolution(mmc, width, height):
    """Select rectangular ROI in center of the frame.
    """
    x = (mmc.getImageWidth() - width) / 2
    y = (mmc.getImageHeight() - height) / 2
    if not all((x > 0, y > 0)):
        raise ValueError('ROI w%d h%d is out of image size' % (width, height))
    mmc.setROI(x, y, width, height)


def correct_background():
    """normalisation, white balance, etc"""
    pass


def normalize(img):
    """G. Landini proposal.
    http://imagejdocu.tudor.lu/doku.php?id=howto:working:how_to_correct_background_illumination_in_brightfield_microscopy
    corrected = (Specimen - Darkfield) / (Brightfield - Darkfield) * 255
    """
    immin = img.min()
    return (img - immin) / (img.max() - immin).astype(np.float32)


def rescale(source, scale):
    """Если scale > 2 um/px, то изображение будет уменьшаться.
    """
    if scale == 2.0:  # pixels per um
        return source
    else:
        scl_factor = 2.0 / scale
        fy = int(round(source.shape[0] * scl_factor))
        fx = int(round(source.shape[1] * scl_factor))
        # cv2.INTER_CUBIC
        return cv2.resize(source, dsize=(fx, fy), interpolation=cv2.INTER_LINEAR)


def threshold_isodata(image, nbins=256, shift=None):
    """Return threshold value based on ISODATA method.

    Histogram-based threshold, known as Ridler-Calvard method or intermeans.

    Parameters
    ----------
    image : array
        Input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    shift : int, optional
        Shift threshold value by percent up/down.

    Returns
    -------
    threshold : float or int, corresponding input array dtype.
        Upper threshold value. All pixels intensities that less or equal of
        this value assumed as background.

    References
    ----------
    .. [1] Ridler, TW & Calvard, S (1978), "Picture thresholding using an
           iterative selection method"
    .. [2] IEEE Transactions on Systems, Man and Cybernetics 8: 630-632,
           http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4310039
    .. [3] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
           Techniques and Quantitative Performance Evaluation" Journal of
           Electronic Imaging, 13(1): 146-165,
           http://www.busim.ee.boun.edu.tr/~sankur/SankurFolder/Threshold_survey.pdf
    .. [4] ImageJ AutoThresholder code,
           http://fiji.sc/wiki/index.php/Auto_Threshold

    Examples
    --------
    >>> from skimage.data import coins
    >>> image = coins()
    >>> thresh = threshold_isodata(image)
    >>> binary = image > thresh
    """
    hist, bin_centers = histogram(image, nbins)
    # On blank images (e.g. filled with 0) with int dtype, `histogram()`
    # returns `bin_centers` containing only one value. Speed up with it.
    if bin_centers.size == 1:
        return bin_centers[0]
    # It is not necessary to calculate the probability mass function here,
    # because the l and h fractions already include the normalization.
    pmf = hist.astype(np.float32)  # / hist.sum()
    cpmfl = np.cumsum(pmf, dtype=np.float32)
    cpmfh = np.cumsum(pmf[::-1], dtype=np.float32)[::-1]

    binnums = np.arange(pmf.size, dtype=np.uint8)
    # l and h contain average value of pixels in sum of bins, calculated
    # from lower to higher and from higher to lower respectively.
    l = np.ma.divide(np.cumsum(pmf * binnums, dtype=np.float32), cpmfl)
    h = np.ma.divide(
        np.cumsum((pmf[::-1] * binnums[::-1]), dtype=np.float32)[::-1],
        cpmfh)

    allmean = (l + h) / 2.0
    threshold = bin_centers[np.nonzero(allmean.round() == binnums)[0][0]]
    # This implementation returns threshold where
    # `background <= threshold < foreground`.

    if shift:
        threshold += (bin_centers[-1] - bin_centers[0]) * shift / 100.
    return threshold


def watershed_segmentation(mask, edt, local_maxi):
    """Segment clumped nuclei.

    Something like Malpica N, de Solorzano CO, Vaquero JJ, Santos et all
    'Applying watershed algorithms to the segmentation of clustered nuclei.'
    Cytometry 28, 289-297 (1997).
    """
    markers, num_markers = ndimage.label(local_maxi)
    labels = watershed(-edt, markers, mask=mask)
    return labels, num_markers


def filter_objects(labels, num=None, min_size=150, max_size=2000, in_place=False):
    """Remove too small or too big objects from array.

    labels: array labels given by ndimage.label function.
    num: int Total number of labels given by ndimage.label function.

    return: filtered_array, number_of_objects on this array
    """
    if num is None:
        num = labels.max()  # Not too safe
    if in_place:
        lbls = labels
        # Compute labels sizes. Not so fast as np.bincount.
        comp_sizes = ndimage.histogram(
            input=labels, min=0, max=num, bins=num + 1)
    else:
        # We can't just copy cause numpy.bincount has an bug and falls with uint dtypes.
        # lbls = labels.copy()
        lbls = labels.astype(np.int32)
        # Compute labels sizes.
        comp_sizes = np.bincount(lbls.ravel())
    fmask = (max_size < comp_sizes) | (min_size > comp_sizes)
    # fmask[lbls] Returns two-dimensional bool array. `True` for objects that's need to removal.
    lbls[fmask[lbls]] = 0
    return lbls, num - fmask[1:].sum()


def circularity(arr):
    """Оценить, насколько массив на входе близок к окружности.

    Функция нацелена на производительность. Никаких bounding box.

    0.6-1.1 окружность.
    <1 - маленький изогнутый фрагмент, не окружность.
    >1 - скорее четырёхугольник, чем круг. Например полностью залитый квадрат.
    """
    """
    # CellProfiler / cellprofiler / modules / measureobjectsizeshape.py
    # FormFactor: Calculated as 4*π*Area / Perimeter^2.
    # Equals 1 for a perfectly circular object.

    Roundness – (Perimeter^2) / 4 * PI * Area).
    This gives the reciprocal value of Shape Factor for those that are used to using it.
    A circle will have a value slightly greater than or equal to 1.
    Other shapes will increase in value.
    """
    # Найдём максимальный радиус окружности, которую можно вписать в этот массив.
    radius = np.min(arr.shape) / 2.0
    S = np.pi * radius ** 2
    return arr.sum() / S


def draw_overlay(srcrgb, red, blue):
    """Draw objects in single channel. Alpha-like.
    """
    rgb = srcrgb.copy()
    np.copyto(
        rgb[...,2], 255,
        where=blue.astype(dtype=np.bool8))
    np.copyto(
        rgb[...,0], 255,
        where=red.astype(dtype=np.bool8))
    return rgb


def draw_masks(srcrgb, red, blue):
    """Draw objects on image"""
    """
    Draws on source image with different colors under stains masks.

    После завершения работы с выделением DAB и гематоксилина, проверь,
    пересекаются ли эти маски. Это можно сделать, заменяя один канал:
    np.copyto(bgr[:,:,0], 255, where=mask.view(dtype=np.bool8))
    """
    rgb = srcrgb.copy()
    # Заменяет все пикселы определённым цветом по маске. Маска псевдотрёхмерная.
    np.copyto(
        rgb, np.array([0,0,255], dtype=np.uint8),
        where=blue.astype(dtype=np.bool8)[:,:,None])
    np.copyto(
        rgb, np.array([255,0,0], dtype=np.uint8),
        where=red.astype(dtype=np.bool8)[:,:,None])
    np.copyto(
        rgb, np.array([255,0,255], dtype=np.uint8),
        where=np.logical_and(red, blue)[:,:,None])
#     np.copyto(rgb, np.array([0,0,255], dtype=np.uint8), where=hem.view(dtype=np.bool8)[:,:,None])

    '''An alternative for grayscale stains.
    np.dstack((rescale_intensity(mask, out_range=(0, 1)), np.zeros_like(mask),
               rescale_intensity(hem, out_range=(0, 1))))
    '''
    return rgb

################################################################################
# class AnalyticsTools(object):
#     """Plots."""
#     def __init__(self):
#         super(AnalyticsTools, self).__init__()


def draw_histogram():
    """RGB, Log.
    Думаю стоит рисовать ее как в Leica - тонкими линиями по каналам."""
    pass


def calc_stats(hemlabels, dablabels):
    """Return area fraction.

    TODO:
        * выдавать статистику по labels (хотя бы размеры)
        * размеры подсчитаны в функции фильтрации
    """
    hemarea = np.count_nonzero(hemlabels)
    dabarea = np.count_nonzero(dablabels)
    dabhem = dabarea + hemarea
    if dabhem == 0:
        return 0
    else:
        return float(dabarea) / dabhem


def calc_stats_binary(hemlabels, dablabels):
    """DAB / (All area covered with DAB or HEM)

    Sometimes DAB and HEM areas intersects.
    """
    botharea = np.count_nonzero(np.logical_or(hemlabels, dablabels))
    if botharea == 0:
        return 0
    else:
        return float(np.count_nonzero(dablabels)) / botharea


def fit_polynominal(percent):
    """Correct result basing on Immunoratio polynominal.
    """
    a = 0.00006442
    b = -0.001984
    c = 0.611
    d = 0.4321
    return a * percent ** 3 + b * percent ** 2 + c * percent + d


def autofocus():
    """help find focus position
    В оригинале статьи про watershed была функция для проверки фокусировки.
    """
    pass
