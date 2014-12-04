#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on 2014-01-16

@author: Eugene Dvoretsky

Immunopy image processing core.
"""

import warnings
from multiprocessing import Pool
import numpy as np
from scipy import ndimage
from skimage.exposure import histogram
from skimage.morphology import watershed
from skimage import color
from skimage.feature import peak_local_max
from skimage.util import dtype
import cv2
from PySide import QtCore
import lut
import statdata


class CalibMicro(QtCore.QObject):
    """Microscope objective calibration and size conversion.

    Instance it with default objective name (e.g. '20') or scale in um/px.

    TODO:
        * Use binning
    """

    scale_changed = QtCore.Signal(float)

    def __init__(self, objective=None, scale=None):
        """Objective __scales (um/px) from Leica Acquisition Suite *.cal.xml
        files for Leica DMI 2000 optical microscope.
        """
        super(CalibMicro, self).__init__()
        self.__scales = { '5': 9.1428482142944335E-01,
                         '10': 4.5714241071472167E-01,
                         '20': 2.2857120535736084E-01,
                         '40': 1.1428560267868042E-01,
                         '63': 7.2562287415035193E-02,
                        '100': 4.571E-02}
        self.binning = 1  # Still not using
        if objective is not None:
            self.scalename = objective
        elif scale is not None:
            self.scale = scale
        else:
            ValueError("Must specify default objective name or scale.")

    @property
    def scale(self):
        """Get current scale in um/px."""
        return self.__curr_scale  # um/px!
    @scale.setter
    def scale(self, value):
        """Set current scale in um/px."""
        if isinstance(value, str) and value in self.__scales:
            RuntimeWarning(
                """There is objective name '%s' same as argument. Are you
                really wanted to use 'scale' property?""" % \
                self.__scales[value])
        elif isinstance(value, float):
            self.__curr_scale = value
        else:
            raise ValueError("Wrong argument type %s") % type(value)

    @property
    def scalename(self):
        """Get current scalename (objective name)."""
        return self.__curr_scale_name
    @scalename.setter
    def scalename(self, value):
        """Set microscope objective scale by available scalename."""
        assert(isinstance(value, str))
        if value not in self.__scales:
            raise ValueError("Unknown microscope objective name")
        self.__curr_scale = self.__scales[value]
        self.__curr_scale_name = value
        self.scale_changed.emit(self.__curr_scale)

    def um2px(self, um=1):
        """Convert um to pixel line by current scale"""
        return um / self.__curr_scale

    def px2um(self, px=1):
        """Convert pixel line to um by current scale."""
        return px * self.__curr_scale

    def um2circle(self, diameter):
        """Diameter (um) to area of equal circle in px."""
        return np.pi * (self.um2px(diameter) / 2.0) ** 2

    def um2rect(self, diameter):
        """Diameter (um) to area of equal rectangle in px."""
        return self.um2px(diameter) ** 2

    # def set_all_scales(self):
    #     pass

    def get_all_scalenames(self):
        """Scale names list sorted from lesser to bigger magnification."""
        return sorted(self.__scales.iterkeys(),
                      key=self.__scales.__getitem__, reverse=True)


class HistogramPlotter(object):
    """Plot histogram from RGB images."""
    def __init__(self, width=256, height=100, gradient=False):
        super(HistogramPlotter, self).__init__()
        self.width = width
        self.height = height
        self.blank = np.zeros((height, width, 4), dtype=np.uint8)
        if gradient:
            self.gradient = np.linspace(0, 256, width, endpoint=False).astype(np.uint8)
        else:
            self.gradient = None

    def plot(self, rgb):
        """Return BGRA histogram picture.
        """
        bgra = self.blank.copy()
        histr = np.bincount(rgb[...,0].ravel()).astype(np.float32)
        histg = np.bincount(rgb[...,1].ravel()).astype(np.float32)
        histb = np.bincount(rgb[...,2].ravel()).astype(np.float32)
        if self.gradient is not None:
            for k, c in enumerate(histb / histb.max() * self.height):
                bgra[self.height-c:,k,0] = self.gradient[k]
                bgra[self.height-c:,k,3] = 255.
            for k, c in enumerate(histg / histg.max() * self.height):
                bgra[self.height-c:,k,1] = self.gradient[k]
                bgra[self.height-c:,k,3] = 255.
            for k, c in enumerate(histr / histr.max() * self.height):
                bgra[self.height-c:,k,2] = self.gradient[k]
                bgra[self.height-c:,k,3] = 255.
        else:
            for k, c in enumerate(histb / histb.max() * self.height):
                bgra[self.height-c:,k,0] = 255.
                bgra[self.height-c:,k,3] = 150.
            for k, c in enumerate(histg / histg.max() * self.height):
                bgra[self.height-c:,k,1] = 255.
                bgra[self.height-c:,k,3] = 150.
            for k, c in enumerate(histr / histr.max() * self.height):
                bgra[self.height-c:,k,2] = 255.
                bgra[self.height-c:,k,3] = 150.
        return bgra


class CellProcessor(object):
    """Segment and visualize cell image.

    Accept RGB image, return statistics and visualization.

    scale: init scale
    colormap: an numpy array colormap (Look Up Table)
    mp: bool, use multiprocessing
    """
    def __init__(self, scale, colormap, mp=False):
        super(CellProcessor, self).__init__()
        self.white_balance_shift = [0, 0, 0]  # RGB colors shift
        self.th_dab_shift = 0
        self.th_hem_shift = 0
        self.min_size = 80
        self.max_size = 9999999 # 3000 temporary disabled
        self.vtype = 1

        self.peak_distance = 8
        self.scale = scale
        self.blur = 2
        self.colormap = colormap
        if mp:
            self.pool = Pool(processes=2)
        else:
            self.pool = None

        self.st_dab_cell_count = 0
        self.st_hem_cell_count = 0
        self.st_dabdabhem_fraction = 0.0

    def take_assay(self):
        """Return assay object for processed image.
        """
        return statdata.Assay(
            dab_cell_count=self.st_dab_cell_count,
            hem_cell_count=self.st_hem_cell_count,
            dab_dabhemfraction=self.st_dabdabhem_fraction)

    @property
    def scale(self):
        return self.__scale
    @scale.setter
    def scale(self, value):
        assert(isinstance(value, float))
        self.__scale = value
        print('Scale changed to %f um/px (%f px/um)') % (
            self.__scale, 1.0 / self.__scale)

    @property
    def vtype(self):
        return self.__vtype
    @vtype.setter
    def vtype(self, value):
        self.__vtype = value

    @property
    def white_balance_shift(self):
        return self.__white_balance_shift
    @white_balance_shift.setter
    def white_balance_shift(self, value):
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError("White balance correction value must be present \
                as (R, G, B) list or tuple")
        self.__white_balance_shift = value

    @property
    def th_dab_shift(self):
        return self.__threshold_dab_shift
    @th_dab_shift.setter
    def th_dab_shift(self, value):
        self.__threshold_dab_shift = value

    @property
    def th_hem_shift(self):
        return self.__threshold_hem_shift
    @th_hem_shift.setter
    def th_hem_shift(self, value):
        self.__threshold_hem_shift = value

    @property
    def min_size(self):
        return self.__min_size
    @min_size.setter
    def min_size(self, value):
        if value > 0:
            self.__min_size = value

    @property
    def max_size(self):
        return self.__max_size
    @max_size.setter
    def max_size(self, value):
        if value > 0:
            self.__max_size = value

    @property
    def peak_distance(self):
        return self.__peak_distance
    @peak_distance.setter
    def peak_distance(self, value):
        if value > 0:
            self.__peak_distance = value

    def process(self, image):
        """Segmentation and statistical calculation.
        """
        # Light source correction
        # correct_wb(image, self.__white_balance_shift)
        if self.vtype == 0:
            return image
        rgb = image.copy()

        # Enhancement (can abuse threshold output)
        meaned = cv2.blur(rgb, (self.blur, self.blur))

        # Resize to fixed scale
        scaled = rescale(meaned, self.__scale)

        # Unmix stains
        hdx = color_deconvolution(scaled, color.hdx_from_rgb)
        hem = hdx[:,:,0]
        dab = hdx[:,:,1]

        # MULTICORE -----------------------------------------------------------
        if self.pool:
            dproc = self.pool.apply_async(worker, (dab, self.__threshold_dab_shift, self.peak_distance, self.min_size, self.max_size))
            hproc = self.pool.apply_async(worker, (hem, self.__threshold_hem_shift, self.peak_distance, self.min_size, self.max_size))
            dabfiltered, self.st_dab_cell_count = dproc.get(timeout=10)
            hemfiltered, self.st_hem_cell_count = hproc.get(timeout=10)
        else:
            dabfiltered, self.st_dab_cell_count = worker(dab, self.__threshold_dab_shift, self.peak_distance, self.min_size, self.max_size)
            hemfiltered, self.st_hem_cell_count = worker(hem, self.__threshold_hem_shift, self.peak_distance, self.min_size, self.max_size)
        # MULTICORE END -------------------------------------------------------

        # Stats
        # self.stCellFraction =  float(dabfnum) / (hemfnum + dabfnum + 0.001) * 100
        # self.stDabHemFraction = areaFraction(hemfiltered, dabfiltered) * 100
        self.st_dabdabhem_fraction = areaDisjFraction(hemfiltered, dabfiltered) * 100

        # Visualization
        if self.vtype == 1:
            overlay = drawOverlay(scaled, dabfiltered, hemfiltered)
        elif self.vtype == 2:
            overlay = lut.apply_lut(dabfiltered, self.colormap)
        elif self.vtype == 3:
            overlay = lut.apply_lut(hemfiltered, self.colormap)
        else:
            overlay = drawOverlay(scaled, dabfiltered, hemfiltered)
            dabcolored = lut.apply_lut(dabfiltered, self.colormap)
            hemcolored = lut.apply_lut(hemfiltered, self.colormap)
            cv2.putText(scaled, 'Area disj fr. %.1f %%' % (self.st_dabdabhem_fraction), (12,65), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 0), thickness=5)
            cv2.putText(overlay, 'Colocalization', (12,65), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 0), thickness=5)
            cv2.putText(dabcolored, 'DAB %3.d objects' % self.st_dab_cell_count, (12,65), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 0), thickness=5)
            cv2.putText(hemcolored, 'HEM %3.d objects' % self.st_hem_cell_count, (12,65), cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 0), thickness=5)
            return montage(scaled, hemcolored, overlay, dabcolored)

        cv2.putText(overlay, 'Num D%3.d/H%3.d' % (self.st_dab_cell_count, self.st_hem_cell_count), (2,25), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
        cv2.putText(overlay, 'DAB/DAB||HEM %.2f %%' % self.st_dabdabhem_fraction, (2,55), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
        return overlay


def worker(stain, threshold_shift, peak_distance, min_size, max_size):
    """Process each stain.

    Return filtered objects and their count.
    Would not work with processes as class method.
    """
    stth = threshold_yen(stain, shift=threshold_shift)
    stmask = stain < stth
    stmed = ndimage.filters.median_filter(stmask, size=2)
    stedt = cv2.distanceTransform(
        stmed.view(np.uint8), distanceType=cv2.cv.CV_DIST_L2, maskSize=3)
    st_max = peak_local_max(
        stedt, min_distance=peak_distance, exclude_border=False, indices=False)
    stlabels, stlnum = watershedSegmentation(stmed, stedt, st_max)
    stfiltered, stfnum = filterObjects(
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


def getCentralRect(width, height, divisor=1):
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


def setMmcResolution(mmc, width, height):
    """Select rectangular ROI in center of the frame.
    """
    x = (mmc.getImageWidth() - width) / 2
    y = (mmc.getImageHeight() - height) / 2
    if not all((x > 0, y > 0)):
        raise ValueError('ROI w%d h%d is out of image size' % (width, height))
    mmc.setROI(x, y, width, height)


def correct_wb(rgb, rgb_shift):
    """Correct white balance inplace.
    """
    # Numpy does not provide saturated inplace math
    if rgb_shift[0] != 0:
        rgb[...,0] = cv2.add(rgb[...,0], rgb_shift[0])
    if rgb_shift[1] != 0:
        rgb[...,1] = cv2.add(rgb[...,1], rgb_shift[1])
    if rgb_shift[2] != 0:
        rgb[...,2] = cv2.add(rgb[...,2], rgb_shift[2])


def normalize(img):
    """G. Landini proposal.
    http://imagejdocu.tudor.lu/doku.php?id=howto:working:how_to_correct_background_illumination_in_brightfield_microscopy
    corrected = (Specimen - Darkfield) / (Brightfield - Darkfield) * 255
    """
    immin = img.min()
    return (img - immin) / (img.max() - immin).astype(np.float32)


def rescale(source, scale):
    """If scale > 2 px/um, image will be downsampled.

    scale in um/px
    """
    if scale == 0.5:
        return source
    scl_factor = scale / 0.5  # Target scale - 0.5 um/px (2 px/um)
    if scl_factor > 1:
        warnings.warn("Input image resolution worse than 0.5 um/px. Upscaling will be used.", UserWarning)
    fy = int(round(source.shape[0] * scl_factor))
    fx = int(round(source.shape[1] * scl_factor))
    # cv2.INTER_CUBIC
    return cv2.resize(source, dsize=(fx, fy), interpolation=cv2.INTER_LINEAR)


def color_deconvolution(rgb, conv_matrix):
    """Unmix stains for histogram analysis.
    :return: Image values in normal space (not optical density i.e. log space)
             and in range 0...1.
    :rtype: float array
    """
    rgb = dtype.img_as_float(rgb, force_copy=True)
    rgb += 1
    stains = np.dot(np.reshape(-np.log10(rgb), (-1, 3)), conv_matrix)
    stains = np.exp(-stains)
    return np.reshape(stains, rgb.shape)


def threshold_isodata(image, nbins=256, shift=None, max_limit=None, min_limit=None):
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
        Shift threshold value by percent up (positive) or down (negative).
    max_limit : int, optional
        Percent 0-100. If calculated threshold higher then max_limit,
        return corresponding to max_limit threshold value.
    min_limit : int, optional
        Percent 0-100. If calculated threshold lower then min_limit,
        return corresponding to min_limit threshold value.

    Returns
    -------
    threshold : float or int, corresponding input array dtype.
        Upper threshold value. All pixels intensities that less or equal of
        this value assumed as background.
        `foreground (cells) > threshold >= background`.

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
    if max_limit is not None and min_limit is not None:
        if min_limit > max_limit:
            raise ValueError('min_limit greater then max_limit')
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

    ptp = (bin_centers[-1] - bin_centers[0]) / 100.  # Peek to peek range
    if shift:
        threshold += ptp * shift
    print("Threshold value shift", (threshold - bin_centers[0]) / float(ptp))
    if max_limit is not None:
        # Limit foreground
        lim = bin_centers[0] + ptp * max_limit
        if threshold > lim:
            threshold = lim
    if min_limit is not None:
        # Limit background
        lim = bin_centers[0] + ptp * min_limit
        if threshold < lim:
            threshold = lim
    return threshold


def threshold_yen(image, nbins=256, shift=None):
    """Return threshold value based on Yen's method.

    Parameters
    ----------
    image : array
        Input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    shift : int, optional
        Shift threshold value by percent up (positive) or down (negative).

    Returns
    -------
    threshold : float
        Upper threshold value. All pixels intensities that less or equal of
        this value assumed as foreground.

    References
    ----------
    .. [1] Yen J.C., Chang F.J., and Chang S. (1995) "A New Criterion
           for Automatic Multilevel Thresholding" IEEE Trans. on Image
           Processing, 4(3): 370-378
    .. [2] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
           Techniques and Quantitative Performance Evaluation" Journal of
           Electronic Imaging, 13(1): 146-165,
           http://www.busim.ee.boun.edu.tr/~sankur/SankurFolder/Threshold_survey.pdf
    .. [3] ImageJ AutoThresholder code, http://fiji.sc/wiki/index.php/Auto_Threshold

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> thresh = threshold_yen(image)
    >>> binary = image <= thresh
    """
    hist, bin_centers = histogram(image, nbins)
    # On blank images (e.g. filled with 0) with int dtype, `histogram()`
    # returns `bin_centers` containing only one value. Speed up with it.
    if bin_centers.size == 1:
        return bin_centers[0]

    # Calculate probability mass function
    pmf = hist.astype(np.float32) / hist.sum()
    P1 = np.cumsum(pmf)  # Cumulative normalized histogram
    P1_sq = np.cumsum(pmf ** 2)
    # Get cumsum calculated from end of squared array:
    P2_sq = np.cumsum(pmf[::-1] ** 2)[::-1]
    # P2_sq indexes is shifted +1. I assume, with P1[:-1] it's help avoid '-inf'
    # in crit. ImageJ Yen implementation replaces those values by zero.
    crit = np.log(((P1_sq[:-1] * P2_sq[1:]) ** -1) *
                  (P1[:-1] * (1.0 - P1[:-1])) ** 2)

    threshold = bin_centers[crit.argmax()]
    ptp = (bin_centers[-1] - bin_centers[0]) / 100.  # Peek to peek range
    if shift:
        threshold += ptp * shift
    print("Threshold value shift", (threshold - bin_centers[0]) / float(ptp))
    return threshold


def watershedSegmentation(mask, edt, local_maxi):
    """Segment clumped nuclei.

    Something like Malpica N, de Solorzano CO, Vaquero JJ, Santos et all
    'Applying watershed algorithms to the segmentation of clustered nuclei.'
    Cytometry 28, 289-297 (1997).
    """
    markers, num_markers = ndimage.label(local_maxi)
    labels = watershed(-edt, markers, mask=mask)
    return labels, num_markers


def filterObjects(labels, num=None, min_size=150, max_size=2000, in_place=False):
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


def drawOverlay(srcrgb, red, blue):
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


def drawMasks(srcrgb, red, blue):
    """Draw objects on image"""
    """
    Draws on source image with different colors under stains masks.

    После завершения работы с выделением DAB и гематоксилина, проверь,
    пересекаются ли эти маски. Это можно сделать, заменяя один канал:
    np.copyto(bgr[:,:,0], 255, where=mask.view(dtype=np.bool8))
    """
    rgb = srcrgb.copy()
    # Replaces all pixels with specifed color by mask. Mask is faked-3d.
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


def montage(tl, tr, bl, br):
    """All shapes must be same. Memory-save.
    """
    h, w, d = tl.shape
    blank = np.empty((h * 2, w * 2, d), dtype=np.uint8)
    blank[:h, :w] = tl
    blank[h:, :w] = bl
    blank[:h, w:] = tr
    blank[h:, w:] = br
    return blank


def areaFraction(hemlabels, dablabels):
    """Return simple area fraction.

    TODO:
        * выдавать статистику по labels (хотя бы размеры)
        * размеры подсчитаны в функции фильтрации
    """
    hemarea = np.count_nonzero(hemlabels)
    dabarea = np.count_nonzero(dablabels)
    dabhem = dabarea + hemarea
    if dabhem == 0:
        return 0.
    else:
        return float(dabarea) / dabhem


def areaDisjFraction(hemlabels, dablabels):
    """DAB / (All area covered with DAB or HEM i.e. disjunction).

    Sometimes DAB and HEM areas intersects.
    """
    botharea = np.count_nonzero(np.logical_or(hemlabels, dablabels))
    if botharea == 0:
        return 0.
    else:
        return float(np.count_nonzero(dablabels)) / botharea


def fitPolynominal(percent):
    """Correct result basing on Immunoratio polynominal.
    """
    a = 0.00006442
    b = -0.001984
    c = 0.611
    d = 0.4321
    return a * percent ** 3 + b * percent ** 2 + c * percent + d


def autofocus():
    """Help find focus position.

    There is an autofocusing algorytm in original watershed article.
    """
    pass
