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
import cv2


class CalibMicro(object):
    """Microscope objective calibration and size conversion.
    
    Instance it with default objective scale as parameter.
    """
    
    def __init__(self, scale):
        """Objective scales (um/px) from Leica Acquisition Suite *.cal.xml
        for Leica DMI 2000 optical microscope
        """
        super(CalibMicro, self).__init__()
        self.scales = {'5': 9.1428482142944335E-01,
                       '10': 4.5714241071472167E-01,
                       '20': 2.2857120535736084E-01,
                       '63': 7.2562287415035193E-02}
        self.cur_scale = None
        self.set_curr_scale(scale)  # objective name
#         self.roi = (2048, 1536)
#         self.binning = 1, 2  # Не уверен, что этот параметр нужен.

    def um2px(self, um, scale=None):
        """Convert um to pixel line."""
        return um / self.cur_scale

    def px2um(self, um, scale=None):
        """Convert pixel line to um."""
        return um * self.cur_scale
    
    def um2circle(self, diameter):
        """Диаметр (um) в площадь эквивалентного круга в px."""
        return np.pi * (self.um2px(diameter) / 2.0) ** 2
    
    def um2rect(self, diameter):
        """Диаметр (um) в площадь эквивалентного квадрата в px."""
        return self.um2px(diameter) ** 2
    
    def set_curr_scale(self, scale):
        """Set microscope scale from available."""
        assert(isinstance(scale, str))
        self.cur_scale = self.scales[scale]
    
    def get_curr_scale(self):
        return self.cur_scale
    
    # def set_all_scales(self):
    #     pass
    
    # def get_all_scales(self):
    #     pass

################################################################################
# class Improc(object):
#     """Обработка и анализ изображений."""
#     def __init__(self):
#         super(Improc, self).__init__()


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


def get_random_cm():
    """Generate random colormap for easy visual distinguishing of objects."""
    import random
    from matplotlib import colors as mc
    from matplotlib.cm import jet

    colors = map(jet, range(0, 256, 4))
    random.shuffle(colors)
    colors[0] = (0., 0., 0., 1.)
    return mc.ListedColormap(colors)


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


def overlay():
    pass


def draw_masks(srcrgb, mask):
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
        rgb, np.array([255,0,0], dtype=np.uint8),
        where=mask.view(dtype=np.bool8)[:,:,None])
#     np.copyto(rgb, np.array([0,0,255], dtype=np.uint8), where=hem.view(dtype=np.bool8)[:,:,None])
    
    '''An alternative variant for grayscale stains.
    np.dstack((rescale_intensity(mask, out_range=(0, 1)), np.zeros_like(mask),
               rescale_intensity(hem, out_range=(0, 1))))
    '''
    return rgb

################################################################################
# class AnalyticsTools(object):
#     """Графики."""
#     def __init__(self):
#         super(AnalyticsTools, self).__init__()
    

def draw_histogram():
    """RGB, RGB, Log.
    Думаю стоит рисовать ее как в Leica - тонкими линиями по каналам."""
    pass


def calc_stats(hemlabels, dablabels):
    """Return area fraction.
    выдавать статистику по labels (хотя бы размеры)
    размеры подсчитаны в функции фильтрации
    """
    hemarea = np.count_nonzero(hemlabels)
    dabarea = np.count_nonzero(dablabels)
    dabhem = dabarea + hemarea
    if dabhem == 0:
        return 0
    else:
        return float(dabarea) / dabhem

def autofocus():
    """help find focus position
    В оригинале статьи про watershed была функция для проверки фокусировки.
    """
    pass
