# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 17:46:02 2015

@author: eusebio
"""

import numpy as np
from skimage.feature import local_binary_pattern # Local binary pattern
from skimage.feature import hog


"""
    This class implements the LBPU features computation to use in classification

    * See: https://en.wikipedia.org/wiki/Local_binary_patterns
"""

class LBPUFeatures(object):
    """
        This class
    """

    __radius__ = 3
    __nPoints__ = __radius__ * 8
    __image__ = None
    __LBP_BINS__ = 59

    def __init__(self, image, LBP_BINS = 59, radius = 3):
        self.__image__ = image
        self.__LBP_BINS__ = LBP_BINS
        self.__radius__ = radius
        self.__nPoints__ = self.__radius__ * 8


    def getFeatures(self):
        lbp = local_binary_pattern(self.__image__, self.__nPoints__, self.__radius__, method='uniform')
        lbp_hist, bin_edges = np.histogram(lbp, self.__LBP_BINS__)

        # Histogram normalization
        lbp_hist_norm = sum(abs(lbp_hist))
        lbp_hist_l1sqrtnorm = np.sqrt(lbp_hist/float(lbp_hist_norm))

        return lbp_hist_l1sqrtnorm

"""
    This implements a different version of the LBPU features. In this case the
    computation of the features divide the image (subimage), into several
    subimages, and then computes LBPU of these subimages and concatenates the
    values as the features vector.
"""

class LBPUMultiblockFeatures(object):
    __radius__ = 3
    __nPoints__ = __radius__ * 8
    __image__ = None
    __LBP_BINS__ = 59

    def __init__(self, image, LBP_BINS = 59, radius = 3):
        self.__image__ = image
        self.__LBP_BINS__ = LBP_BINS
        self.__radius__ = radius
        self.__nPoints__ = self.__radius__ * 8


    def getFeatures(self):
        # Subimage1
        lbp1 = local_binary_pattern(self.__image__[0:13,0:13], self.__nPoints__, self.__radius__, method='uniform')
        lbp1_hist, bin_edges = np.histogram(lbp1, self.__LBP_BINS__)
        # Histogram normalization
        lbp1_hist_norm = sum(abs(lbp1_hist))
        lbp1_hist_l1sqrtnorm = np.sqrt(lbp1_hist/float(lbp1_hist_norm))

        # Subimage2
        lbp2 = local_binary_pattern(self.__image__[0:13,14:27], self.__nPoints__, self.__radius__, method='uniform')
        lbp2_hist, bin_edges = np.histogram(lbp2, self.__LBP_BINS__)
        # Histogram normalization
        lbp2_hist_norm = sum(abs(lbp2_hist))
        lbp2_hist_l1sqrtnorm = np.sqrt(lbp2_hist/float(lbp2_hist_norm))

        # Subimage3
        lbp3 = local_binary_pattern(self.__image__[14:27,0:13], self.__nPoints__, self.__radius__, method='uniform')
        lbp3_hist, bin_edges = np.histogram(lbp3, self.__LBP_BINS__)
        # Histogram normalization
        lbp3_hist_norm = sum(abs(lbp3_hist))
        lbp3_hist_l1sqrtnorm = np.sqrt(lbp3_hist/float(lbp3_hist_norm))

        # Subimage4
        lbp4 = local_binary_pattern(self.__image__[14:27,14:27], self.__nPoints__, self.__radius__, method='uniform')
        lbp4_hist, bin_edges = np.histogram(lbp4, self.__LBP_BINS__)
        # Histogram normalization
        lbp4_hist_norm = sum(abs(lbp4_hist))
        lbp4_hist_l1sqrtnorm = np.sqrt(lbp4_hist/float(lbp4_hist_norm))

        # Subimage5
        lbp5 = local_binary_pattern(self.__image__[7:20,7:20], self.__nPoints__, self.__radius__, method='uniform')
        lbp5_hist, bin_edges = np.histogram(lbp5, self.__LBP_BINS__)
        # Histogram normalization
        lbp5_hist_norm = sum(abs(lbp5_hist))
        lbp5_hist_l1sqrtnorm = np.sqrt(lbp5_hist/float(lbp5_hist_norm))

        feat = np.array([lbp1_hist_l1sqrtnorm, lbp2_hist_l1sqrtnorm, lbp3_hist_l1sqrtnorm, lbp4_hist_l1sqrtnorm, lbp5_hist_l1sqrtnorm])

        return feat.flatten()




"""
    This class implements the HOG features computation to use in classification

    * See: https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
"""

class HOGFeatures(object):
    """
        This class implements the HOG features
    """

    __image__ = None
    __orientations__ = 8
    __pixels_per_cell__ = (7, 7)
    __cells_per_block__ = (3, 3)

    def __init__(self, image, orientations = 8, pixels_per_cell=(7, 7), cells_per_block=(3, 3)):
        self.__image__ = image
        self.__orientations__ = orientations
        self.__pixels_per_cell__ = pixels_per_cell
        self.__cells_per_block__ = cells_per_block


    def getFeatures(self):

        hog_feat = hog(self.__image__, self.__orientations__, self.__pixels_per_cell__, self.__cells_per_block__)

        return hog_feat


"""
    This class implements the LBPUMultiblock + HOG features computation to use in classification
"""

class LBPUMultiBlockAndHOGFeatures(object):
    """
        This class implements the HOG features
    """

    __image__ = None

    def __init__(self, image):
        self.__image__ = image


    def getFeatures(self):

        lbpu = LBPUMultiblockFeatures(self.__image__)

        hog = HOGFeatures(self.__image__)

        feat = np.concatenate((lbpu.getFeatures(), hog.getFeatures()))

        return feat
