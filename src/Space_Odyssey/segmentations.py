"""
This file contains the code for different image segmentation methods.
"""


import numpy as np
from skimage import segmentation, morphology, filters
import cv2


#TODO;
#Support batch processing
#Support GPU acceleration
# change SLIC to https://github.com/rosalindfranklininstitute/cuda-slic
# add Marker based watershed
# graph constructiokn in different script

class segmenter:
    def __init__(self, config):
        self.config = config

    def watershed(self, image, params):
        segments = segmentation.watershed(image, markers=None, compactness=1, watershed_line=True)
        return segments
    
    def slic(self, image,n_segments,compactness,slic_zero=True):

        
        image = np.mean(image, axis=2)  # Average across RGB channels

        print(image.shape)
        segments = segmentation.slic(image, n_segments=n_segments, compactness=compactness, slic_zero=slic_zero, channel_axis=None)

        superpixels = [np.argwhere(segments == i) for i in np.unique(segments)]
        centroids = np.array([np.mean(sp, axis=0) for sp in superpixels])
        return segments, centroids


    def quickshift(self, image):
        segments = segmentation.quickshift(image, sigma=1, max_dist=10, ratio=0.5)
        return segments
    
    def felzenszwalb(self, image, params):
        segments = segmentation.felzenszwalb(image, scale=1, sigma=0.8, min_size=20)
        return segments

    def dbscan(self, image, params):
        # Real-Time Superpixel Segmentation
        # by DBSCAN Clustering Algorithm Shen et al. 2016
        # IEEE TRANSACTIONS ON IMAGE PROCESSING,
        #
        # https://github.com/shenjianbing/realtimesuperpixel Repo private or deleted
        #
        pass

    def dilation(self, image, params):
        dilation = morphology.dilation(image,cval=0,)
        return dilation
    
    def morphological_opening(self, image, params):
        opening = morphology.opening(image, footprint=None)
        return opening
    
    def binarize(self, image, params):
        """
        Binarize an image using Otsu's thresholding method.
        
        Parameters:
        -----------
        image : ndarray
            Input image (can be RGB or grayscale)
        params : dict
            Optional parameters:
            - method: 'otsu' (default), 'adaptive', 'manual'
            - threshold: manual threshold value (0-255) if method='manual'
        
        Returns:
        --------
        binary : ndarray
            Binary image (0 or 255)
        """
        # Convert to grayscale if RGB
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        method = params.get('method', 'otsu') if params else 'otsu'
        channel = params.get('channel', 'all') if params else 'all'
        
        if method == 'otsu':
            if channel == 'all':
                threshold = filters.threshold_otsu(gray)
                binary = (gray > threshold).astype(np.uint8) * 255
            else:    
                if channel == 'r':
                    binary = image[:,:,0]
                elif channel == 'g':
                    binary = image[:,:,1]
                elif channel == 'b':
                    binary = image[:,:,2]
    
        elif method == 'adaptive':
            # Adaptive thresholding
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        elif method == 'manual':
            threshold = params.get('threshold', 127)
            binary = (gray > threshold).astype(np.uint8) * 255
        else:
            # Default to Otsu
            threshold = filters.threshold_otsu(gray)
            binary = (gray > threshold).astype(np.uint8) * 255
        
        return binary
    
    def multiotsu(self, image, params):
        thresholds = filters.threshold_multiotsu(image)
        regions = np.digitize(image, bins=thresholds)
        # Create a mask for pixels with value 0 (black) (nuclei)
        nuclei_mask = regions == 0
        return nuclei_mask

