"""
This file contains the code for different image segmentation methods.
"""


import numpy as np
from skimage import segmentation



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


        
        segments = segmentation.slic(image, n_segments=100, compactness=10,slic_zero=True)

        superpixels = [np.argwhere(segments == i) for i in np.unique(segments)]
        centroids = np.array([np.mean(sp, axis=0) for sp in superpixels])
        return segments, centroids


    def quickshift(self, image, params):
        segments = segmentation.quickshift(image, sigma=1, max_dist=10, ratio=0.5)
        return segments
    
    def felzenszwalb(self, image, params):
        segments = segmentation.felzenszwalb(image, scale=1, sigma=0.8, min_size=20)
        return segments