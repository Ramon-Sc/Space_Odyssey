"""
This file contains the code for different image segmentation methods.
"""

#TODO;
#Support batch processing
#Support GPU acceleration
# change SLIC to https://github.com/rosalindfranklininstitute/cuda-slic
# add Marker based watershed

class segmenter:
    def __init__(self, config):
        self.config = config

    def watershed(self, image, params):
        skimage.segmentation.watershed(image, markers=None, compactness=1, watershed_line=True)
    
    def SLIC(self, image,n_segments,compactness,slic_zero=True):
        image = np.mean(image, axis=2)  # Average across RGB channels


        
        segments = slic(image, n_segments=100, compactness=10,slic_zero=True)

        superpixels = [np.argwhere(segments == i) for i in np.unique(segments)]
        centroids = np.array([np.mean(sp, axis=0) for sp in superpixels])
        return segments, centroids


    def quickshift(self, image, params):
        pass
    
    def felzenszwalb(self, image, params):