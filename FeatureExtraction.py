import cv2
import numpy as np
import os
from skimage import feature
from sklearn.externals import joblib



class LBP:
    def __init__(self, p, r):
        # store the number of points p and radius r
        self.p = p
        self.r = r

    def getLBPH(self, image, eps=1e-7):
        # compute the LBP representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.p, self.r, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.p + 3), range=(0, self.p + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the LBPH
        return hist, lbp

def getFeatureVector(feature_path, lbp):
    feature_img = cv2.imread(feature_path)
    hsv_feature_img = cv2.cvtColor(feature_img, cv2.COLOR_BGR2HSV)
    # convert to hsv as this is less affected by illumination
    h_feature_img = hsv_feature_img[:, :, 0]

    h_feature_img_lbph, _ = lbp.getLBPH(h_feature_img)

    return h_feature_img_lbph


