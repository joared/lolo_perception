
import cv2 as cv
import numpy as np
from lolo_perception.image_processing import contourCentroid, contourRatio

class LightSource:
    def __init__(self, cnt, intensity, overlappingLightSource=None):

        self.cnt = cnt
        self.center = contourCentroid(cnt)
        self.area = cv.contourArea(cnt)
        self.circleExtent = contourRatio(cnt)
        self.perimeter = max(1, cv.arcLength(cnt,True))
        self.circularity = 4*np.pi*self.area/self.perimeter # https://learnopencv.com/blob-detection-using-opencv-python-c/
        self.circleCenter, self.radius = cv.minEnclosingCircle(cnt)

        self.intensity = intensity

        self.rmseUncertainty = self.radius

        self.overlappingLightSource = overlappingLightSource

    
    def isOverlapping(self):
        return self.overlappingLightSource is not None


    def __repr__(self):
        return "LightSource - center={}, intensity={}".format(self.center, self.intensity)