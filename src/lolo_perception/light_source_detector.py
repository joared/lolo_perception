#!/usr/bin/env python
import sys
import inspect

import cv2 as cv
import numpy as np
import lolo_perception.py_logging as logging
from scipy import signal
from lolo_perception.image_processing import circularKernel, contourCentroid, contourRatio, drawInfo, localPeak, localPeakWindowed, localPeakFloodFill, regionOfInterest
from lolo_perception.light_source import LightSource
from lolo_perception.light_source_filter import MultiLightSourceFilter


def printLightSourceDetectors():
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if obj != AbstractLightSourceDetector and issubclass(obj, AbstractLightSourceDetector):
            print(name)


def getLightSourceDetectorFromName(className):
    """
    Returns a class reference to the light source detector with the specified className.
    The class must be a subclass of AbstractLightSourceDetector, otherwise an exception is thrown.
    """
    for name, cls in inspect.getmembers(sys.modules[__name__]):
        if name == className:
            if cls != AbstractLightSourceDetector and issubclass(cls, AbstractLightSourceDetector):
                return cls
            else:
                print("'{}' is not a light source detector, make sure your light source detector inherits from '{}'".format(cls, AbstractLightSourceDetector))
                print("Available light source detectors are:")
                printLightSourceDetectors()
                raise Exception("Invalid light source detector '{}'".format(className))
    else:
        print("Available light source detectors are:")
        printLightSourceDetectors()
        raise Exception("'{}' not found.".format(className))


class AbstractLightSourceDetector:
    def __init__(self):
        self.img = np.zeros((10, 10)) # store the last processed image

    def __repr__(self):
        return "Light source detector name"

    def detect(self, gray, nFeatures, additionalCandidates=0, estDSPose=None, roiMargin=None, drawImg=None):
        candidates = []
        roiCnt = None
        return self.img, candidates, roiCnt


class ModifiedHATS(AbstractLightSourceDetector):
    MODE_SIMPLE = "simple"
    MODE_PEAK = "peak"
    MODE_VALLEY = "valley"
    
    def __init__(self,  
                 peakMargin=0, 
                 minArea=1, 
                 minRatio=0, 
                 maxIntensityChange=0.7, 
                 blurKernelSize=5,
                 noNewCandidatesIntensityThresh=10, 
                 mode="valley", 
                 ignorePeakAtMax=False, 
                 separationThresh=150,
                 showHistogram=False,
                 lightSourceFilters=None):

        self.threshold = 256
        self.peakMargin = peakMargin # 0-1
        self.minArea = minArea
        self.minRatio = minRatio
        self.maxIntensityChange = maxIntensityChange
        self.blurKernelSize = blurKernelSize
        self.noNewCandidatesIntensityThresh = noNewCandidatesIntensityThresh

        # https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
        self.sigma = 0.3*((self.blurKernelSize-1)*0.5 - 1) + 0.8 # TODO: does this work properly?
        self.thresholdType = cv.THRESH_BINARY

        assert mode in (self.MODE_SIMPLE, self.MODE_PEAK, self.MODE_VALLEY), "Invalid mode '', must be '{}, {}' or '{}'".format(mode, self.MODE_SIMPLE, self.MODE_PEAK, self.MODE_VALLEY)
        self.mode = mode
        self.ignorePeakAtMax = ignorePeakAtMax
        self.separationThresh = separationThresh
        self.showHistogram = showHistogram

        self.kernel = np.ones((21,21)) # for light source separation

        self.img = np.zeros((10, 10)) # store the last processed image
        self.iterations = 0

        self.lightSourceFilter = MultiLightSourceFilter(*lightSourceFilters)


    def __repr__(self):
        return "HATS"


    def _calcCandidates(self, contours):
        candidates = []
        removedCandidates = []
        for cnt in contours:
            if self.lightSourceFilter(cnt, 255):
                candidates.append(cnt)
            else:
                removedCandidates.append(cnt)

        return candidates, removedCandidates


    def _sortLightSources(self, candidates, nCandidates):
        candidates.sort(key=lambda p: p.area, reverse=True)
        candidates = candidates[:nCandidates]
        
        return candidates


    def detect(self, gray, nFeatures, additionalCandidates=0, estDSPose=None, roiMargin=None, drawImg=None):

        roiCnt = None
        offset = (0, 0)
        if estDSPose:
            featurePointsGuess = estDSPose.reProject()
            roiMargin += max([ls.radius for ls in estDSPose.associatedLightSources])
            (x, y, w, h), roiCnt = regionOfInterest(featurePointsGuess, wMargin=roiMargin, hMargin=roiMargin, shape=gray.shape)

            minIntensity = min([ls.intensity for ls in estDSPose.associatedLightSources])
            minIntensity = self.maxIntensityChange*minIntensity

            offset = (x, y)
            gray = gray[y:y+h, x:x+w]
            _, gray = cv.threshold(gray, minIntensity, 256, cv.THRESH_TOZERO)

        img = gray.copy()
        img = cv.GaussianBlur(img, (self.blurKernelSize,self.blurKernelSize), self.sigma)

        hist = cv.calcHist([img], [0], None, [256], [0,256])
        hist = hist.ravel()

        if self.mode == self.MODE_SIMPLE:
            # Go through every intensity level
            histPeaks = range(255)
        if self.mode == self.MODE_VALLEY:
            # find valleys
            histPeaks, _ = signal.find_peaks(-hist)
        else:
            # find peaks
            histPeaks, _ = signal.find_peaks(hist)
            histPeaks = list(histPeaks)
            if hist[-1] > hist[-2]:
                histPeaks.append(255)
        
        histPeaks = list(histPeaks)

        if self.ignorePeakAtMax and np.max(img) == 255:
            self.threshold = 254
        else:
            if len(histPeaks) > 0:
                self.threshold = histPeaks.pop()-1
            else:
                self.threshold = np.max(img)-1

        ret, imgTemp = cv.threshold(img, self.threshold, 255, self.thresholdType)
        _, contours, hier = cv.findContours(imgTemp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        candidates, removedCandidates = self._calcCandidates(contours)
        prevNCandidates = len(candidates)

        img = img.copy()
        i = 0
        lastThreshold = self.threshold
        while True:
            i += 1

            if len(histPeaks) > 0:
                self.threshold = histPeaks.pop()-1
            else:
                break

            ret, imgTemp = cv.threshold(img, self.threshold, 255, self.thresholdType)

            _, contours, hier = cv.findContours(imgTemp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            newCandidates, removedCandidates = self._calcCandidates(contours)

            # Continue until we get less candidates (we want the most nof candidates)
            if len(newCandidates) < prevNCandidates:
                break
            else:
                # We have detected at least the amount of light sources that we need
                candidates = newCandidates
                prevNCandidates = len(candidates)

                # We stop if the intensity change is to large
                if lastThreshold - self.threshold > self.noNewCandidatesIntensityThresh:
                    break
                
        if self.peakMargin > 0 and len(histPeaks) > 0:
            self.threshold = histPeaks.pop()-1
            ret, imgTemp = cv.threshold(img, self.threshold, 255, self.thresholdType)
            _, contours, hier = cv.findContours(imgTemp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            candidates, removedCandidates = self._calcCandidates(contours)

        logging.debug("HATS iterations: {}, Threshold: {}".format(i, self.threshold))
        self.iterations = i
        self.img = imgTemp

        candidates = [LightSource(cnt+offset, self.threshold) for cnt in candidates]
        candidates = self._sortLightSources(candidates, nFeatures+additionalCandidates)

        # Draw the candidates and non-candidates on the drawImg
        if drawImg is not None:
            for ls in candidates:
                cnt = ls.cnt
                cx, cy = contourCentroid(cnt)
                r = contourRatio(cnt)
                drawInfo(drawImg, (cx+25,cy-15), str(r), fontScale=0.5)
                cv.drawContours(drawImg, [cnt], 0, (255,0,0), -1)

            for cnt in removedCandidates:
                r = contourRatio(cnt+offset)
                cx, cy = contourCentroid(cnt+offset)
                drawInfo(drawImg, (cx+25,cy-15), str(r), color=(0,0,255), fontScale=0.5)
                cv.drawContours(drawImg, [cnt+offset], 0, (0,0,255), -1)

        self.img, candidates = separateOverExposedCandidatesSimple(candidates, 
                                                                   self.separationThresh,
                                                                   gray.shape, 
                                                                   offset=offset, 
                                                                   drawImg=drawImg)

        candidates = candidates[:nFeatures+additionalCandidates]

        return self.img, candidates, roiCnt


class LocalPeak(AbstractLightSourceDetector):
    def __init__(self, 
                 kernelSize, 
                 pMin, 
                 pMax, 
                 maxIntensityChange, 
                 minArea, 
                 minCircleExtent, 
                 blurKernelSize, 
                 ignorePAtMax, 
                 maxIter, 
                 separationFraction,
                 lightSourceFilters):
        """
        kernelSize - size of the local max kernel
        pMin - minimum decimal fraction used to 
        pMax -
        """
        self.kernelSize = kernelSize
        self.kernel = np.ones((self.kernelSize,self.kernelSize)) 
        self.pMin = pMin
        self.pMax = pMax
        self.img = np.zeros((10, 10), dtype=np.uint8) # store the last processed image
        self.maxIntensityChange = maxIntensityChange

        self.minArea = minArea
        self.minCircleExtent = minCircleExtent

        self.blurKernelSize = blurKernelSize
        self.sigma = 0.3*((self.blurKernelSize-1)*0.5 - 1) + 0.8

        self.ignorePAtMax = ignorePAtMax

        self.maxIter = maxIter
        self.iterations = 0

        self.separationFraction = separationFraction

        self.lightSourceFilter = MultiLightSourceFilter(*lightSourceFilters)


    def __repr__(self):
        return "Local Peak"
    

    def _sortLightSources(self, candidates, nCandidates):
        candidates.sort(key=lambda p: (p.intensity, p.area), reverse=True)
        candidates = candidates[:nCandidates]
        
        return candidates


    def detect(self, gray, nFeatures, additionalCandidates=0, estDSPose=None, roiMargin=None, drawImg=None):

        offset = (0, 0)
        roiCnt = None

        peakMargin = 3 # TODO: should work with 0 but needs proper testing
        minIntensity = 0
        self.minArea = 0
        if estDSPose:
            featurePointsGuess = estDSPose.reProject()
            roiMargin += max([ls.radius for ls in estDSPose.associatedLightSources])
            (x, y, w, h), roiCnt = regionOfInterest(featurePointsGuess, wMargin=roiMargin, hMargin=roiMargin, shape=gray.shape)

            offset = (x, y)
            gray = gray[y:y+h, x:x+w]

            minIntensity = min([ls.intensity for ls in estDSPose.associatedLightSources])
            minIntensity = self.maxIntensityChange*minIntensity

            # TODO: needs testing
            peakMargin = 0
        
        if self.blurKernelSize > 0:
            gray = cv.GaussianBlur(gray, (self.blurKernelSize,self.blurKernelSize), self.sigma)

        # TODO: localPeakFloodFill might be more efficient
        (localMaxImg, 
        localMaxImgMasked, 
        peakCenters, 
        peakContours,
        iterations) = localPeak(gray, 
                                kernel=self.kernel, 
                                pMin=self.pMin,
                                pMax=self.pMax, 
                                n=nFeatures+additionalCandidates,
                                minThresh=minIntensity,
                                margin=peakMargin,
                                ignorePAtMax=self.ignorePAtMax,
                                offset=offset,
                                maxIter=self.maxIter,
                                validCntCB=self.lightSourceFilter,
                                drawImg=drawImg,
                                drawInvalidPeaks=True)

        candidates = [LightSource(cnt, gray[pc[1]-offset[1], pc[0]-offset[0]]) for pc, cnt in zip(peakCenters, peakContours)]
        
        candidates = self._sortLightSources(candidates, nFeatures+additionalCandidates)

        self.iterations = iterations
        logging.debug("Local max iterations: {}".format(self.iterations))
        
        binaryImg, candidates = separateOverExposedCandidatesLocalPeak(candidates, 
                                                                       gray.shape, 
                                                                       self.kernel, # TODO: this is probably generally too small
                                                                       nFeatures,
                                                                       self.minArea,
                                                                       self.separationFraction,
                                                                       offset=offset, 
                                                                       drawImg=drawImg)

        self.img = binaryImg

        candidates = candidates[:nFeatures+additionalCandidates]

        if drawImg is not None:
            for i, ls in enumerate(candidates):
                drawInfo(drawImg, (ls.center[0]+15,ls.center[1]-15), str(i+1))

        return drawImg, candidates, roiCnt
    

class ModifiedHATSLocalPeak(AbstractLightSourceDetector):
    def __init__(self, toHATSScale, toPeakScale, modifiedHATSParams, localPeakParams):
        self.modifiedHATS = ModifiedHATS(**modifiedHATSParams)
        self.localPeak = LocalPeak(**localPeakParams)
        self.toHATSScale = toHATSScale  # change to HATS when min area > minArea*areaScale
        self.toPeakScale = toPeakScale  # change back to peak when min area < minArea*areaScale

        # Start with local peak
        self.currentDetector = self.localPeak

        self.img = None

    def __repr__(self):
        return str(self.currentDetector)

    def detect(self,
               gray, 
               nFeatures,
               additionalCandidates, 
               estDSPose=None,
               roiMargin=None, 
               drawImg=None):
        
        if estDSPose:
            if self.currentDetector == self.localPeak:
                changeToHATS = all([ls.area > self.toHATSScale*self.modifiedHATS.minArea for ls in estDSPose.associatedLightSources])
            else:
                changeToHATS = all([ls.area > self.toPeakScale*self.modifiedHATS.minArea for ls in estDSPose.associatedLightSources])
        
            if changeToHATS:
                self.currentDetector = self.modifiedHATS
            else:
                self.currentDetector = self.localPeak
        else:
            self.currentDetector = self.localPeak

        drawImg, candidates, roiCnt = self.currentDetector.detect(gray, 
                                                                  nFeatures=nFeatures,
                                                                  additionalCandidates=additionalCandidates, 
                                                                  estDSPose=estDSPose,
                                                                  roiMargin=roiMargin, 
                                                                  drawImg=drawImg)

        self.img = self.currentDetector.img

        return drawImg, candidates, roiCnt


def separateOverExposedCandidatesSimple(candidates, threshold, imgShape, offset=(0,0), drawImg=None):
    """
    Separates light source candidates that are potentially overlapping.
    1. Iterate through each candidate light source (only light sources with area > minAre are considered).
    2. Create a binary image using the contour from the current light source.
    3. Apply distance transform.
    4. Extract peaks from the image from the distance transform using simple thresholding. Each peak counts as a new candidate light source.
    Note: If only one peak is found, the original contour is kept. 
    If multiple peaks are found, the new contour is used for the new candidate (these are usuallt a lot smaller than the original peak).  
    """

    # Distance transform for overexposed light sources
    lightSources = []
    overlappingLightSources = []
    for ls in candidates:
        # First draw each contour
        binaryImg = np.zeros(imgShape, dtype=np.uint8)
        cv.drawContours(binaryImg, [ls.cnt], 0, 255, -1, offset=(-offset[0], -offset[1]))
        
        # TODO: for efficiency
        #x,y,w,h = cv.boundingRect(ls.cnt)
        #binaryImg = binaryImg[y:y+h, x:x+w] 
        
        # https://docs.opencv.org/3.4/d2/dbd/tutorial_distance_transform.html
        dist = cv.distanceTransform(binaryImg, cv.DIST_L2, 3)
        # Normalize the distance image for range = {0.0, 1.0}
        # so we can visualize and threshold it
        cv.normalize(dist, dist, 0, 255.0, cv.NORM_MINMAX)
        dist = dist.astype(np.uint8)
        
        # Otsu threshold is not suitable, generally too high threshold.
        _, threshImg = cv.threshold(dist, threshold, 255, cv.THRESH_BINARY)
        
        # FInd the new contours in the thresholded distance transform
        _, contours, _ = cv.findContours(threshImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE, offset=offset)
        
        if len(contours) == 1:
            # If we didn't detect any new light sources, we keep the original one
            lightSources.append(ls)
        else:
            for cnt in contours:
                overlappingLightSources.append(LightSource(cnt, ls.intensity))

    binaryImg = np.zeros(imgShape, dtype=np.uint8)
    if drawImg is not None:
        # only plot the overlapping light sources
        for ls in overlappingLightSources:
            cv.drawContours(drawImg, [ls.cnt], 0, (0,100,255), -1)
            cv.circle(drawImg, ls.center, 3, (0,255,255), -1)
            cv.drawContours(binaryImg, [ls.cnt], 0, 255, -1, offset=(-offset[0], -offset[1]))

    candidates = lightSources + overlappingLightSources

    return binaryImg, candidates


def separateOverExposedCandidatesLocalPeak(candidates, imgShape, kernel, n, minArea, p, offset=(0,0), drawImg=None):
    """
    Separates light source candidates that are potentially overlapping.
    1. Iterate through each candidate light source (only light sources with area > minArea are considered).
    2. Create a binary image using the contour from the current light source.
    3. Apply distance transform.
    4. Extract peaks from the image from the distance transform using a windowed version of local peak (slower than the simple version). 
       Each peak counts as a new candidate light source.

    Note: If only one peak is found, the original contour is kept. 
    If multiple peaks are found, the new contour is used for the new candidate (these are usuallt a lot smaller than the original peak).  
    """
    # Distance transform for overexposed light sources
    lightSources = []

    for ls in candidates:

        # We don't try to separate small light sources, keep them as they are
        if ls.area < minArea:
            lightSources.append(ls)
            continue
        
        # Draw the contour
        binaryImg = np.zeros(imgShape, dtype=np.uint8)
        cv.drawContours(binaryImg, [ls.cnt], 0, 255, -1, offset=(-offset[0], -offset[1]))
        
        # https://docs.opencv.org/3.4/d2/dbd/tutorial_distance_transform.html
        dist = cv.distanceTransform(binaryImg, cv.DIST_L2, 3)
        
        # Normalize the distance image
        cv.normalize(dist, dist, 0, 255.0, cv.NORM_MINMAX)
        dist = dist.astype(np.uint8)
        
        # Extract the peak contours
        # TODO: use localPeakFloodfill instead?
        (_, _, _, contours, _) = localPeakWindowed(dist, 
                                             kernel=kernel, 
                                             pMin=p,
                                             pMax=p, 
                                             n=n,
                                             offset=offset,
                                             windowRad=0,
                                             maxIter=n)

        if len(contours) == 1:
            # If we didn't detect any new countours, we keep the original light source
            lightSources.append(ls)
        else:
            for cnt in contours:
                # Overlapping
                lightSources.append(LightSource(cnt, ls.intensity, overlappingLightSource=ls))

            if drawImg is not None:
                # Draw the overlapping light source in red
                cv.drawContours(drawImg, [ls.cnt], 0, (0,0,255), -1)

    binaryImg = np.zeros(imgShape, dtype=np.uint8)
    if drawImg is not None:
        # only plot the overlapping light sources
        for ls in lightSources:
            cv.drawContours(binaryImg, [ls.cnt], 0, 255, -1, offset=(-offset[0], -offset[1]))
            if ls.isOverlapping():
                cv.drawContours(drawImg, [ls.cnt], 0, (255,0,0), -1)
                cv.circle(drawImg, ls.center, 3, (0,255,255), -1)

    return binaryImg, lightSources

def contourDistanceTransform(cnt):
    pass


if __name__ == "__main__":
    pass
