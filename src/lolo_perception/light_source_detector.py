#!/usr/bin/env python
import cv2 as cv
import numpy as np
import yaml # https://circleci.com/blog/what-is-yaml-a-beginner-s-guide/?utm_source=google&utm_medium=sem&utm_campaign=sem-google-dg--emea-en-dsa-maxConv-auth-nb&utm_term=g_-_c__dsa_&utm_content=&gclid=Cj0KCQiAj4ecBhD3ARIsAM4Q_jHZ-zN1bx3ZjWMeJlzAwDGioniBoU-hO9sJl-ytnKDGYmJOhj4nCzgaAkdVEALw_wcB
import lolo_perception.py_logging as logging

from matplotlib import pyplot as plt
from scipy import signal

from lolo_perception.image_processing import LightSource, circularKernel, contourCentroid, contourRatio, drawInfo, findNPeaks2
from lolo_perception.perception_utils import plotHistogram, regionOfInterest

# def fromYaml(classToInstatiate, yamlPath):
#     with open(yamlPath, "r") as file:
#         d = yaml.safe_load(file)

#     return fromDict(classToInstatiate, d)

# def fromDict(classToInstatiate, d):
#     d[classToInstatiate.__name__]
#     return classToInstatiate(**d)

class ModifiedHATS:
    MODE_SIMPLE = "simple"
    MODE_PEAK = "peak"
    MODE_VALLEY = "valley"
    
    def __init__(self,  
                 peakMargin=0, 
                 minArea=1, 
                 minRatio=0, 
                 maxIntensityChange=0.7, 
                 blurKernelSize=5, 
                 mode="valley", 
                 ignorePeakAtMax=False, 
                 showHistogram=False):

        self.threshold = 256
        self.peakMargin = peakMargin # 0-1
        self.minArea = minArea
        self.minRatio = minRatio
        self.maxIntensityChange = maxIntensityChange
        self.blurKernelSize = blurKernelSize

        # https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
        self.sigma = 0.3*((self.blurKernelSize-1)*0.5 - 1) + 0.8 # TODO: does this work properly?
        self.thresholdType = cv.THRESH_BINARY

        assert mode in (self.MODE_SIMPLE, self.MODE_PEAK, self.MODE_VALLEY), "Invalid mode '', must be '{}, {}' or '{}'".format(mode, self.MODE_SIMPLE, self.MODE_PEAK, self.MODE_VALLEY)
        self.mode = mode
        self.ignorePeakAtMax = ignorePeakAtMax
        self.showHistogram = showHistogram

        self.morphKernel = None
        if blurKernelSize > 0:
            self.morphKernel = circularKernel(blurKernelSize)

        self.img = np.zeros((10, 10)) # store the last processed image
        self.iterations = 0

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    def __repr__(self):
        return "HATS"

    @staticmethod
    def fromYaml(yamlPath):
        with open(yamlPath, "r") as file:
            p = yaml.safe_load(file)

        p = p["modified_hats"]

        return ModifiedHATS(p["peak_margin"], 
                            p["min_area"], 
                            p["min_circle_extent"], 
                            p["max_intensity_change"], 
                            p["blur_kernel_size"], 
                            p["mode"], 
                            p["ignore_at_max"])

    def _calcCandidates(self, contours):
        candidates = []
        removedCandidates = []
        for cnt in contours:
            if cv.contourArea(cnt) > self.minArea and contourRatio(cnt) > self.minRatio:
                candidates.append(cnt)
            else:
                removedCandidates.append(cnt)

        return candidates, removedCandidates

    def _sortLightSources(self, candidates, nCandidates):
        candidates.sort(key=lambda p: p.area, reverse=True)
        candidates = candidates[:nCandidates]
        
        return candidates

    def process(self, gray, nFeatures, additionalCandidates=0, estDSPose=None, roiMargin=None, drawImg=None):

        grayFull = gray.copy()

        roiCnt = None
        offset = (0, 0)
        if estDSPose:
            featurePointsGuess = estDSPose.reProject()
            roiMargin += max([ls.radius for ls in estDSPose.associatedLightSources])
            (x, y, w, h), roiCnt = regionOfInterest(featurePointsGuess, wMargin=roiMargin, hMargin=roiMargin, shape=gray.shape)

            offset = (x, y)
            gray = gray[y:y+h, x:x+w]

            minIntensity = min([ls.intensity for ls in estDSPose.associatedLightSources])
            minIntensity = self.maxIntensityChange*minIntensity

            _, gray = cv.threshold(gray, minIntensity, 256, cv.THRESH_TOZERO)

        grayROI = gray.copy()
        img = gray.copy()
        img = cv.GaussianBlur(img, (self.blurKernelSize,self.blurKernelSize), self.sigma)

        # TODO: OPEING onland, CLOSING underwater
        if self.morphKernel is not None:
            #img = cv.morphologyEx(img, cv.MORPH_OPEN, self.morphKernel, iterations=1)
            img = cv.morphologyEx(img, cv.MORPH_CLOSE, self.morphKernel, iterations=1)

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
        
        img = img.copy()
        i = 0
        while True:
            i += 1

            if len(candidates) >= nFeatures:
                break

            if len(histPeaks) > 0:
                self.threshold = histPeaks.pop()-1
            else:
                break

            ret, imgTemp = cv.threshold(img, self.threshold, 255, self.thresholdType)

            _, contours, hier = cv.findContours(imgTemp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            candidates, removedCandidates = self._calcCandidates(contours)

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

        ########### plot histogram peaks ##############
        if self.showHistogram:
            plt.cla()
            N = 150
            grayROI = cv.GaussianBlur(grayROI, (self.blurKernelSize,self.blurKernelSize), self.sigma)
            grayFull = cv.GaussianBlur(grayFull, (self.blurKernelSize,self.blurKernelSize), self.sigma)
            plotHistogram(grayFull, N=N, highlightPeak=self.threshold, facecolor="b")
            plotHistogram(grayROI, N=N, highlightPeak=self.threshold, facecolor="r", limitAxes=False)
            plt.pause(0.0001)
            
        ########### \plot histogram peaks ##############

        return self.img, candidates, roiCnt


class LocalPeak:
    def __init__(self, kernelSize, pMin, pMax, maxIntensityChange, minArea, minCircleExtent, blurKernelSize, ignorePAtMax, maxIter, filterAfter):
        """
        try cv.THRESH_OTSU?
        """
        self.kernelSize = kernelSize
        self.kernel = np.ones((self.kernelSize,self.kernelSize)) # circularKernel(self.kernelSize) # Insanely much faster with square!
        self.pMin = pMin
        self.pMax = pMax
        self.img = np.zeros((10, 10), dtype=np.uint8) # store the last processed image
        self.maxIntensityChange = maxIntensityChange

        self.minArea = minArea
        self.minCircleExtent = minCircleExtent
        self.filterAfter = filterAfter

        self.blurKernelSize = blurKernelSize
        self.sigma = 0.3*((self.blurKernelSize-1)*0.5 - 1) + 0.8

        self.morphKernel = None
        if blurKernelSize > 0:
            self.morphKernel = circularKernel(blurKernelSize)

        self.ignorePAtMax = ignorePAtMax

        self.maxIter = maxIter
        self.iterations = 0

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    def __repr__(self):
        return "Local Peak"

    @staticmethod
    def fromYaml(yamlPath, namespace=None):
        with open(yamlPath, "r") as file:
            p = yaml.safe_load(file)

        p = p["local_peak"]

        return LocalPeak(p["kernel_size"], 
                         p["p_min"], 
                         p["p_max"], 
                         p["max_intensity_change"], 
                         p["min_area"], 
                         p["min_circle_extent"], 
                         p["blur_kernel_size"], 
                         p["ignore_at_max"], 
                         p["max_iter"], 
                         p["filter_after"])

    def validContour(self, cnt):
        if cv.contourArea(cnt) > self.minArea and contourRatio(cnt) < self.minCircleExtent:
            return False
        return True

    def _sortLightSources(self, candidates, nCandidates):
        candidates.sort(key=lambda p: (p.intensity, p.area), reverse=True)
        candidates = candidates[:nCandidates]
        
        return candidates

    def process(self, gray, nFeatures, additionalCandidates=0, estDSPose=None, roiMargin=None, drawImg=None):

        grayOrig = gray.copy()

        offset = (0, 0)
        roiCnt = None
        minLightSourceRadius = 0
        peakMargin = 3
        minIntensity = 0
        if estDSPose:
            featurePointsGuess = estDSPose.reProject()
            roiMargin += max([ls.radius for ls in estDSPose.associatedLightSources])
            (x, y, w, h), roiCnt = regionOfInterest(featurePointsGuess, wMargin=roiMargin, hMargin=roiMargin, shape=gray.shape)

            offset = (x, y)
            gray = gray[y:y+h, x:x+w]

            minIntensity = min([ls.intensity for ls in estDSPose.associatedLightSources])
            minIntensity = self.maxIntensityChange*minIntensity

            # TODO: When is this reliable? Should not be used in findPeaks2 (centroid should be used) 
            #peakMargin = max([ls.radius for ls in estDSPose.associatedLightSources])
            peakMargin = 0
        
        if self.blurKernelSize > 0:
            gray = cv.GaussianBlur(gray, (self.blurKernelSize,self.blurKernelSize), self.sigma)

        #peakMargin = 0 # TODO: should work with 0 but needs proper testing
        (peakDilationImg, 
        peaksDilationMasked, 
        peakCenters, 
        peakContours,
        iterations) = findNPeaks2(gray, 
                                  kernel=self.kernel, 
                                  pMin=self.pMin,
                                  pMax=self.pMax, 
                                  n=nFeatures+additionalCandidates,
                                  minThresh=minIntensity,
                                  margin=peakMargin,
                                  ignorePAtMax=self.ignorePAtMax,
                                  offset=offset,
                                  maxIter=self.maxIter,
                                  validCntCB=self.validContour if not self.filterAfter else None,
                                  drawImg=drawImg,
                                  drawInvalidPeaks=True)

        candidates = [LightSource(cnt, gray[pc[1]-offset[1], pc[0]-offset[0]]) for pc, cnt in zip(peakCenters, peakContours)]

        # TODO: This should probably be done
        candidatesNew = []
        removedCandidates = []
        if self.filterAfter:
            for ls in candidates:
                # check convexity defects
                # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gada4437098113fd8683c932e0567f47ba
                found = True

                if ls.area > self.minArea and ls.circleExtent() < self.minCircleExtent:
                    found = False

                if found:
                    candidatesNew.append(ls)
                else:
                    removedCandidates.append(ls)
        else:
            candidatesNew = candidates
        candidates = candidatesNew
            
        candidates = self._sortLightSources(candidates, nFeatures+additionalCandidates)
        
        if drawImg is not None:
            for ls in candidates:
                cv.circle(drawImg, ls.center, 2, (255,0,0), 2)
            for ls in removedCandidates:
                cv.drawContours(drawImg, [ls.cnt], 0, (0,0,255), -1)
        
        self.img = peakDilationImg
        self.iterations = iterations
        logging.debug("Local max iterations: {}".format(self.iterations))

        return drawImg, candidates, roiCnt

class ModifiedHATSLocalPeakDetector:
    def __init__(self, modifiedHATS, localPeak, toHATSScale, toPeakScale):
        self.modifiedHATS = modifiedHATS
        self.localPeak = localPeak
        self.toHATSScale = toHATSScale  # change to HATS when min area > minArea*areaScale
        self.toPeakScale = toPeakScale  # change back to peak when min area < minArea*areaScale

        # Start with local peak
        self.currentDetector = self.localPeak

        self.img = None

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    def __repr__(self):
        return str(self.currentDetector)

    def process(self,
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

        ret = self.currentDetector(gray, 
                                   nFeatures=nFeatures,
                                   additionalCandidates=additionalCandidates, 
                                   estDSPose=estDSPose,
                                   roiMargin=roiMargin, 
                                   drawImg=drawImg)

        self.img = self.currentDetector.img

        return ret

    @staticmethod
    def fromYaml(yamlPath):

        modifiedHATS = ModifiedHATS.fromYaml(yamlPath)
        localPeak = LocalPeak.fromYaml(yamlPath)

        with open(yamlPath, "r") as file:
            p = yaml.safe_load(file)

        p = p["modified_hats_local_peak"]

        return ModifiedHATSLocalPeakDetector(modifiedHATS, 
                                             localPeak,
                                             toHATSScale=p["to_hats_scale"],
                                             toPeakScale=p["to_peak_scale"])

if __name__ == "__main__":
    pass
