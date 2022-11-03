#!/usr/bin/env python
import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt
from scipy import signal

from lolo_perception.image_processing import LightSource, circularKernel, contourCentroid, contourRatio, drawInfo, findNPeaks2
from lolo_perception.perception_utils import plotHistogram, regionOfInterest


class ModifiedHATS:
    MODE_SIMPLE = "simple"
    MODE_PEAK = "peak"
    MODE_VALLEY = "valley"
    
    def __init__(self, 
                 nFeatures, 
                 peakMargin=0, 
                 minArea=1, 
                 minRatio=0, 
                 maxIntensityChange=0.7, 
                 blurKernelSize=5, 
                 mode="valley", 
                 ignorePeakAtMax=False, 
                 showHistogram=False):
        """
        try cv.THRESH_OTSU?
        """
        self.nFeatures = nFeatures
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

    def _calcCandidates(self, contours):
        candidates = []
        removedCandidates = []
        for cnt in contours:
            #if cv.contourArea(cnt)+1 > self.minArea and contourRatio(cnt) > self.minRatio:
            if cv.contourArea(cnt) > self.minArea and contourRatio(cnt) > self.minRatio:
                candidates.append(cnt)
            else:
                removedCandidates.append(cnt)

        return candidates, removedCandidates

    def _sortLightSources(self, candidates, maxAdditionalCandidates):
        candidates.sort(key=lambda p: p.area, reverse=True)
        candidates = candidates[:self.nFeatures+maxAdditionalCandidates]
        
        return candidates

    def process(self, gray, maxAdditionalCandidates=0, estDSPose=None, roiMargin=None, drawImg=None):

        grayFull = gray.copy()

        roiCnt = None
        offset = (0, 0)
        if estDSPose:
            featurePointsGuess = estDSPose.reProject()
            roiMargin += max([ls.radius for ls in estDSPose.associatedLightSources])
            (x, y, w, h), roiCnt = regionOfInterest(featurePointsGuess, wMargin=roiMargin, hMargin=roiMargin)
            x = max(0, x)
            x = min(gray.shape[1]-1, x)
            y = max(0, y)
            y = min(gray.shape[0]-1, y)

            offset = (x, y)
            gray = gray[y:y+h, x:x+w]

            minIntensity = min([ls.intensity for ls in estDSPose.associatedLightSources])
            minIntensity = self.maxIntensityChange*minIntensity

            # local max that are below 70% of the mean intensity of the previous light sources are discarded
            _, gray = cv.threshold(gray, minIntensity, 256, cv.THRESH_TOZERO)

        upper = 255

        grayROI = gray.copy()
        img = gray.copy()
        img = cv.GaussianBlur(img, (self.blurKernelSize,self.blurKernelSize), self.sigma)

        # TODO: OPEING onland, CLOSING underwater
        #img = cv.morphologyEx(img, cv.MORPH_OPEN, self.morphKernel, iterations=1)
        if self.morphKernel is not None:
            img = cv.morphologyEx(img, cv.MORPH_CLOSE, self.morphKernel, iterations=1)

        #localMaxImg = localMax(img) # only use peaks defined by local maximas
        hist = cv.calcHist([img], [0], None, [256], [0,256])
        hist = hist.ravel()

        if self.mode == self.MODE_SIMPLE:
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
                #if np.max(img) == 255:
                #    self.threshold = 254
                #else:
                self.threshold = histPeaks.pop()-1
            else:
                self.threshold = np.max(img)-1

        ret, imgTemp = cv.threshold(img, self.threshold, upper, self.thresholdType)
        _, contours, hier = cv.findContours(imgTemp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        candidates, removedCandidates = self._calcCandidates(contours)
        
        img = img.copy()
        i = 0
        while True:
            i += 1

            if len(candidates) >= self.nFeatures:
                break

            if len(histPeaks) > 0:
                self.threshold = histPeaks.pop()-1
            else:
                break

            ret, imgTemp = cv.threshold(img, self.threshold, upper, self.thresholdType)

            _, contours, hier = cv.findContours(imgTemp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            candidates, removedCandidates = self._calcCandidates(contours)

        if self.peakMargin > 0 and len(histPeaks) > 0:
            self.threshold = histPeaks.pop()-1
            ret, imgTemp = cv.threshold(img, self.threshold, upper, self.thresholdType)
            _, contours, hier = cv.findContours(imgTemp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            candidates, removedCandidates = self._calcCandidates(contours)

        print("HATS iterations: {}".format(i))
        print("Threshold: {}".format(self.threshold))
        self.iterations = i
        self.img = imgTemp

        #candidates = [cv.approxPolyDP(cnt, 0.01*cv.arcLength(cnt, True), True) for cnt in candidates]
        candidates = [LightSource(cnt+offset, self.threshold) for cnt in candidates]
        candidates = self._sortLightSources(candidates, maxAdditionalCandidates)


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

        ########### plot peaks ##############
        if self.showHistogram:
            plt.cla()
            N = 150
            grayROI = cv.GaussianBlur(grayROI, (self.blurKernelSize,self.blurKernelSize), self.sigma)
            grayFull = cv.GaussianBlur(grayFull, (self.blurKernelSize,self.blurKernelSize), self.sigma)
            plotHistogram(grayFull, N=N, highlightPeak=self.threshold, facecolor="b")
            plotHistogram(grayROI, N=N, highlightPeak=self.threshold, facecolor="r", limitAxes=False)
            plt.pause(0.0001)
            
        ########### plot peaks ##############

        return self.img, candidates, roiCnt


class LocalPeak:
    def __init__(self, nFeatures, kernelSize, pMin, pMax, maxIntensityChange, minArea, minCircleExtent, blurKernelSize, ignorePAtMax, maxIter):
        """
        try cv.THRESH_OTSU?
        """
        self.nFeatures = nFeatures
        self.kernelSize = kernelSize
        self.kernel = np.ones((self.kernelSize,self.kernelSize)) # circularKernel(self.kernelSize) # Insanely much faster with square!
        self.pMin = pMin
        self.pMax = pMax
        self.img = np.zeros((10, 10), dtype=np.uint8) # store the last processed image
        self.maxIntensityChange = maxIntensityChange

        self.minArea = minArea
        self.minCircleExtent = minCircleExtent
        self.filterAfter = True # TODO: change to false

        self.blurKernelSize = blurKernelSize
        self.sigma = 0.3*((self.blurKernelSize-1)*0.5 - 1) + 0.8

        self.morphKernel = None
        if blurKernelSize > 0:
            self.morphKernel = circularKernel(blurKernelSize)

        self.ignorePAtMax = ignorePAtMax

        self.maxIter = maxIter
        # keep track of number of peak iterations
        self.iterations = 0

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    def validContour(self, cnt):
        if cv.contourArea(cnt) > self.minArea and contourRatio(cnt) < self.minCircleExtent:
            return False
        return True

    def _sortLightSources(self, candidates, maxAdditionalCandidates):
        candidates.sort(key=lambda p: (p.intensity, p.area), reverse=True)
        candidates = candidates[:self.nFeatures+maxAdditionalCandidates]
        
        return candidates

    def process(self, gray, maxAdditionalCandidates=0, estDSPose=None, roiMargin=None, drawImg=None):

        grayOrig = gray.copy()

        offset = (0, 0)
        roiCnt = None
        minLightSourceRadius = 0
        peakMargin = 3
        minIntensity = 0
        if estDSPose:
            featurePointsGuess = estDSPose.reProject()
            roiMargin += max([ls.radius for ls in estDSPose.associatedLightSources])
            (x, y, w, h), roiCnt = regionOfInterest(featurePointsGuess, wMargin=roiMargin, hMargin=roiMargin)

            # TODO: this doesn't really do what it is intendent to do, but it's alright
            x = max(0, x)
            x = min(gray.shape[1]-1, x)
            y = max(0, y)
            y = min(gray.shape[0]-1, y)

            offset = (x, y)
            gray = gray[y:y+h, x:x+w]

            minIntensity = min([ls.intensity for ls in estDSPose.associatedLightSources])
            minIntensity = self.maxIntensityChange*minIntensity

            # TODO: When is this reliable? Should not be used in findPeaks2 (centroid should be used) 
            #peakMargin = max([ls.radius for ls in estDSPose.associatedLightSources])
            peakMargin = 0
        
        # blurring seems to help for large resolution 
        # test_sessions/171121_straight_test.MP4
        if self.blurKernelSize > 0:
            gray = cv.GaussianBlur(gray, (self.blurKernelSize,self.blurKernelSize), self.sigma)

            #gray = cv.blur(gray, (self.blurKernelSize,self.blurKernelSize))
            # TODO: This is good for overexposed light sources, reducing the need for blurring
            #gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, self.morphKernel, iterations=1)

        # TODO: works pretty good in ROI
        #ret, otsu = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        #minIntensity = max(ret, minIntensity)

        #peakMargin = 0 # TODO: should work with 0 but needs proper testing
        (peakDilationImg, 
        peaksDilationMasked, 
        peakCenters, 
        peakContours,
        iterations) = findNPeaks2(gray, 
                                  kernel=self.kernel, 
                                  pMin=self.pMin,
                                  pMax=self.pMax, 
                                  n=self.nFeatures+maxAdditionalCandidates,
                                  minThresh=minIntensity,
                                  # maybe this should be (self.kernelSize-1)/2 instead of peakMargin?
                                  # or maybe it will remove the true candidates in case of weird shapes from "non-peaks"?
                                  margin=peakMargin,
                                  ignorePAtMax=self.ignorePAtMax,
                                  offset=offset,
                                  maxIter=self.maxIter,
                                  validCntCB=self.validContour if not self.filterAfter else None,
                                  drawImg=drawImg,
                                  drawInvalidPeaks=True)

        # TODO: check if offset is correct
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
            
        candidates = self._sortLightSources(candidates, maxAdditionalCandidates)
        if drawImg is not None:
            for i, ls in enumerate(candidates):
                #cv.circle(drawImg, ls.center, int(round(ls.radius)), (255,0,255), 2)
                cv.circle(drawImg, ls.center, 2, (255,0,0), 2)
            for ls in removedCandidates:
                cv.drawContours(drawImg, [ls.cnt], 0, (0,0,255), -1)
        
        """
        peakDilationImgDraw = cv.cvtColor(peakDilationImg, cv.COLOR_GRAY2BGR)
        if drawImg is not None:
            for i, ls in enumerate(candidates):
                #cv.drawContours(drawImg, [ls.cnt], 0, (255,0,0), -1)

                color = (0, float(255)/(i+11), float(255)/(i+11))
                color = (0,255, 255)
                cv.drawContours(peakDilationImgDraw, [ls.cnt - offset], 0, color, 1)
                drawInfo(peakDilationImgDraw, (ls.center[0]-offset[0]+15, ls.center[1]-offset[1]-15), str(i+1), color=color)
                #drawInfo(drawImg, (ls.center[0]+15, ls.center[1]-15), str(i+1), color=(255,0,0))
                #cv.circle(drawImg, ls.center, int(round(ls.radius)), (255,0,255), 2)

            #for ls in removedCandidates:
            #    drawInfo(drawImg, (ls.center[0]+15, ls.center[1]-15), str(ls.area), color=(0,0,255))
            #    cv.drawContours(drawImg, [ls.cnt], 0, (0,0,255), -1)
        """
        
        self.img = peakDilationImg
        self.iterations = iterations
        print("Local max iterations:", self.iterations)
        return drawImg, candidates, roiCnt


        grayOrig = gray.copy()

        offset = (0, 0)
        roiCnt = None
        minLightSourceRadius = 0
        peakMargin = 3
        minIntensity = 0
        if estDSPose:
            featurePointsGuess = estDSPose.reProject()
            roiMargin += max([ls.radius for ls in estDSPose.associatedLightSources])
            (x, y, w, h), roiCnt = regionOfInterest(featurePointsGuess, wMargin=roiMargin, hMargin=roiMargin)

            # TODO: this doesn't really do what it is intendent to do, but it's alright
            x = max(0, x)
            x = min(gray.shape[1]-1, x)
            y = max(0, y)
            y = min(gray.shape[0]-1, y)

            offset = (x, y)
            gray = gray[y:y+h, x:x+w]

            minIntensity = min([ls.intensity for ls in estDSPose.associatedLightSources])
            minIntensity = self.maxIntensityChange*minIntensity

            # TODO: When is this reliable? Should not be used in findPeaks2 (centroid should be used) 
            #peakMargin = max([ls.radius for ls in estDSPose.associatedLightSources])
            peakMargin = 0
        
        # blurring seems to help for large resolution 
        # test_sessions/171121_straight_test.MP4
        if self.blurKernelSize > 0:
            gray = cv.GaussianBlur(gray, (self.blurKernelSize,self.blurKernelSize), self.sigma)

            #gray = cv.blur(gray, (self.blurKernelSize,self.blurKernelSize))
            # TODO: This is good for overexposed light sources, reducing the need for blurring
            #gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, self.morphKernel, iterations=1)

        # TODO: works pretty good in ROI
        #ret, otsu = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        #minIntensity = max(ret, minIntensity)

        #peakMargin = 0 # TODO: should work with 0 but needs proper testing
        (peakDilationImg, 
        peaksDilationMasked, 
        peakCenters, 
        peakContours,
        iterations) = findNPeaks2(gray, 
                                  kernel=self.kernel, 
                                  pMin=self.pMin,
                                  pMax=self.pMax, 
                                  n=self.nFeatures+maxAdditionalCandidates,
                                  minThresh=minIntensity,
                                  # maybe this should be (self.kernelSize-1)/2 instead of peakMargin?
                                  # or maybe it will remove the true candidates in case of weird shapes from "non-peaks"?
                                  margin=peakMargin,
                                  ignorePAtMax=self.ignorePAtMax,
                                  offset=offset,
                                  maxIter=self.maxIter,
                                  validCntCB=self.validContour if not self.filterAfter else None,
                                  drawImg=drawImg,
                                  drawInvalidPeaks=True)

        # TODO: check if offset is correct
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
            
        candidates = self._sortLightSources(candidates, maxAdditionalCandidates)
        if drawImg is not None:
            for i, ls in enumerate(candidates):
                #cv.circle(drawImg, ls.center, int(round(ls.radius)), (255,0,255), 2)
                cv.circle(drawImg, ls.center, 2, (255,0,0), 2)
            for ls in removedCandidates:
                cv.drawContours(drawImg, [ls.cnt], 0, (0,0,255), -1)
        
        """
        peakDilationImgDraw = cv.cvtColor(peakDilationImg, cv.COLOR_GRAY2BGR)
        if drawImg is not None:
            for i, ls in enumerate(candidates):
                #cv.drawContours(drawImg, [ls.cnt], 0, (255,0,0), -1)

                color = (0, float(255)/(i+11), float(255)/(i+11))
                color = (0,255, 255)
                cv.drawContours(peakDilationImgDraw, [ls.cnt - offset], 0, color, 1)
                drawInfo(peakDilationImgDraw, (ls.center[0]-offset[0]+15, ls.center[1]-offset[1]-15), str(i+1), color=color)
                #drawInfo(drawImg, (ls.center[0]+15, ls.center[1]-15), str(i+1), color=(255,0,0))
                #cv.circle(drawImg, ls.center, int(round(ls.radius)), (255,0,255), 2)

            #for ls in removedCandidates:
            #    drawInfo(drawImg, (ls.center[0]+15, ls.center[1]-15), str(ls.area), color=(0,0,255))
            #    cv.drawContours(drawImg, [ls.cnt], 0, (0,0,255), -1)
        """
        
        self.img = peakDilationImg
        self.iterations = iterations
        print("Local max iterations:", self.iterations)
        return drawImg, candidates, roiCnt

class HATSandLocalPeakDetector:
    pass

if __name__ == "__main__":
    pass
