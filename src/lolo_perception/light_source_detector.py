#!/usr/bin/env python
import sys
import inspect

import cv2 as cv
import numpy as np
import yaml # https://circleci.com/blog/what-is-yaml-a-beginner-s-guide/?utm_source=google&utm_medium=sem&utm_campaign=sem-google-dg--emea-en-dsa-maxConv-auth-nb&utm_term=g_-_c__dsa_&utm_content=&gclid=Cj0KCQiAj4ecBhD3ARIsAM4Q_jHZ-zN1bx3ZjWMeJlzAwDGioniBoU-hO9sJl-ytnKDGYmJOhj4nCzgaAkdVEALw_wcB
import lolo_perception.py_logging as logging

from matplotlib import pyplot as plt
from scipy import signal

from lolo_perception.image_processing import LightSource, circularKernel, contourCentroid, contourRatio, drawInfo, findNPeaks2, findNPeaks3, localMax, localMaxSupressed, localMaxDiff, localMaxDiff2
from lolo_perception.perception_utils import plotHistogram, regionOfInterest

def detectorFromYaml(yamlPath):
    with open(yamlPath, "r") as file:
        d = yaml.safe_load(file)

    # The parameters should be under the key "light_source_detector" in the yaml file
    d = d["light_source_detector"]

    # The name of the detector class
    className = d["class_name"]

    # Get the class from the name
    detector = getLightSourceDetectorFromName(className)

    # Remove the class_name key from the dictionary
    del d["class_name"]

    # Return an instance of the class. The rest of the key-value pairs of d
    # is assumed to be the correct parameters to the constructor.
    return detector(**d)

def printLightSourceDetectors():
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if obj != AbstractLightSourceDetector and issubclass(obj, AbstractLightSourceDetector):
            print(obj, name)

def getLightSourceDetectorFromName(className):
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
            self.morphKernel = circularKernel(blurKernelSize*2+1)

        self.img = np.zeros((10, 10)) # store the last processed image
        self.iterations = 0

    def __repr__(self):
        return "HATS"

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

    def detect(self, gray, nFeatures, additionalCandidates=0, estDSPose=None, roiMargin=None, drawImg=None):

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

        # TODO: OPENING onland, CLOSING underwater
        if self.morphKernel is not None:
            pass
            #img = cv.morphologyEx(img, cv.MORPH_OPEN, self.morphKernel, iterations=1)
            #img = cv.morphologyEx(img, cv.MORPH_CLOSE, self.morphKernel, iterations=1)

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
        while True:
            i += 1

            #if len(candidates) >= nFeatures:
            #    break

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
            elif len(newCandidates) > prevNCandidates:
                candidates = newCandidates
                prevNCandidates = len(candidates)
            else:
                # If its equal, we keep the candidates from before to keep the contours as small as possible
                pass

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

        self.img, candidates = separateOverExposedCandidates(candidates, 
                                                             gray.shape, 
                                                             offset=offset, 
                                                             drawImg=drawImg)

        candidates = candidates[:nFeatures+additionalCandidates]

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


class LocalPeak(AbstractLightSourceDetector):
    def __init__(self, kernelSize, pMin, pMax, maxIntensityChange, minArea, minCircleExtent, blurKernelSize, windowRad, ignorePAtMax, maxIter, filterAfter):
        """
        try cv.THRESH_OTSU?
        """
        self.kernelSize = kernelSize
        self.kernel = np.ones((self.kernelSize,self.kernelSize)) # circularKernel(self.kernelSize) # Insanely much faster with square!
        self.pMin = pMin
        self.pMax = pMax
        self.img = np.zeros((10, 10), dtype=np.uint8) # store the last processed image
        self.maxIntensityChange = maxIntensityChange

        self.minAreaDefault = 0# TODO remove
        self.minArea = 0
        self.minCircleExtent = minCircleExtent
        self.filterAfter = filterAfter

        self.blurKernelSize = blurKernelSize
        self.sigma = 0.3*((self.blurKernelSize-1)*0.5 - 1) + 0.8

        self.morphKernel = None
        if blurKernelSize > 0:
            self.morphKernel = circularKernel(blurKernelSize)

        self.ignorePAtMax = ignorePAtMax

        self.windowRad = windowRad
        self.maxIter = maxIter
        self.iterations = 0

    def __repr__(self):
        return "Local Peak"


    def validContour(self, cnt):
        if cv.contourArea(cnt) > self.minArea and contourRatio(cnt) < self.minCircleExtent:
            return False
        return True
    

    def _sortLightSources(self, candidates, nCandidates):
        candidates.sort(key=lambda p: (p.intensity, p.area), reverse=True)
        candidates = candidates[:nCandidates]
        
        return candidates


    def detect(self, gray, nFeatures, additionalCandidates=0, estDSPose=None, roiMargin=None, drawImg=None):

        grayOrig = gray.copy()

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

            # TODO: When is this reliable? Should not be used in findPeaks2 (centroid should be used) 
            #peakMargin = max([ls.radius for ls in estDSPose.associatedLightSources])
            peakMargin = 0
        
        if self.blurKernelSize > 0:
            gray = cv.GaussianBlur(gray, (self.blurKernelSize,self.blurKernelSize), self.sigma)

        (peakDilationImg, 
        peaksDilationMasked, 
        peakCenters, 
        peakContours,
        iterations) = findNPeaks3(gray, 
                                  kernel=self.kernel, 
                                  pMin=self.pMin,
                                  pMax=self.pMax, 
                                  n=nFeatures+additionalCandidates,
                                  windowRad=self.windowRad,
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

        self.iterations = iterations
        logging.debug("Local max iterations: {}".format(self.iterations))

        # self.img, candidates = separateOverExposedCandidates(candidates, 
        #                                                      gray.shape, 
        #                                                      offset=offset, 
        #                                                      drawImg=drawImg)

        self.img, candidates = separateOverExposedCandidates2(candidates, 
                                                             gray.shape, 
                                                             self.kernel,
                                                             offset=offset, 
                                                             drawImg=drawImg)

        candidates = candidates[:nFeatures+additionalCandidates]

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

        # # TODO: Distance transform for overexposed light sources
        # offset = (0, 0)
        # if roiCnt is not None:
        #     offset = tuple(roiCnt[0])
        # distTransCandidates = []
        # for ls in candidates:
        #     # First draw each contour
        #     binaryImg = np.zeros(gray.shape, dtype=np.uint8)
        #     cv.drawContours(binaryImg, [ls.cnt], 0, 255, -1, offset=(-offset[0], -offset[1]))
            
        #     # TODO: for efficiency
        #     #x,y,w,h = cv.boundingRect(ls.cnt)
        #     #binaryImg = binaryImg[y:y+h, x:x+w] 
            
        #     # https://docs.opencv.org/3.4/d2/dbd/tutorial_distance_transform.html
        #     dist = cv.distanceTransform(binaryImg, cv.DIST_L2, 3)
        #     # Normalize the distance image for range = {0.0, 1.0}
        #     # so we can visualize and threshold it
        #     cv.normalize(dist, dist, 0, 255.0, cv.NORM_MINMAX)
        #     dist = dist.astype(np.uint8)
            
        #     # TODO: how does otsu perform on the dist image?
        #     # Edit: not that well...
        #     _, threshImg = cv.threshold(dist, 150, 255, cv.THRESH_BINARY)
        #     #_, threshImg = cv.threshold(dist, 0, 255, cv.THRESH_OTSU)
            
        #     _, contours, hier = cv.findContours(threshImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE, offset=offset)
        #     for cnt in contours:
        #         distTransCandidates.append(LightSource(cnt, ls.intensity))

        # candidates = distTransCandidates[:nFeatures+additionalCandidates]

        # binaryImg = np.zeros(gray.shape, dtype=np.uint8)
        # if drawImg is not None:
        #     for ls in candidates:
        #         cv.drawContours(drawImg, [ls.cnt], 0, (255,0,255), -1)
        #         cv.circle(drawImg, ls.center, 3, (0,255,255), -1)
        #         #cv.drawContours(binaryImg, [ls.cnt], 0, 255, -1, offset=(-offset[0], -offset[1]))

        # #self.img = binaryImg

        return drawImg, candidates, roiCnt


class LogPyramid(AbstractLightSourceDetector):
    def __init__(self):
        pass

    def __repr__(self):
        return "Pyramid"

    def detect(self,
               gray, 
               nFeatures,
               additionalCandidates, 
               estDSPose=None,
               roiMargin=None, 
               drawImg=None):
                
        roiCnt = None
        offset = (0, 0)
        if estDSPose:
            featurePointsGuess = estDSPose.reProject()
            roiMargin += max([ls.radius for ls in estDSPose.associatedLightSources])
            (x, y, w, h), roiCnt = regionOfInterest(featurePointsGuess, wMargin=roiMargin, hMargin=roiMargin, shape=gray.shape)

            offset = (x, y)
            gray = gray[y:y+h, x:x+w]

            #minIntensity = min([ls.intensity for ls in estDSPose.associatedLightSources])
            #minIntensity = self.maxIntensityChange*minIntensity

        blurred = cv.GaussianBlur(gray, (11,11), 0)
        #cv.imshow("thresholded", thresh)

        # Laplacian pyramid
        blobRadius = 3
        sigma = (blobRadius-1.0)/3.0
        ksize = blobRadius*2-1#int(round(sigma*3))
        if ksize % 2 == 0:
            ksize += 1

        blurred = cv.GaussianBlur(gray, (ksize,ksize), sigma) # ksize

        morphKernel = np.ones((3,3))
        pyramid = []
        pyrImg = blurred
        for i in range(4):
            logging.info("first")
            dst = cv.Laplacian(pyrImg, ddepth=cv.CV_64F, ksize=5)
            dst = cv.convertScaleAbs(dst, alpha=255./dst.max())
            dst = dst.astype(np.uint8)

            pyramid.append(cv.resize(dst, (blurred.shape[1], blurred.shape[0])))

            #pyrImg = cv.morphologyEx(pyrImg, cv.MORPH_ERODE, morphKernel)
            pyrImg = cv.pyrDown(pyrImg)
            

        #for i, img in enumerate(pyramid):
        #    cv.imshow("Pyr {}".format(i), img)

        # Weighted pyramid
        weightedPyrImg = None
        for pyrImg in pyramid:
            if weightedPyrImg is None:
                weightedPyrImg = pyrImg.astype(np.float32)
            else:
                weightedPyrImg += pyrImg.astype(np.float32) + blurred.astype(np.float32)
                #weightedPyrImg *= pyrImg.astype(np.float32) # multiply?
                
        weightedPyrImg *= pyramid[-1].astype(np.float32)
        
        weightedPyrImg = weightedPyrImg*255./weightedPyrImg.max()
        weightedPyrImg = weightedPyrImg.astype(np.uint8)
        #_,weightedPyrImg = cv.threshold(weightedPyrImg,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        #cv.imshow("Weighted pyramid", weightedPyrImg)

        
        otsuThreshold, logThresh = cv.threshold(weightedPyrImg, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

        _, contours, hier = cv.findContours(logThresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        cv.drawContours(drawImg, contours, -1, (255, 0, 0), -1)

        candidates = [LightSource(cnt+offset, otsuThreshold) for cnt in contours]
        candidates = candidates[:11]
        #cv.imshow("LoG", logImg)
        #cv.imshow("LoG+Otsu", logThresh)
        self.img = weightedPyrImg
        return drawImg, candidates, roiCnt

class LogDiff(AbstractLightSourceDetector):
    def __init__(self):
        self.localMaxKernel = np.ones((11,11))
        self.localMinKernel = np.ones((50,50))
        self.p = 0.975

        self.img = np.zeros((50,50), dtype=np.uint8)

    def __repr__(self):
        return "Pyramid"

    def detect(self,
               gray, 
               nFeatures,
               additionalCandidates, 
               estDSPose=None,
               roiMargin=None, 
               drawImg=None):

        localMaxImg = localMaxDiff(gray, self.localMaxKernel, self.p)
        localMaxImg = localMaxSupressed(gray, self.localMaxKernel, self.p)
        #localMaxImg = localMaxDiff(gray, self.localMaxKernel, self.p)
        #localMaxImg = localMax(gray, self.localMaxKernel, borderValue=0)

        _, contours, _ = cv.findContours(localMaxImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if drawImg is not None:
            cv.drawContours(drawImg, contours, -1, (255,0,0), -1)

        self.img = localMaxImg

        candidates = [LightSource(cnt, 255) for cnt in contours]
        candidates = candidates[:nFeatures+additionalCandidates]

        return drawImg, candidates, None

class SimpleBlobDetector(AbstractLightSourceDetector):
    def __init__(self):
        self.init(255)

    def init(self, threshold):
        params = cv.SimpleBlobDetector_Params()
        
        # Change thresholds
        params.minThreshold = 0
        params.maxThreshold = threshold
        params.thresholdStep = 1
        # Filter by Area.
        #params.filterByArea = True
        #params.minArea = 1500
        
        # Filter by Circularity
        #params.filterByCircularity = True
        #params.minCircularity = 0.01
        
        # Filter by Convexity
        #params.filterByConvexity = True
        #params.minConvexity = 0.87
        
        # Filter by Inertia
        #params.filterByInertia = True
        #params.minInertiaRatio = 0.01
 
        # Create a detector with the parameters
        ver = (cv.__version__).split('.')
        if int(ver[0]) < 3 :
            self.detector = cv.SimpleBlobDetector(params)
        else:
            self.detector = cv.SimpleBlobDetector_create(params)

        self.img = np.zeros((50,50), dtype=np.uint8)

    def __repr__(self):
        return "Simple blob detector"

    def detect(self,
               gray, 
               nFeatures,
               additionalCandidates, 
               estDSPose=None,
               roiMargin=None, 
               drawImg=None):

        roiCnt = None
        offset = (0, 0)
        if estDSPose:
            
            featurePointsGuess = estDSPose.reProject()
            roiMargin += max([ls.radius for ls in estDSPose.associatedLightSources])
            (x, y, w, h), roiCnt = regionOfInterest(featurePointsGuess, wMargin=roiMargin, hMargin=roiMargin, shape=gray.shape)

            offset = (x, y)
            gray = gray[y:y+h, x:x+w]

            minIntensity = min([ls.intensity for ls in estDSPose.associatedLightSources])
            self.init(minIntensity*0.7)

        # Detect blobs.
        keypoints = self.detector.detect(cv.bitwise_not(gray))

        centers = [(int(k.pt[0]+offset[0]), int(k.pt[1]+offset[1])) for k in keypoints]

        candidates = [LightSource(np.array([[center]]), gray[center[1]-offset[1], center[0]-offset[0]]) for center in centers]
        candidates = candidates[:nFeatures+additionalCandidates]
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        self.img = cv.drawKeypoints(cv.bitwise_not(gray), keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
        # Show keypoints
        #cv.imshow("Keypoints", im_with_keypoints)

        return drawImg, candidates, roiCnt

def separateOverExposedCandidates(candidates, imgShape, offset=(0,0), drawImg=None):
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
        # TODO: don't use a hardcoded threshold
        # TODO: can we apply local peak on the distance transform image?
        _, threshImg = cv.threshold(dist, 150, 255, cv.THRESH_BINARY)
        
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
            cv.drawContours(drawImg, [ls.cnt], 0, (255,0,255), -1)
            cv.circle(drawImg, ls.center, 3, (0,255,255), -1)
            cv.drawContours(binaryImg, [ls.cnt], 0, 255, -1, offset=(-offset[0], -offset[1]))

    candidates = lightSources + overlappingLightSources

    return binaryImg, candidates


def separateOverExposedCandidates2(candidates, imgShape, kernel, offset=(0,0), drawImg=None):
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
        
        (_, _, _, contours, _) = findNPeaks3(dist, 
                                             kernel=kernel, 
                                             pMin=0.9,
                                             pMax=0.9, 
                                             n=5,
                                             offset=offset,
                                             windowRad=0,
                                             maxIter=5*2)

        # if len(contours) == 1:
        #     # If we didn't detect any new light sources, we keep the original one
        #     distTransCandidates.append(ls)
        # else:
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
            cv.drawContours(drawImg, [ls.cnt], 0, (255,0,255), -1)
            cv.circle(drawImg, ls.center, 3, (0,255,255), -1)
            cv.drawContours(binaryImg, [ls.cnt], 0, 255, -1, offset=(-offset[0], -offset[1]))

    candidates = lightSources + overlappingLightSources

    return binaryImg, candidates



if __name__ == "__main__":
    pass
