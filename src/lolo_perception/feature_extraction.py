#!/usr/bin/env python
import time
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy import signal
import rospy
from sensor_msgs.msg import Image
import itertools
from lolo_perception.perception_utils import plotHistogram, regionOfInterest

class PercentageThreshold:
    def __init__(self, p, thresholdType=cv.THRESH_BINARY):
        """
        try cv.THRESH_OTSU?
        """
        self.p = p
        self.thresholdType = thresholdType
        self.threshold = 0 # store the last threshold that was used 
        self.img = np.zeros((10, 10)) # store the last processed image

    def process(self, img):
        r = np.max(img) - np.min(img)
        low = int(np.max(img) - self.p*r)
        self.threshold = low
        ret, img = cv.threshold(img, low, 256, self.thresholdType)
        self.img = img
        return img

class AdaptiveThreshold:
    def __init__(self, nFeatures, thresStart, thresInc, thresMin, thresholdType=cv.THRESH_BINARY):
        """
        try cv.THRESH_OTSU?
        """
        self.nFeatures = nFeatures
        self.threshold = thresStart
        self.thresInc = thresInc
        self.thresMin = thresMin
        self.thresholdType = thresholdType 
        self.img = np.zeros((10, 10)) # store the last processed image

    def process(self, img):
        ret, imgTemp = cv.threshold(img, self.threshold, 256, self.thresholdType)
        _, contours, hier = cv.findContours(imgTemp, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        while len(contours) < self.nFeatures and self.threshold >= self.thresMin+self.thresInc:
            self.threshold -= self.thresInc
            ret, imgTemp = cv.threshold(img, self.threshold, 256, self.thresholdType)
            _, contours, hier = cv.findContours(imgTemp, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        self.img = imgTemp
        return imgTemp

class AdaptiveErode:
    """
    Iteratively uses "opening" operations until the nr points detected in the 
    image are as close to but at least the desired nFeatures.
    The number of iterations are saved until next call to prevent unecessary reiteration.
    """
    def __init__(self, nFeatures, kernelSize=5, kernelType=cv.MORPH_ELLIPSE, startIterations=1):
        self.iterations = startIterations
        self.nFeatures = nFeatures
        self.kernel = cv.getStructuringElement(kernelType, (kernelSize,kernelSize)) 
        self.img = np.zeros((10, 10))

    def _findCandidatesByExclution(self, img, contours):
        """
        Find candidates by erodin the image
        """
        i = 0
        while len(contours) > self.nFeatures:
            i += 1
            imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_OPEN, self.kernel, iterations=i)
            #imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_ERODE, self.kernel, iterations=i)
            _, contours, hier = cv.findContours(imgTemp, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        if len(contours) < self.nFeatures:
            i -= 1
            imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_OPEN, self.kernel, iterations=i)
            #imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_ERODE, self.kernel, iterations=i)
            _, contours, hier = cv.findContours(imgTemp, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            f = lambda cnt, img: contourAveragePixelIntensity(cnt, img)
            contours.sort(key=f, reverse=True)
            #avgPixelInt = contourAveragePixelIntensity(cnt, img)
            
        return list(contours)

    def _findCandidatesByInclusion(self, img, contours):
        
        while len(contours) < self.nFeatures and i > 0:
            i -= 1
            imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_OPEN, self.kernel, iterations=i)
            #imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_ERODE, self.kernel, iterations=i)
            #imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_CROSS,(5,5)), iterations=i)
            points = _contours(imgTemp, nFeatures=None)

    def process(self, img):
        # img - binary image
        # this removes reflections effectively at close distances but reduces range
        
        imgTemp = img.copy()
        _, contours, hier = cv.findContours(imgTemp, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        i = self.iterations

        
        N = len(contours)
        contourCandidates = []
        # find candidates
        if N == self.nFeatures:
            contourCandidates = list(contours)
        elif N > self.nFeatures:
            contourCandidates = self._findCandidatesByExclution(img, contours)
        elif N < self.nFeatures:
            contourCandidates = self._findCandidatesByInclusion(img, contours)

        i = self.iterations
        while len(contours) > self.nFeatures:
            i += 1
            #imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_OPEN, self.kernel, iterations=i)
            imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_ERODE, self.kernel, iterations=i)
            _, contours, hier = cv.findContours(imgTemp, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)    
        
        contours.sort(key=contourRatio, reverse=True)

        while len(points) < self.nFeatures and i > 0:
            i -= 1
            imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_OPEN, self.kernel, iterations=i)
            #imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_ERODE, self.kernel, iterations=i)
            #imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_CROSS,(5,5)), iterations=i)
            points = _contours(imgTemp, nFeatures=None)

        self.iterations = i # save to we don't have to try next time
        img = imgTemp
        self.img = img
        return img

class AdaptiveOpen:
    """
    Iteratively uses "opening" operations until the nr points detected in the 
    image are as close to but at least the desired nFeatures.
    The number of iterations are saved until next call to prevent unecessary reiteration.
    """
    def __init__(self, nFeatures, kernelSize=5, kernelType=cv.MORPH_ELLIPSE, startIterations=1, maxIter=10):
        self.iterations = startIterations
        self.maxIter = maxIter
        self.nFeatures = nFeatures
        self.kernel = cv.getStructuringElement(kernelType, (kernelSize, kernelSize)) 
        self.img = np.zeros((10, 10))
        self.extraContours = 0

    def process(self, img):
        # this removes reflections effectively at close distances but reduces range

        imgTemp = img.copy()

        i = self.iterations
        imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_OPEN, self.kernel, iterations=i)
        _, contours, hier = cv.findContours(imgTemp, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        while len(contours) >= (self.nFeatures+self.extraContours) and i < self.maxIter:
            i += 1
            # doesnt really matter if open or erode is used?
            imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_OPEN, self.kernel, iterations=i)
            _, contours, hier = cv.findContours(imgTemp, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        while len(contours) < self.nFeatures and i > 0:
            i -= 1
            # doesnt really matter if open or erode is used?
            imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_OPEN, self.kernel, iterations=i)
            _, contours, hier = cv.findContours(imgTemp, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        self.iterations = max(i, 0) # save to we don't have to try next time
        img = imgTemp
        self.img = img
        print("Open iterations:", self.iterations)
        print("Kernel size", self.kernel.shape)
        return img

class AdaptiveErodeKernel:
    """
    Iteratively uses "opening" operations until the nr points detected in the 
    image are as close to but at least the desired nFeatures.
    The number of iterations are saved until next call to prevent unecessary reiteration.
    """
    def __init__(self, nFeatures, kernelSize=15, kernelType=cv.MORPH_ELLIPSE, startIterations=1, maxIter=10):
        self.iterations = startIterations
        self.maxIter = maxIter
        self.nFeatures = nFeatures
        self.kernelSize = kernelSize
        self.kernel = cv.getStructuringElement(kernelType, (kernelSize,kernelSize)) 
        self.img = np.zeros((10, 10))

    def process(self, img):
        # this removes reflections effectively at close distances but reduces range
        _, contours, hier = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        contours.sort(key=contourRatio, reverse=True)

        imgTemp = img.copy()        
        i = self.iterations
        while len(contours) >= self.nFeatures and i <= self.maxIter:
            # doesnt really matter if open or erode is used?
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (i, i))
            imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_OPEN, kernel)
            _, contours, hier = cv.findContours(imgTemp, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            #contours = removeContoursOnEdges(contours)
            #contours.sort(key=contourRatio, reverse=True)
            i += 1
        i -= 1

        while len(contours) < self.nFeatures and i > 1:
            i -= 1
            # doesnt really matter if open or erode is used?
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (i, i))
            imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_OPEN, kernel)
            _, contours, hier = cv.findContours(imgTemp, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            #contours = removeContoursOnEdges(contours)
            #contours.sort(key=contourRatio, reverse=True)

        self.iterations = max(i, 1) # save to we don't have to try next time
        img = imgTemp
        self.img = img
        print("Kernel size:", self.iterations)
        return img

def drawInfo(img, center, text, color=(255, 0, 0), fontScale=1, thickness=2):
    # font
    font = cv.FONT_HERSHEY_SIMPLEX
    # org
    org = (center[0]-10, center[1]+10) # some displacement to make it look good
    # fontScale
    #fontScale = 1
    # Line thickness of 2 px
    #thickness = 2
    # Using cv2.putText() method
    image = cv.putText(img, text, org, font, 
                    fontScale, color, thickness, cv.LINE_AA)
    
    return image

def circularKernel(size):
    """
    size - radius of circle
    """
    assert size % 2 == 1, "Must be of uneven size"
    radius = size/2
    center = (radius, radius)
    kernel = np.zeros((size, size), dtype=np.uint8)
    cv.circle(kernel, center, radius, 1, -1)

    return kernel

def fillContours(img, ratio):
    _, contours, hier = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        r = contourRatio(cnt)
        if r < ratio:
            cv.fillPoly(img, pts=[cnt], color=(0,0,0))
        else:
            cv.fillPoly(img, pts=[cnt], color=(255,255,255))

    return img

def fillContoursOnEdges(img):
    _, contours, hier = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cv.fillPoly(img, contoursOnEdges(contours, img.shape), color=(0, 0, 0))
    return img

def contoursOnEdges(contours, resolution):
    onEdges = []
    for cnt in contours:
        leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
        topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
        bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
        if leftmost[0] == 0 or rightmost[0] == resolution[1]-1 or topmost[1] == 0 or bottommost[1] == resolution[0]-1:
            onEdges.append(cnt)

    return onEdges

def removeContoursOnEdges(img, contours):
    newContours = []
    for cnt in contours:
        leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
        topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
        bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
        
        if leftmost[0] != 0 and rightmost[0] != img.shape[1]-1 and topmost[1] != 0 and bottommost[1] != img.shape[0]-1:
            #print(leftmost)
            #print(rightmost)
            newContours.append(cnt)
        else:
            pass
            #print("Removed contour on edge")
    return newContours

def contourAveragePixelIntensity(cnt, img):
    # https://stackoverflow.com/questions/63456523/opencv-average-intensity-of-contour-figures-in-grayscale-image
    mask = np.zeros(img.shape, np.uint8)
    cv.fillPoly(mask, pts=[cnt], color=(255,255,255))
    mask_contour = mask == 255
    intensity = np.mean(img[mask_contour])
    return intensity

def contourMaxPixelIntensity(cnt, img):
    # https://stackoverflow.com/questions/63456523/opencv-average-intensity-of-contour-figures-in-grayscale-image
    mask = np.zeros(img.shape, np.uint8)
    cv.fillPoly(mask, pts=[cnt], color=(255,255,255))
    mask_contour = mask == 255
    intensity = np.max(img[mask_contour])
    return intensity

def contourMinPixelIntensity(cnt, img):
    # https://stackoverflow.com/questions/63456523/opencv-average-intensity-of-contour-figures-in-grayscale-image
    mask = np.zeros(img.shape, np.uint8)
    cv.fillPoly(mask, pts=[cnt], color=(255,255,255))
    mask_contour = mask == 255
    intensity = np.min(img[mask_contour])
    return intensity

def contourCentroid(cnt):
    area = cv.contourArea(cnt)
    if area == 0:
        cx, cy = cnt[0][0]
    else:
        M = cv.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    return cx, cy

def withinContour(cx, cy, cnt):
    result = cv.pointPolygonTest(cnt, (cx,cy), False)
    if result in (1, 0): # inside or on the contour
        return True
    return False

def nextToContour(cx, cy, cnt, margin=1):
    result = cv.pointPolygonTest(cnt, (cx,cy), True)
    if result >= -margin: #TODO: if outside the contour, is the result always negative?
        return True
    return False

def medianContourArea(contours):
    contourAreas = [cv.contourArea(cnt) for cnt in contours]
    return np.median(contourAreas)
    
def medianContourAreaFromImg(img):
    _, contours, hier = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    return medianContourArea(contours)

def removeNeighbouringContours(contours, minDist, key):
    """
    contours - list of contours
    minDist - contours within minDist will be candidates for removal (contour position determined by minEnclosingCircle)
    key - function callback that determines which out of two contours are removed (key1 > key2 == True means cnt2 will be removed)
          (contourArea is an example for key function)
    """
    newContours = []
    keys = [key(cnt) for cnt in contours]
    removedIdxs = []
    for i in range(len(contours)):
        for j in range(i + 1, len(contours)):
            cntPos1, r1 = cv.minEnclosingCircle(contours[i])
            cntPos2, r2 = cv.minEnclosingCircle(contours[j])
            if np.linalg.norm([cntPos1[0]-cntPos2[0], cntPos1[1]-cntPos2[1]]) < r1+r2+minDist:
                if keys[i] < keys[j]:
                    break
                else:
                    removedIdxs.append(j)
        else:
            if i not in removedIdxs:
                newContours.append(contours[i])

    return newContours

def removeNeighbouringContoursFromImg(img, minDist, key):
    """
    Removes neighbouring contours by filling erased contours with black
    """
    _, contours, hier = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    newContours = removeNeighbouringContours(contours, minDist, key)
    newImg = np.zeros(img.shape, dtype=np.uint8)
    cv.drawContours(newImg, newContours, -1, (255, 255, 255), -1)

    return newImg

def contourRatio(cnt):
    area = cv.contourArea(cnt)
    
    shape = "circ"
    if shape == "circ":
        (x,y),radius = cv.minEnclosingCircle(cnt)
        enclosingArea = (np.pi*radius*radius)
        if radius == 0:
            print("WTF")
            ratio = 1
        else:
            ratio = round(area/enclosingArea, 2)
    else:
        x,y,w,h = cv.boundingRect(cnt)
        enclosingArea = w*h
        ratio = round(area/enclosingArea, 2)

    return ratio

def contoursMinAreaRemove(contours, minArea=3):
    """Remove contours with area < minArea"""
    contoursNew = []
    for cnt in contours:
        if cv.contourArea(cnt) >= minArea:
            contoursNew.append(cnt)
    return contoursNew

def contourAreaDistanceFromMean(contours):
    areas = np.array([cv.contourArea(cnt) for cnt in contours])
    mean = np.mean(areas)
    std = np.std(areas)
    distanceFromMean = abs(areas - mean)
    return distanceFromMean/std

def contourAreaOutliers(contours):
    maxStd = 5
    areas = np.array([cv.contourArea(cnt) for cnt in contours])
    mean = np.mean(areas)
    std = np.std(areas)
    distanceFromMean = abs(areas - mean)
    outliers = distanceFromMean > maxStd*std
    
    return [cnt for outlier, cnt in zip(outliers, contours) if not outlier]
    #return outliers

def contourPositionY(cnt):
    maxY = 0
    for v in cnt:
        _, y = v[0]
        maxY = max(maxY, y)

    return maxY

def weightedCentroid(patch):
    # https://stackoverflow.com/questions/53719588/calculate-a-centroid-with-opencv
    M = cv.moments(patch)
    if M["m00"] == 0:
        return None
    cx = int(round(M["m10"] / M["m00"]))
    cy = int(round(M["m01"] / M["m00"]))
    return cx, cy

def refineCentroidGradient(gray, contours, ksize=3):
    g = GradientFeatureExtractor(None, None, ksize)
    gradImg = g(gray)
    gradImg = cv.GaussianBlur(gradImg, (ksize,ksize), 0)

    centroids = []

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv.drawContours(mask, contours, -1, (255), 1)
    gradMasked = cv.bitwise_and(gradImg, gradImg, mask=mask)
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        patch = gradMasked[y:y+h, x:x+w]
        centroid = weightedCentroid(patch)
        centroid = centroid[0]+x, centroid[1]+y
        centroids.append(centroid)

    return centroids

def RCF((x,y), r, gray):
    I = float(gray[y, x])

    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask = cv.circle(mask, (x,y), r, (1), -1)
    mask[y, x] = 0
    N = np.sum(mask)

    grayMasked = cv.bitwise_and(gray, gray, mask=mask)
    
    mean = np.sum(grayMasked)/float(N)
    rcf = (I-mean)/I

    return rcf

def RCFS((i, j), rStart, gray, rInc=1, drawImg=None):
    """
    Computes the sum of neighbouring pixels at radius R
    Based on section 4.2 in https://www.tdx.cat/bitstream/handle/10803/275990/tmmm1de1.pdf?sequence=5
    i - pixel x coordinate
    j - pixel y coordinate
    nStart - start radius (increment is 0.5)
    nEnd - end radius (increment is 0.5)
    gray - gray scale image
    rcfs - list of rcfs from first radius to last
    """ 
    assert rStart >= 1, "Start radius must be larger than 1"

    def continueCondition(rcfs, maxDiff, n):
        # when the last n samples has had a slope lesser than maxDiff
        # we break
        
        if len(rcfs) >= n+1:
            for i in range(n):
                diff = (rcfs[-n+i+1] - rcfs[-n+i])

                if diff >= maxDiff:
                    return True
            else:
                return False
        else:
            return True
    
    rcfSum = 0
    N = 0.0
    I = float(gray[j, i])
    rcfs = []
    r = 1
    #rInc = 1

    maxDiff = 0#RCF((i,j), r, gray.copy())
    n = 6
    while continueCondition(rcfs, maxDiff, n):
        rcf = RCF((i,j), r, gray.copy())

        if rcfs:
            diff = rcf - rcfs[-1]
            if diff < 0:
                # not a light source
                return 1, []
            if diff > maxDiff:
                maxDiff = diff

        rcfs.append(rcf)
        r += rInc
        r = int(round(r))
        if drawImg is not None:
            cv.circle(drawImg, (i,j), int(r), (0, 0, 150), 1)

    r -= int(n*rInc)
    rcfs = rcfs[:len(rcfs)-n]
    return r, rcfs

    """
    Computes the sum of neighbouring pixels at radius R
    Based on section 4.2 in https://www.tdx.cat/bitstream/handle/10803/275990/tmmm1de1.pdf?sequence=5
    i - pixel x coordinate
    j - pixel y coordinate
    nStart - start radius (increment is 0.5)
    nEnd - end radius (increment is 0.5)
    gray - gray scale image
    rcfs - list of rcfs from first radius to last
    """ 
    assert rStart >= 1, "Start radius must be larger than 1"

    I = float(gray[j, i])
    rcfSum = 0.0
    nNeigh = 0.0
    rcfs = []
    print(gray.shape)
    for r in np.arange(rStart, rEnd, 0.5): 
        sum, N = RCFSumAtR(i, j, r, gray)
        rcfSum += sum
        nNeigh += N
        rcf = (I - rcfSum/nNeigh)/I
        rcfs.append(rcf)
        if drawImg is not None:
            print(i, j)
            cv.circle(drawImg, (i,j), int(r), (0, 0, 150), 1)

    return rcfs

def peakThreshold(gray, kernel, p, other=None):
    data = gray
    #ret, data = cv.threshold(grayMasked, self.analyzeThreshold, 256, cv.THRESH_TOZERO)

    #data_max = cv.morphologyEx(data, cv.MORPH_DILATE, np.ones((k,k)))
    if other is not None:
        dataMax = other
        diff = (other > 0)
        data[diff == 0] = 0
    else:
        dataMax = cv.dilate(data, kernel)
    #data_min = cv.morphologyEx(data, cv.MORPH_ERODE, np.ones((k,k)))
    diff = (data > dataMax*p)

    maxima = data.copy()
    maxima[diff == 0] = 0
    
    return maxima

def peakThresholdMin(gray, kernel, p):
    data = gray
    #ret, data = cv.threshold(grayMasked, self.analyzeThreshold, 256, cv.THRESH_TOZERO)
    kernel = circularKernel(25)
    #data_max = cv.morphologyEx(data, cv.MORPH_DILATE, np.ones((k,k)))
    locMax = localMax(gray, kernel)
    dataMax = cv.dilate(data, kernel)
    dataMin = cv.erode(data, kernel)
    diff = (dataMin.astype(np.float32) < dataMax.astype(np.float32)*p)

    maxima = data.copy()
    maxima[diff == 0] = 0
    
    kernel = circularKernel(11)
    ret, mask = cv.threshold(locMax, 0, 256, cv.THRESH_BINARY)
    maxima = cv.bitwise_and(maxima, maxima, mask=mask)

    ret, add = cv.threshold(locMax, 254, 255, cv.THRESH_BINARY)

    maxima[add > 0] = 255

    return maxima

def findPeaksDilation(gray, kernel):
    """
    kernelSizes - iterable (list or tuple) of kernelSize (int)
    """
    # We set border value to max to prevent false positive peaks at edges
    dilated = cv.dilate(gray.copy(), kernel, borderValue=255, iterations=1)
    #dilated = cv.morphologyEx(gray.copy(), cv.MORPH_OPEN, kernel, iterations=1)
    indices = np.where(gray == dilated)
    peaksDilated = np.zeros(gray.shape, np.uint8)
    peaksDilated[indices] = gray[indices]
    return peaksDilated

def localMax(gray, kernel):
    dilated = cv.dilate(gray, kernel, borderValue=255, iterations=1)
    eqMask = cv.compare(gray, dilated, cv.CMP_EQ)
    localMaxImg = cv.bitwise_and(gray, gray, mask=eqMask)
    return localMaxImg

def localMaxSupressed2(gray, kernel, p):
    seed = gray*p
    seed = seed.astype(np.uint8)
    dilated = cv.dilate(seed, kernel, borderValue=0, iterations=1)
    eqMask = cv.compare(gray, dilated, cv.CMP_GE)
    localMaxImg = cv.bitwise_and(gray, gray, mask=eqMask)
    return localMaxImg

def localMaxSupressed(gray, kernel, p):
    gray = gray*p
    gray = gray.astype(np.uint8)
    dilated = cv.dilate(gray, kernel, borderValue=0, iterations=1)
    eqMask = cv.compare(gray, dilated, cv.CMP_GE)
    localMaxImg = cv.bitwise_and(gray, gray, mask=eqMask)
    return localMaxImg

def localMaxChange(gray, kernel, p):
    dilated = cv.dilate(gray, kernel, borderValue=255, iterations=1)
    eroded = cv.erode(gray, kernel, borderValue=255, iterations=1)
    if p < 1:
        # Percentage threshold
        dilatedSupressed = dilated.astype(np.float32)*p
    else:
        # Fixed threshold
        dilatedSupressed = dilated-p
    dilatedSupressed = dilatedSupressed.astype(np.uint8)
    eqMask = cv.compare(dilatedSupressed, eroded, cv.CMP_GE)
    localMaxImg = cv.bitwise_and(gray, gray, mask=eqMask)
    return localMaxImg

def localMin(gray, kernel):
    dilated = cv.erode(gray, kernel, borderValue=0, iterations=1)
    eqMask = cv.compare(gray, dilated, cv.CMP_EQ)
    localMaxImg = cv.bitwise_and(gray, gray, mask=eqMask)
    return localMaxImg


def findContourAt(gray, center):    
    _, contours, hier = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    contours.sort(key=cv.contourArea, reverse=True)

    foundCnt = None
    if contours:
        for cnt in contours:
            if withinContour(center[0], center[1], cnt):
                foundCnt = cnt
                break

                #TODO: Not sure if this should be done
                if foundCnt is None:
                    foundCnt = cnt
                else:
                    if cv.contourArea(cnt) > cv.contourArea(foundCnt):
                        foundCnt = cnt

    else:
        return None

    """
    cv.drawContours(img, [foundCnt], 0, (255, 0, 0), -1)
    if cv.contourArea(foundCnt) < 10:
        ratioThresh = 0.2
    else:
        ratioThresh = 0.5
    color = (255, 0, 0)
    if contourRatio(foundCnt) < ratioThresh:
        color = (0, 0, 255)
    drawInfo(img, (cx+20, cy-20), str(contourRatio(foundCnt)), color=color)
    """
    return foundCnt

def findPeakContourAt(gray, center, offset=None, mode=cv.RETR_EXTERNAL):
    _, contours, hier = cv.findContours(gray, mode, cv.CHAIN_APPROX_SIMPLE)

    
    if offset is None:
        contoursOffset = [cnt for cnt in contours]
    else:
        _, contoursOffset, hier = cv.findContours(gray, mode, cv.CHAIN_APPROX_SIMPLE, offset=offset)

    #contours.sort(key=cv.contourArea, reverse=True)
    #contoursOffset.sort(key=cv.contourArea, reverse=True)

    foundCnt = None
    foundCntOffset = None
    if contours:
        for cnt, cntOffset in zip(contours, contoursOffset):
            if withinContour(center[0], center[1], cnt):
                foundCnt = cnt
                foundCntOffset = cntOffset
                break

                #TODO: Not sure if this should be done
                if foundCnt is None:
                    foundCnt = cnt
                else:
                    if cv.contourArea(cnt) > cv.contourArea(foundCnt):
                        foundCnt = cnt

    else:
        return None

    return foundCnt, foundCntOffset

def findMaxPeakAt(gray, center, p):
    
    _, grayThreholded = cv.threshold(gray, maxIntensity-1, 256, cv.THRESH_TOZERO)

def findMaxPeak(gray, p):
    """
    Finds the maximum peak and the contour surrounding p % of its max value 
    """
    maxIndx = np.unravel_index(np.argmax(gray), gray.shape)
    center = maxIndx[1], maxIndx[0]

    return center, findPeakContourAt(threshImg, center, p)

def _findMaxPeaks(gray, p, drawImg=None):
    grayMasked = gray.copy()
    maxIntensity = np.max(gray)
    #_, grayThreholded = cv.threshold(gray, maxIntensity-1, 256, cv.THRESH_TOZERO) # possibly faster thresholding first
    peakCenters = []
    peakContours = []
    while np.max(grayMasked) == maxIntensity:
        center, cntPeak = findMaxPeak(grayMasked, p)
        if cntPeak is None:
            break

        """
        TODO: Is this necessary?
        for (cx,cy), cnt in zip(peakCenters, peakContours):
            if withinContour(cx, cy, cntPeak):
                raise Exception("This shouldn't happen")
                if cv.contourArea(cntPeak) > cv.contourArea(cnt):
                    peakContours.remove(cnt)
                    peakCenters.remove((cx, cy))

                    peakCenters.append(center)
                    peakContours.append(cntPeak)
        else:
            peakCenters.append(center)
            peakContours.append(cntPeak)
        """
        peakCenters.append(center)
        peakContours.append(cntPeak)
        grayMasked = cv.drawContours(grayMasked, [cntPeak], 0, (0, 0, 0), -1)
        if drawImg is not None:
            cv.drawContours(drawImg, [cntPeak], 0, (255, 0, 0), 1)

    return peakCenters, peakContours

def findMaxPeaks(gray, p, drawImg=None):
    grayMasked = gray.copy()
    maxIntensity = np.max(gray)
    #_, grayThreholded = cv.threshold(gray, maxIntensity-1, 256, cv.THRESH_TOZERO) # possibly faster thresholding first
    peakCenters = []
    peakContours = []
    while np.max(grayMasked) == maxIntensity:
        center, cntPeak = findMaxPeak(grayMasked, p)
        if cntPeak is None:
            break

        """
        TODO: Is this necessary?
        for (cx,cy), cnt in zip(peakCenters, peakContours):
            if withinContour(cx, cy, cntPeak):
                raise Exception("This shouldn't happen")
                if cv.contourArea(cntPeak) > cv.contourArea(cnt):
                    peakContours.remove(cnt)
                    peakCenters.remove((cx, cy))

                    peakCenters.append(center)
                    peakContours.append(cntPeak)
        else:
            peakCenters.append(center)
            peakContours.append(cntPeak)
        """
        peakCenters.append(center)
        peakContours.append(cntPeak)
        grayMasked = cv.drawContours(grayMasked, [cntPeak], 0, (0, 0, 0), -1)
        if drawImg is not None:
            cv.drawContours(drawImg, [cntPeak], 0, (255, 0, 0), 1)

    return peakCenters, peakContours

def pDecay(b, pMin, pMax, I, IMax=255.):
    assert 1-(pMax-pMin)/b > 0, "Invalid value of b, (b >={})".format(pMax-pMin)
    c = -1./255*np.log(1-(pMax-pMin)/b)
    return pMin + b*(1-np.exp(-c*I))


def findNPeaks2(gray, kernel, pMin, pMax, n, minThresh=0, margin=1, ignorePAtMax=True, offset=(0,0), maxIter=10000000, drawImg=None, drawInvalidPeaks=False):
    peaksDilation = localMax(gray, kernel) # local max
    _, peaksDilation = cv.threshold(peaksDilation, minThresh, 256, cv.THRESH_TOZERO)

    grayMasked = gray.copy()
    peaksDilationMasked = peaksDilation.copy()
    maxIntensity = np.inf
    peakCenters = []
    peakContours = []

    iterations = 0
    while True:
        if np.max(peaksDilationMasked) != maxIntensity:
            if len(peakCenters) >= n:
                print("Found {} peaks, breaking".format(n))
                break
            else:
                maxIntensity = np.max(peaksDilationMasked)
                if maxIntensity == 0:
                    break
                if maxIntensity == 255 and ignorePAtMax:
                    # if it is maximum intensity, ignore p
                    threshold = 254
                    #threshold = 251 # TODO: maybe another value is more suitable?
                else:
                    #pTemp = maxIntensity/255.*(pMax-pMin) + pMin
                    pTemp = pDecay(.175, pMin, pMax, I=maxIntensity)
                    threshold = int(pTemp*maxIntensity - 1)

                ret, threshImg = cv.threshold(grayMasked, threshold, 256, cv.THRESH_BINARY)


        iterations += 1
        # TODO: this could probably be more efficient, extracting all contours at this intensity level at the same time
        maxIndx = np.unravel_index(np.argmax(peaksDilationMasked), gray.shape)
        center = maxIndx[1], maxIndx[0]
        cntPeak, cntPeakOffset = findPeakContourAt(threshImg, center, offset=offset)#, mode=cv.RETR_LIST)

        if cntPeak is None:
            print("Something went wrong?!")
            break

        # TODO: Check if contour includes the threshold value.
        # If it doesn't, we can probably break here (noise).
        # TODO: Check convexity. Non-convex shapes are probably not light sources

        # Check if contour is on edge
        cnts = [cntPeak]
        if maxIntensity != 255:
            # we only remove edge contours if they don't have max value
            cnts = removeContoursOnEdges(peaksDilation, cnts)
        if not cnts:
            if drawImg is not None and drawInvalidPeaks:
                    cv.drawContours(drawImg, [cntPeakOffset], 0, (0, 255, 255), 1)
        else:
            for pc in peakCenters:
                if withinContour(pc[0], pc[1], cntPeakOffset) or nextToContour(pc[0], pc[1], cntPeakOffset, margin=margin): # TODO: this should be checked using the center of the new peak instead
                    if drawImg is not None and drawInvalidPeaks:
                        cv.drawContours(drawImg, [cntPeakOffset], 0, (0, 0, 255), 1)
                    break
            else:
                peakCenters.append((center[0]+offset[0], center[1]+offset[1]))
                peakContours.append(cntPeakOffset)
                if drawImg is not None:
                    cv.drawContours(drawImg, [cntPeakOffset], 0, (255, 0, 0), -1)

        # Always used this but I think it makes it worse:
        # 1. Weird shapes around already found peaks
        # 2. The weird shapes are not peaks, and return false positives affects the speed of 
        #    iterating through all combinations of light sources
        #grayMasked = cv.drawContours(grayMasked, [cntPeak], 0, (0, 0, 0), -1)
        peaksDilationMasked = cv.drawContours(peaksDilationMasked, [cntPeak], 0, (0, 0, 0), -1)

        if iterations >= maxIter:
            print("Peak maxiter reached")
            break
    
    if drawImg is not None:
        for i, pc in enumerate(peakCenters):
            cv.circle(drawImg, pc, 1, (255,0,255), 1)
            drawInfo(drawImg, (pc[0]+15,pc[1]-15), str(i+1))

    return peaksDilation, peaksDilationMasked, peakCenters, peakContours, iterations

def findNPeaksWithBlur(gray, kernel, pMin, pMax, n, blurKernelSize, minThresh=0, margin=1, offset=(0,0), maxIter=10000000, drawImg=None, drawInvalidPeaks=False):
    
    blurred = cv.GaussianBlur(gray, (blurKernelSize,blurKernelSize), 0) 

    peaksDilation = localMax(gray, kernel) # local max
    _, peaksDilation = cv.threshold(peaksDilation, minThresh, 256, cv.THRESH_TOZERO)

    grayMasked = gray.copy()
    peaksDilationMasked = peaksDilation.copy()
    maxIntensity = np.inf
    #_, grayThreholded = cv.threshold(gray, maxIntensity-1, 256, cv.THRESH_TOZERO) # possibly faster thresholding first
    peakCenters = []
    peakContours = []

    iterations = 0
    while True:
        if np.max(peaksDilationMasked) != maxIntensity: 
            if len(peakCenters) >= n:
                break
            else:
                maxIntensity = np.max(peaksDilationMasked)
                if maxIntensity == 0:
                    break
                if maxIntensity == 255:
                    # if it is maximum intensity, ignore p
                    threshold = 254
                else:
                    maxIndx = np.unravel_index(np.argmax(peaksDilationMasked), gray.shape)
                    averageIntensity = blurred[maxIndx]

                    pTemp = averageIntensity/255.*(pMax-pMin) + pMin

                    threshold = int(pTemp*averageIntensity - 1)

                ret, threshImg = cv.threshold(grayMasked, threshold, 256, cv.THRESH_BINARY)

        iterations += 1
        # TODO: this could probably be more efficient, extracting all contours at this intensity level at the same time
        maxIndx = np.unravel_index(np.argmax(peaksDilationMasked), gray.shape)
        center = maxIndx[1], maxIndx[0]
        cntPeak, cntPeakOffset = findPeakContourAt(threshImg, center, offset=offset)#, mode=cv.RETR_LIST)

        if cntPeak is None:
            break

        # Check if contour is on edge
        cnts = [cntPeak]
        if maxIntensity != 255:
            # we only remove edge contours if they don't have max value
            cnts = removeContoursOnEdges(peaksDilation, cnts)
        if not cnts:
            if drawImg is not None and drawInvalidPeaks:
                    cv.drawContours(drawImg, [cntPeakOffset], 0, (0, 0, 255), 1)
        else:
            for pc in peakCenters:
                if withinContour(pc[0], pc[1], cntPeakOffset) or nextToContour(pc[0], pc[1], cntPeakOffset, margin=margin): # TODO: this should be checked using the center of the new peak instead
                    if drawImg is not None and drawInvalidPeaks:
                        cv.drawContours(drawImg, [cntPeakOffset], 0, (0, 255, 255), 1)
                    break
            else:
                peakCenters.append((center[0]+offset[0], center[1]+offset[1]))
                peakContours.append(cntPeakOffset)
                if drawImg is not None:
                    cv.drawContours(drawImg, [cntPeakOffset], 0, (255, 0, 0), -1)

        # Always used this but I think it makes it worse:
        # 1. Weird shapes around already found peaks
        # 2. The weird shapes are not peaks, and return false positives affects the speed of 
        #    iterating through all combinations of light sources
        #grayMasked = cv.drawContours(grayMasked, [cntPeak], 0, (0, 0, 0), -1)
        peaksDilationMasked = cv.drawContours(peaksDilationMasked, [cntPeak], 0, (0, 0, 0), -1)

        if iterations >= maxIter:
            print("Peak maxiter reached")
            break
    
    if drawImg is not None:
        for i, pc in enumerate(peakCenters):
            cv.circle(drawImg, pc, 1, (255,0,255), 1)
            drawInfo(drawImg, (pc[0]+15,pc[1]-15), str(i+1))

    return peaksDilation, peaksDilationMasked, peakCenters, peakContours, iterations


def findNPeaksWithSignificance(gray, kernel, p, n, rcfRadius, rcfThresh, margin=1, offset=(0,0), drawImg=None, drawInvalidPeaks=False):
    peaksDilation = localMax(gray, kernel) # local max

    grayMasked = gray.copy()
    peaksDilationMasked = peaksDilation.copy()
    maxIntensity = np.inf
    #_, grayThreholded = cv.threshold(gray, maxIntensity-1, 256, cv.THRESH_TOZERO) # possibly faster thresholding first
    peakCenters = []
    peakContours = []

    while True:
        if np.max(peaksDilationMasked) != maxIntensity: 
            if len(peakCenters) >= n:
                break
            else:
                maxIntensity = np.max(peaksDilationMasked)
                if maxIntensity == 0:
                    break
                threshold = int(p*maxIntensity - 1)
                ret, threshImg = cv.threshold(grayMasked, threshold, 256, cv.THRESH_BINARY)

        # TODO: this could probably be more efficient, extracting all contours at this intensity level at the same time
        maxIndx = np.unravel_index(np.argmax(peaksDilationMasked), gray.shape)
        center = maxIndx[1], maxIndx[0]
        cntPeak, cntPeakOffset = findPeakContourAt(threshImg, center, offset=offset)#, mode=cv.RETR_LIST)

        if cntPeak is None:
            break

        # check significance of peak based on rcf
        minRad = 11
        if cv.minEnclosingCircle(cntPeak)[1] < minRad and maxIntensity < 255:
            rcfRad = rcfRadius
            rcfRad = max(rcfRad, minRad)
            centroid = contourCentroid(cntPeak)
            rcf = RCF(np.array(centroid), rcfRad, gray)

            if rcf < rcfThresh:
                if drawImg is not None:
                    cv.circle(drawImg, (centroid[0]+offset[0], centroid[1]+offset[1]), rcfRad, (0, 0, 255), 1)
                    cv.drawContours(drawImg, [cntPeakOffset], 0, (0, 0, 255), 1)
                grayMasked = cv.drawContours(grayMasked, [cntPeak], 0, (0, 0, 0), -1)
                peaksDilationMasked = cv.drawContours(peaksDilationMasked, [cntPeak], 0, (0, 0, 0), -1)
                continue

        for pc in peakCenters:
            if withinContour(pc[0], pc[1], cntPeakOffset) or nextToContour(pc[0], pc[1], cntPeakOffset, margin=margin): # TODO: this should be checked using the center of the new peak instead
                if drawImg is not None and drawInvalidPeaks:
                    cv.drawContours(drawImg, [cntPeakOffset], 0, (0, 0, 255), 1)
                break
        else:
            peakCenters.append((center[0]+offset[0], center[1]+offset[1]))
            peakContours.append(cntPeakOffset)
            if drawImg is not None:
                cv.drawContours(drawImg, [cntPeakOffset], 0, (255, 0, 0), -1)
        grayMasked = cv.drawContours(grayMasked, [cntPeak], 0, (0, 0, 0), -1)
        peaksDilationMasked = cv.drawContours(peaksDilationMasked, [cntPeak], 0, (0, 0, 0), -1)
    
    if drawImg is not None:
        for i, pc in enumerate(peakCenters):
            cv.circle(drawImg, pc, 1, (255,0,255), 1)
            drawInfo(drawImg, (pc[0]+15,pc[1]-15), str(i+1))

    return peaksDilation, peaksDilationMasked, peakCenters, peakContours

def findNLocalMax(gray, kernel, p, n, margin=1, offset=(0,0), drawImg=None, drawInvalidPeaks=False):
    peaksDilation = findPeaksDilation(gray, kernel) # local max

    peakCenters = []
    peakContours = []

    _, contours, hier = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if offset is None:
        contoursOffset = [cnt for cnt in contours]
    else:
        _, contoursOffset, hier = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE, offset=offset)
    
    contoursOffset.sort(key=lambda cnt: contourMaxPixelIntensity(gray, cnt), reverse=True)

    while True:
        if np.max(peaksDilationMasked) != maxIntensity: 
            if len(peakCenters) >= n:
                break
            else:
                maxIntensity = np.max(peaksDilationMasked)
                if maxIntensity == 0:
                    break

        # TODO: this could probably be more efficient, extracting all contours at this intensity level at the same time
        maxIndx = np.unravel_index(np.argmax(peaksDilationMasked), gray.shape)
        center = maxIndx[1], maxIndx[0]
        cntPeak, cntPeakOffset = findPeakContourAt(peaksDilationMasked, center, offset=offset)

        if cntPeak is None:
            break

        peakCenters.append((center[0]+offset[0], center[1]+offset[1]))
        peakContours.append(cntPeakOffset)
        if drawImg is not None:
            cv.drawContours(drawImg, [cntPeakOffset], 0, (255, 0, 0), -1)
        peaksDilationMasked = cv.drawContours(peaksDilationMasked, [cntPeak], 0, (0, 0, 0), -1)
    
    if drawImg is not None:
        for i, pc in enumerate(peakCenters):
            cv.circle(drawImg, pc, 1, (255,0,255), 1)
            drawInfo(drawImg, (pc[0]+15,pc[1]-15), str(i+1))

    return peaksDilation, peaksDilationMasked, peakCenters, peakContours

def findAllPeaks(gray, p, coverage=0, drawImg=None):
    
    maxIntensity = np.max(gray)
    
    grayNew = np.zeros(gray.shape, dtype=np.uint8)
    while np.max(grayMasked) > maxIntensity*coverage:
        intensity = np.max(grayMasked)
        lower = int(p*intensity)
        mask = cv.inRange(gray, lower, intensity)
        grayTemp = cv.bitwise_and(gray, gray, mask=mask)
        grayNew = cv.bitwise_or(grayNew, grayTemp)

    return grayNew

def meanShift(gray, center, kernel, maxIter=100, drawImg=None):
    assert kernel.shape[0] % 2 == 1, "Kernel size must be uneven"
    radius = int(kernel.shape[0]/2)
    
    startPatch = None
    for i in range(maxIter+1):
        startRow = center[1]-radius
        endRow = center[1]+radius+1
        startCol = center[0]-radius
        endCol = center[0]+radius+1
        patch = gray[startRow:endRow, startCol:endCol]

        #if mask.shape[0] == 0 or mask.shape[1] == 0:
        #    break
        patch = cv.bitwise_and(patch, patch, mask=kernel)
        if startPatch is None:
            startPatch = patch

        M = cv.moments(patch)
        if M["m00"] == 0:
            break
        cx = int(round(M["m10"] / M["m00"]))
        cy = int(round(M["m01"] / M["m00"]))

        #################
        #patchColor = cv.cvtColor(patch, cv.COLOR_GRAY2BGR)
        #cv.circle(patchColor, (cx, cy), 1, (255, 0, 0), 1)
        #cv.imshow("mean shift patch", patchColor)
        #cv.waitKey(0)
        #################

        centroid = cx-radius+center[0], cy-radius+center[1]
        #centroid = (int(centroid[0]), int(centroid[1]))
        centroid = max(min(centroid[0], gray.shape[1]-radius-1), radius), max(min(centroid[1], gray.shape[0]-radius-1), radius)

        if i == maxIter:
            if center == centroid:
                i -= 1 # max was found the previous iteration
                center = centroid
            break

        if tuple(centroid) == tuple(center):
            center = tuple(centroid)
            break

        if drawImg is not None:
            cv.line(drawImg, tuple(center), tuple(centroid), (0,0,255), 1)

        center = tuple(centroid)
        
        """
        n = 0
        centroidSum = np.array((0., 0.))

        for cx in range(-radius, radius+1):
            for cy in range(-radius, radius+1):
                if mask[cy+radius, cx+radius] > 0:
                    I = mask[cy+radius, cx+radius]
                    n += I
                    centroidSum += np.array((cx, cy))*I*lamb
        if n == 0:
            break
        centroid = centroidSum/float(n) + np.array(center)
        centroid = (round(centroid[0]), round(centroid[1]))
        centroid = (int(centroid[0]), int(centroid[1]))
        """
        
    if drawImg is not None:
        # plot crosshair
        color = (0,0,255)
        size = int(radius/5)
        cx, cy = center
        cv.line(drawImg, (cx - size, cy), (cx + size, cy), color, 1)
        cv.line(drawImg, (cx, cy - size), (cx, cy + size), color, 1)
        cv.circle(drawImg, center, radius, color, 1)

    return center, patch, i

def maxShift(gray, center, kernel, maxIter=100, drawImg=None):
    assert kernel.shape[0] % 2 == 1, "Kernel size must be uneven"
    radius = int(kernel.shape[0]/2)
    maxPrev = 0
    for i in range(maxIter+1):
        mask = gray[center[1]-radius:center[1]+radius+1, center[0]-radius:center[0]+radius+1]
        if mask.shape[0] != kernel.shape[0] or mask.shape[1] != kernel.shape[1]:
            # TODO: what happens with "i" here?
            break

        mask = cv.bitwise_and(mask, mask, mask=kernel)

        maxIntensity = np.max(mask)
        maxInd = np.unravel_index(np.argmax(mask), mask.shape)
        maxInd = (maxInd[0]-radius, maxInd[1]-radius)
        centroid = center[0] + maxInd[1], center[1] + maxInd[0]

        if i == maxIter:
            if tuple(centroid) == tuple(center) or maxIntensity == maxPrev:
                i -= 1 # max was found the previous iteration

            break

        if drawImg is not None:
            cv.line(drawImg, tuple(center), tuple(centroid), (0,0,255), 1)

        if gray[centroid[1], centroid[0]] == 0:
            break

        elif tuple(centroid) == tuple(center):
            i -= 1 # max was found the previous iteration
            center = centroid
            break

        elif maxIntensity == maxPrev:
            i -= 1
            break

        maxPrev = maxIntensity
        center = centroid

    if drawImg is not None:
        if gray[center[1], center[0]] == 0:
            # if max is 0
            cv.circle(drawImg, tuple(center), radius, (45, 255, 255), 1)
        elif i == maxIter:
            # Don't draw if maximum not reached
            pass
            #cv.circle(drawImg, tuple(center), radius, (255,0,0), 1)
        else:
            cv.circle(drawImg, tuple(center), radius, (0,255,0), 1)

    return center, mask, i

def pixelDist(center1, center2):
    return np.linalg.norm([center1[0]-center2[0], center1[1]-center2[1]])

def __featureAssociation(featurePoints, detectedPoints, featurePointsGuess=None, drawImg=None):
    """
    Assumes that the orientation of the feature model is approximately the identity matrix relative to the camera frame
    i.e. x and y values can be handled directly in the image plane
    """
    if len(detectedPoints) < len(featurePoints):
        print("Not enough features detected")
        return [], []

    if featurePointsGuess is None:
        centerx, centery  = np.mean(detectedPoints, axis=0)
        xys = np.array(detectedPoints)
        maxR = np.max(np.linalg.norm(np.array(detectedPoints) - np.array((centerx, centery)), axis=1))
        maxFR = np.max(np.linalg.norm(featurePoints[:, :2], axis=1)) # only x, y
        # a guess of where the features are based on the max radius of feature 
        # points and the max radius of the detected points
        featurePointsGuess = [(maxR*x/maxFR + centerx, maxR*y/maxFR + centery) for x,y,_ in featurePoints]
    else:
        centerx, centery  = np.mean(detectedPoints, axis=0)
        xys = np.array(detectedPoints)
        
    minDist = np.inf
    minDistPointIdx = None
    associatedPoints = []
    detectedPoints = [tuple(p) for p in detectedPoints]
    xys = [tuple(p) for p in xys]
    
    for i in range(len(featurePoints)):
        fx, fy = featurePointsGuess[i]
        for j in range(len(detectedPoints)):
            cx, cy = xys[j]
            d = np.sqrt( pow(cx-fx, 2) + pow(cy-fy, 2))
            if d < minDist:
                minDist = d
                minDistPointIdx = j

        #p = detectedPoints[minDistPointIdx]
        associatedPoints.append(detectedPoints[minDistPointIdx])
        del xys[minDistPointIdx]
        del detectedPoints[minDistPointIdx]
        
        minDist = np.inf
        minDistPointIdx = None
    
    associatedPoints = np.array(associatedPoints, dtype=np.float32)

    if drawImg is not None:
        for i in range(len(associatedPoints)):
            px = associatedPoints[i][0]
            py = associatedPoints[i][1]
            fpx = featurePointsGuess[i][0]
            fpy = featurePointsGuess[i][1]
            drawInfo(drawImg, (int(px), int(py)), str(i))
            drawInfo(drawImg, (int(fpx), int(fpy)), str(i), color=(0, 0, 255))
            cv.circle(drawImg, (int(px), int(py)), 2, (255, 0, 0), 3)

    return associatedPoints, featurePointsGuess


def featureAssociationWithGuess(detectedLightSources, featurePointGuess, associationRadius):
    """
    detectedLightSources - detected lightsources int the image
    featurePointsGuess - an initial guess of the light source positions in the image
    associationRadius - all candidates 
    """
    associatedLightSources = [[]*len(featurePointGuess)]

    for pointGuess, assLightSourceList in zip(featurePointGuess, associatedLightSources):
        for detLightSource in detectedLightSources:
            if np.linalg.norm((pointGuess - np.array(detLightSource.center))) < associationRadius:
                assLightSourceList.append(detLightSource)

    return associatedLightSources

def featureAssociationSquare(featurePoints, detectedLightSources, resolution, drawImg=None):
    """
    resolution - (h, w)
    """
    if len(detectedLightSources) != len(featurePoints):
        raise Exception("Detected points and feature points does not have the same length")

    assert len(featurePoints) > 3 and len(featurePoints) <= 5, "Can not perform square association with '{}' feature points".format(len(featurePoints))
    
    refIdxs = [None]*4 # [topLeftIdx, topRightIdx, bottomLeftIdx, bottomRightIdx]
    featurePointDists = [np.linalg.norm(fp[:2]) for fp in featurePoints]

    for i, (fp, d) in enumerate(zip(featurePoints, featurePointDists)):

        if fp[0] < 0 and fp[1] < 0:
            # top left
            refIdx = 0
        elif fp[0] > 0 and fp[1] < 0:
            # top right
            refIdx = 1
        elif fp[0] < 0 and fp[1] > 0:
            # bottom left
            refIdx = 2
        elif fp[0] > 0 and fp[1] > 0:
            # bottom right
            refIdx = 3
        else:
            continue

        if refIdxs[refIdx] is None:
            refIdxs[refIdx] = i
        else:
            if d > featurePointDists[refIdxs[refIdx]]:
                refIdxs[refIdx] = i

    for v in refIdxs:
        if v is None:
            print(refIdxs)
            raise Exception("Association failed")

    h, w = resolution
    tLeft = (0, 0)
    tRight = (w-1, 0)
    bLeft = (0, h-1)
    bRight = (w-1, h-1)

    associatedLightSources = [None]*len(featurePoints)
    detectedLightSources = list(detectedLightSources)

    for refIdx, refCorner in zip(refIdxs, [tLeft, tRight, bLeft, bRight]):
        for i, ls in enumerate(detectedLightSources):
            if associatedLightSources[refIdx] is None:
                associatedLightSources[refIdx] = ls
            else:
                lsOther = associatedLightSources[refIdx]

                dist = pixelDist(ls.center, refCorner)
                distOther = pixelDist(lsOther.center, refCorner)
                if dist < distOther:
                    associatedLightSources[refIdx] = ls

    try:
        notCornerIdx = associatedLightSources.index(None)
    except ValueError:
        pass
    else:
        for ls in detectedLightSources:
            if ls not in associatedLightSources:
                associatedLightSources[notCornerIdx] = ls
                break
        else:
            raise Exception("Somthing went wrong")
    
    return associatedLightSources, []

def featureAssociationSquareImproved(featurePoints, detectedLightSources, drawImg=None):
    """
    resolution - (h, w)
    """
    if len(detectedLightSources) != len(featurePoints):
        raise Exception("Detected points and feature points does not have the same length")

    assert len(featurePoints) > 3 and len(featurePoints) <= 5, "Can not perform square association with '{}' feature points".format(len(featurePoints))
    
    refIdxs = [None]*4 # [topLeftIdx, topRightIdx, bottomLeftIdx, bottomRightIdx]
    featurePointDists = [np.linalg.norm(fp[:2]) for fp in featurePoints]

    for i, (fp, d) in enumerate(zip(featurePoints, featurePointDists)):

        if fp[0] < 0 and fp[1] < 0:
            # top left
            refIdx = 0
        elif fp[0] > 0 and fp[1] < 0:
            # top right
            refIdx = 1
        elif fp[0] < 0 and fp[1] > 0:
            # bottom left
            refIdx = 2
        elif fp[0] > 0 and fp[1] > 0:
            # bottom right
            refIdx = 3
        else:
            continue

        if refIdxs[refIdx] is None:
            refIdxs[refIdx] = i
        else:
            if d > featurePointDists[refIdxs[refIdx]]:
                refIdxs[refIdx] = i

    for v in refIdxs:
        if v is None:
            print(refIdxs)
            raise Exception("Association failed")



    # identify top two and bottom two
    sortedByAscendingY = sorted(detectedLightSources, key=lambda ls: ls.center[1])
    topTwo = sortedByAscendingY[:2]
    bottomTwo = sortedByAscendingY[-2:]

    # sort by x to identify [left, right]
    topLeft, topRight = sorted(topTwo, key=lambda ls: ls.center[0])
    bottomLeft, bottomRight = sorted(bottomTwo, key=lambda ls: ls.center[0])

    associatedLightSources = [None]*len(featurePoints)
    #detectedLightSources = list(detectedLightSources)

    for refIdx, ls in zip(refIdxs, [topLeft, topRight, bottomLeft, bottomRight]):
        associatedLightSources[refIdx] = ls

    try:
        notCornerIdx = associatedLightSources.index(None)
    except ValueError:
        pass
    else:
        for ls in detectedLightSources:
            if ls not in associatedLightSources:
                associatedLightSources[notCornerIdx] = ls
                break
        else:
            raise Exception("Somthing went wrong")
    
    return associatedLightSources, []


def featureAssociation(featurePoints, detectedLightSources, featurePointsGuess=None, drawImg=None):
    """
    Assumes that the orientation of the feature model is 0 around every axis relative to the camera frame
    i.e. x and y values can be handled directly in the image plane
    """
    detectedLightSources = list(detectedLightSources)
    detectedPoints = [ls.center for ls in detectedLightSources]
    if len(detectedPoints) < len(featurePoints):
        print("Not enough features detected")
        return [], []

    if featurePointsGuess is None:
        centerx, centery  = np.mean(detectedPoints, axis=0)
        xys = np.array(detectedPoints)
        maxR = np.max(np.linalg.norm(np.array(detectedPoints) - np.array((centerx, centery)), axis=1))
        maxFR = np.max(np.linalg.norm(featurePoints[:, :2], axis=1)) # only x, y
        # a guess of where the features are based on the max radius of feature 
        # points and the max radius of the detected points
        featurePointsGuess = [(maxR*x/maxFR + centerx, maxR*y/maxFR + centery) for x,y,_ in featurePoints]
    else:
        centerx, centery  = np.mean(detectedPoints, axis=0)
        xys = np.array(detectedPoints)
        
    minDist = np.inf
    minDistPointIdx = None
    associatedLightSources = []
    detectedPoints = [tuple(p) for p in detectedPoints]
    xys = [tuple(p) for p in xys]
    
    for i in range(len(featurePoints)):
        fx, fy = featurePointsGuess[i]
        for j in range(len(detectedPoints)):
            cx, cy = xys[j]
            d = np.sqrt( pow(cx-fx, 2) + pow(cy-fy, 2))
            if d < minDist:
                minDist = d
                minDistPointIdx = j

        #p = detectedPoints[minDistPointIdx]
        associatedLightSources.append(detectedLightSources[minDistPointIdx])
        del xys[minDistPointIdx]
        del detectedPoints[minDistPointIdx]
        del detectedLightSources[minDistPointIdx]
        
        minDist = np.inf
        minDistPointIdx = None

    if drawImg is not None:
        for i in range(len(associatedLightSources)):
            px = associatedLightSources[i].center[0]
            py = associatedLightSources[i].center[1]
            fpx = featurePointsGuess[i][0]
            fpy = featurePointsGuess[i][1]
            drawInfo(drawImg, (int(px), int(py)), str(i))
            drawInfo(drawImg, (int(fpx), int(fpy)), str(i), color=(0, 0, 255))
            cv.circle(drawImg, (int(px), int(py)), 2, (255, 0, 0), 3)

    return associatedLightSources, featurePointsGuess

class LightSource:
    def __init__(self, cnt, intensity):

        self.cnt = cnt
        self.center = contourCentroid(cnt)
        self.area = cv.contourArea(cnt)
        self.ratio = contourRatio(cnt)
        self.circleCenter, self.radius = cv.minEnclosingCircle(cnt)

        self.intensity = intensity

        self.rmseUncertainty = self.radius

    def circleExtent(self):
        return contourRatio(self.cnt)

class LightSourceTracker:
    def __init__(self, center, intensity, radius, maxPatchRadius, minPatchRadius, p=.975):
        self.intensity = intensity
        self.minIntensity = 20
        self.center = center
        self.radius = radius
        self.maxPatchRadius = maxPatchRadius
        self.minPatchRadius = minPatchRadius
        
        self.patchScale = 1.5
        self.patchRadius = int(round(self.patchScale*self.radius))
        
        self.p = p

        self.cnt = None

        self.ksize = 11
        self.locMaxKernel = circularKernel(self.ksize)

    def _limitCenter(self, gray, center, radius):
        x = max(min(center[0], gray.shape[1]-radius-1), radius)
        y = max(min(center[1], gray.shape[0]-radius-1), radius)
        return (x, y)

    def __peakSelect(self, gray, peakImgPatch, center, patchRadius, drawImg=None):
        peakImgPatch = peakImgPatch.copy()
        peakContours = []
        peakIntensities = []
        peakPositions = []
        ###
        maxIntensity = np.max(peakImgPatch)
        # filter 
        #alpha = 1 # only intensitie within alpha % is considered
        #threshold = min(maxIntensity, self.intensity)
        #thresholdLower = max(int(threshold*(1-alpha)), 0) # intensity can only change by 10%
        #thresholdUpper = min(int(threshold*(1+alpha)), 255)
        #print("lower", thresholdLower)
        #print("upper", thresholdUpper)
        #peakImgPatch = cv.bitwise_and(peakImgPatch, peakImgPatch, mask=cv.inRange(peakImgPatch, thresholdLower, thresholdUpper))
        #cv.imshow("thresholded peak img patch", peakImgPatch)

        threshold = int(self.p*maxIntensity)
        _, threshImgPatch = cv.threshold(peakImgPatch, threshold, 256, cv.THRESH_BINARY)

        while np.max(peakImgPatch) > 0 and np.max(peakImgPatch) == maxIntensity:
            y, x = np.unravel_index(np.argmax(peakImgPatch), peakImgPatch.shape)
            
            peakCntPatch, peakCnt = findPeakContourAt(threshImgPatch, (x,y), offset=(center[0]-patchRadius, center[1]-patchRadius))
            x += center[0]-patchRadius
            y += center[1]-patchRadius

            #peakCntPatch = peakCnt - np.array((center[0]-patchRadius, center[1]-patchRadius)) # if gray is gray, hehe
            peakImgPatch = cv.drawContours(peakImgPatch, [peakCntPatch], 0, (0), -1)
            #peakCnt += np.array((center[0]-patchRadius, center[1]-patchRadius)) # if gray is a patch
            for (cx,cy) in peakPositions:
                if withinContour(cx, cy, peakCnt):
                    break
            else:        
                peakPositions.append((x,y))
                peakContours.append(peakCnt)
                peakIntensities.append(gray[y, x])


        if drawImg is not None:
            for cnt in peakContours:
                cv.drawContours(drawImg, [cnt], 0, (0,0,255), 1)  

        if not peakContours:
            return None

        thePeakCnt = None
        thePeakPos = None
        #thePeakDistToCenter = None
        thePeakAreaDiff = None
        thePeakInt = None
        # choose the one with similar area
        prevArea = self.radius*self.radius*np.pi
        for peakPos, peakCnt, peakInt in zip(peakPositions, peakContours, peakIntensities):
            peakAreaDiff = abs(cv.contourArea(peakCnt) - prevArea)
            peakCentroid = contourCentroid(peakCnt)
            #peakDistToCenter = np.linalg.norm((peakCentroid[0]-center[0], peakCentroid[1]-center[1]))

            #if thePeakCnt is None or peakDistToCenter < thePeakDistToCenter:
            if thePeakCnt is None or peakAreaDiff < thePeakAreaDiff: 
                thePeakCnt = peakCnt
                thePeakPos = peakPos
                #thePeakDistToCenter = peakDistToCenter
                thePeakAreaDiff = peakAreaDiff
                thePeakInt = gray[peakPos[1], peakPos[0]]


        # filter by radius or area?
        #peakContoursAreaFiltered = []
        #for cnt in peakContours:
        #    rad = cv.minEnclosingCircle(cnt)[1]
        #    if self.radius *< rad
        return thePeakInt, thePeakCnt

    def _peakSelect(self, gray, peakImgPatch, center, patchRadius, drawImg=None):
        # TODO
        peakImgPatch = cv.GaussianBlur(peakImgPatch, (self.ksize,self.ksize), 0)
        (peakDilationImg, 
        peaksDilationMasked, 
        peakCenters, 
        peakContours,
        iterations) = findNPeaks2(peakImgPatch, 
                                  kernel=self.locMaxKernel, 
                                  pMin=self.p,
                                  pMax=self.p, 
                                  n=5,
                                  minThresh=self.intensity*0.7,
                                  # maybe this should be (self.kernelSize-1)/2 instead of peakMargin?
                                  # or maybe it will remove the true candidates in case of weird shapes from "non-peaks"?
                                  margin=0,
                                  ignorePAtMax=True,
                                  offset=(center[0]-patchRadius, center[1]-patchRadius),
                                  maxIter=5,
                                  drawImg=drawImg,
                                  drawInvalidPeaks=True)

        if len(peakCenters) > 0:
            peak = peakCenters[0]
            thePeakCnt = peakContours[0]
            thePeakInt = gray[peak[1], peak[0]]
        else:
            return None
        
        return thePeakInt, thePeakCnt

    def getLightSource(self):
        return LightSource(self.cnt, self.intensity)

    def update(self, gray, drawImg=None):
        if drawImg is not None:
            # plot the kernel used to find the light source
            cv.circle(drawImg, self.center, self.patchRadius, (255,0,255), 1)

        # here we add the kernel radius (size/2) so that the peak dilation do not detect false peaks at the edges 
        patchRadius = self.patchRadius
        patchKernelSize = patchRadius*2+1
        patchKernel = circularKernel(patchKernelSize)
        center = self.center
        patch = gray[center[1]-patchRadius:center[1]+patchRadius+1, center[0]-patchRadius:center[0]+patchRadius+1]

        if patch.shape[0] != patchKernel.shape[0] or patch.shape[1] != patchKernel.shape[1]:
            # TODO: what happens with "i" here?
            print("not right")
            return False

        # TODO: Masking the path with a circular kernel, edges might be accepted by findPeaks 
        #patch = cv.bitwise_and(patch, patch, mask=patchKernel)
        #cv.imshow("patch", patch)
        #cv.waitKey(1)

        ret = self._peakSelect(gray, patch, center, patchRadius, drawImg=drawImg)

        if ret is None:
            #self.intensity = 255
            print("No peak detecteddddddddddd")
            return False

        peakIntensity, peakCnt = ret

        newCenter = contourCentroid(peakCnt)
        #newCenter = int(round(newCenter[0])), int(round(newCenter[1]))
        newRadius = cv.minEnclosingCircle(peakCnt)[1]
        newRadius = int(round(newRadius))
        newRadius = max(1, newRadius)

        self.center = newCenter
        self.intensity = max(peakIntensity, self.minIntensity)
        self.radius = newRadius
        #self.patchRadius = int(round(self.patchScale*self.radius))
        #self.patchRadius = min(self.maxPatchRadius, max(self.minPatchRadius, self.patchRadius))
        self.patchRadius = self.radius + self.minPatchRadius
        self.center = self._limitCenter(gray, self.center, self.patchRadius)
        self.cnt = peakCnt

        if drawImg is not None:
            # plot the detected light source contour and the enclosing circle
            cv.drawContours(drawImg, [peakCnt], 0, (0,255,0), 1)
            cv.circle(drawImg, self.center, self.patchRadius, (255,0,0), 1)
            cv.circle(drawImg, self.center, 1, (255,0,0), 3)

        return True


class LightSourceTracker2:
    def __init__(self, center, intensity, radius, maxPatchRadius, minPatchRadius, p=.97):
        self.intensity = intensity
        self.minIntensity = 20
        self.center = center
        self.radius = radius
        self.maxPatchRadius = maxPatchRadius
        self.minPatchRadius = minPatchRadius
        
        self.patchScale = 1.5
        self.patchRadius = int(round(self.patchScale*self.radius))
        
        self.p = p

        self.cnt = None

    def _limitCenter(self, gray, center, radius):
        x = max(min(center[0], gray.shape[1]-radius-1), radius)
        y = max(min(center[1], gray.shape[0]-radius-1), radius)
        return (x, y)

    def _peakSelect(self, localMaxImg, peakImgPatch, center, patchRadius, drawImg=None):
        peakImgPatch = peakImgPatch.copy()
        peakContours = []
        peakIntensities = []
        peakPositions = []
        ###
        maxIntensity = np.max(peakImgPatch)

        threshold = int(self.p*maxIntensity)
        _, threshImgPatch = cv.threshold(peakImgPatch, threshold, 256, cv.THRESH_BINARY)

        while np.max(peakImgPatch) > 0 and np.max(peakImgPatch) == maxIntensity:
            y, x = np.unravel_index(np.argmax(peakImgPatch), peakImgPatch.shape)
            
            peakCntPatch, peakCnt = findPeakContourAt(threshImgPatch, (x,y), offset=(center[0]-patchRadius, center[1]-patchRadius))
            x += center[0]-patchRadius
            y += center[1]-patchRadius

            #peakCntPatch = peakCnt - np.array((center[0]-patchRadius, center[1]-patchRadius)) # if gray is gray, hehe
            peakImgPatch = cv.drawContours(peakImgPatch, [peakCntPatch], 0, (0), -1)
            #peakCnt += np.array((center[0]-patchRadius, center[1]-patchRadius)) # if gray is a patch
            for (cx,cy) in peakPositions:
                if withinContour(cx, cy, peakCnt):
                    break
            else:        
                peakPositions.append((x,y))
                peakContours.append(peakCnt)
                peakIntensities.append(gray[y, x])


        if drawImg is not None:
            for cnt in peakContours:
                cv.drawContours(drawImg, [cnt], 0, (0,0,255), 1)  

        if not peakContours:
            return None

        thePeakCnt = None
        thePeakPos = None
        #thePeakDistToCenter = None
        thePeakAreaDiff = None
        thePeakInt = None
        # choose the one with similar area
        prevArea = self.radius*self.radius*np.pi
        for peakPos, peakCnt, peakInt in zip(peakPositions, peakContours, peakIntensities):
            peakAreaDiff = abs(cv.contourArea(peakCnt) - prevArea)
            peakCentroid = contourCentroid(peakCnt)
            #peakDistToCenter = np.linalg.norm((peakCentroid[0]-center[0], peakCentroid[1]-center[1]))

            #if thePeakCnt is None or peakDistToCenter < thePeakDistToCenter:
            if thePeakCnt is None or peakAreaDiff < thePeakAreaDiff: 
                thePeakCnt = peakCnt
                thePeakPos = peakPos
                #thePeakDistToCenter = peakDistToCenter
                thePeakAreaDiff = peakAreaDiff
                thePeakInt = gray[peakPos[1], peakPos[0]]


        # filter by radius or area?
        #peakContoursAreaFiltered = []
        #for cnt in peakContours:
        #    rad = cv.minEnclosingCircle(cnt)[1]
        #    if self.radius *< rad
        return thePeakInt, thePeakCnt

    def getLightSource(self):
        return LightSource(self.cnt, self.intensity)

    def update(self, gray, localMaxImage, drawImg=None):
        if drawImg is not None:
            # plot the kernel used to find the light source
            cv.circle(drawImg, self.center, self.patchRadius, (255,0,255), 1)

        # here we add the kernel radius (size/2) so that the peak dilation do not detect false peaks at the edges 
        patchRadius = self.patchRadius
        patchKernelSize = patchRadius*2+1
        patchKernel = circularKernel(patchKernelSize)
        center = self.center
        patch = gray[center[1]-patchRadius:center[1]+patchRadius+1, center[0]-patchRadius:center[0]+patchRadius+1]

        if patch.shape[0] != patchKernel.shape[0] or patch.shape[1] != patchKernel.shape[1]:
            # TODO: what happens with "i" here?
            print("not right")
            return False

        patch = cv.bitwise_and(patch, patch, mask=patchKernel)
        #cv.imshow("patch", patch)

        ret = self._peakSelect(gray, patch, center, patchRadius, drawImg=drawImg)

        if ret is None:
            #self.intensity = 255
            print("No peak detecteddddddddddd")
            return False

        peakIntensity, peakCnt = ret

        newCenter = contourCentroid(peakCnt)
        #newCenter = int(round(newCenter[0])), int(round(newCenter[1]))
        newRadius = cv.minEnclosingCircle(peakCnt)[1]
        newRadius = int(round(newRadius))
        newRadius = max(1, newRadius)

        self.center = newCenter
        self.intensity = max(peakIntensity, self.minIntensity)
        self.radius = newRadius
        self.patchRadius = int(round(self.patchScale*self.radius))
        self.patchRadius = min(self.maxPatchRadius, max(self.minPatchRadius, self.patchRadius))
        self.center = self._limitCenter(gray, self.center, self.patchRadius)
        self.cnt = peakCnt

        if drawImg is not None:
            # plot the detected light source contour and the enclosing circle
            cv.drawContours(drawImg, [peakCnt], 0, (0,255,0), 1)
            cv.circle(drawImg, self.center, self.patchRadius, (255,0,0), 1)
            cv.circle(drawImg, self.center, 1, (255,0,0), 3)

        return True

class LightSourceTrackInitializer:
    def __init__(self, radius=10, maxPatchRadius=50, minPatchRadius=7, p=0.975, maxIntensityChange=0.7, maxMovement=20):
        self.radius = radius
        self.maxPatchRadius = maxPatchRadius
        self.minPatchRadius = minPatchRadius
        self.p = p

        self.maxMovement = maxMovement
        self.maxIntensityChange = maxIntensityChange

        self.trackers = []
        self.iterations = 0

    def getMaskedImg(self, gray):
        grayMasked = gray.copy()
        for tr in self.trackers:
            cv.circle(grayMasked, tr.center, tr.patchRadius, (0,0,0), -1)
        
        return grayMasked

    def _sortLightSources(self, candidates, n):
        candidates.sort(key=lambda p: (p.intensity, p.area), reverse=True)
        candidates = candidates[:n]
        
        return candidates

    def getCandidates(self, n=None):
        if n is None:
            n = len(self.trackers)
        return self._sortLightSources([tr.getLightSource() for tr in self.trackers], n)

    def reset(self):
        self.trackers = []
        self.iterations = 0

    def update(self, gray, newTrackerCenters=None, drawImg=None):
        self.iterations += 1

        if newTrackerCenters is not None:
            for center in newTrackerCenters:
                self.trackers.append(LightSourceTracker(center,
                                                        intensity=gray[center[1], center[0]], 
                                                        radius=self.radius, 
                                                        maxPatchRadius=self.maxPatchRadius, 
                                                        minPatchRadius=self.minPatchRadius,
                                                        p=self.p))

        for tr in list(self.trackers):
            center = tr.center
            intensity = tr.intensity
            success = tr.update(gray, drawImg=drawImg)
            
            if success:
                newCenter = tr.center
                newIntensity = tr.intensity

                if np.linalg.norm([center[0]-newCenter[0], center[1]-newCenter[1]]) > self.maxMovement:
                    print("Movement to large")
                    self.trackers.remove(tr)

                elif newIntensity < intensity*self.maxIntensityChange:
                    print("Intensity change to large")
                    self.trackers.remove(tr)

            else:
                # TODO: remove tracker here?
                self.trackers.remove(tr)

        for tr1, tr2 in itertools.combinations(list(self.trackers), 2):
            if tr1.center == tr2.center:
                print("Trackers same place, removing one")
                try:
                    self.trackers.remove(tr2)
                except:
                    pass

class MeanShiftTracker:
    def __init__(self, center, radius, maxRadius, minRadius):
        self.center = center
        self.radius = radius
        self.maxRadius = maxRadius
        self.minRadius = minRadius
        #self.kernel = circularKernel(int(radius*2)+1)
        #self.lost = True

    @property
    def x(self):
        return self.center[0]

    @property
    def y(self):
        return self.center[1]

    def _peakSelect(self, gray, patch, center, radius, drawImg=None):
        maskedGray = np.zeros(gray.shape, dtype=np.uint8)
        maskedGray[center[1]-radius:center[1]+radius+1, center[0]-radius:center[0]+radius+1] = patch

        # not really centers, just a position within the "peak"
        p = 0.99
        peakCenters, _ = findMaxPeaks(maskedGray, p=p, drawImg=None)
        if not peakCenters:
            return None
        # peak contours might be "cut of" by the mask, find the full contours using the gray img
        # choose the contour based on max area
        contours = [findPeakContourAt(gray, pc, p=p) for pc in peakCenters]

        contours.sort(key=cv.contourArea, reverse=True)
        if drawImg is not None:
            #print("len cnts", len(contours))
            cv.drawContours(drawImg, contours, -1, (255,0,0), 1)

        return contours[0]

    def _limitCenter(self, gray, center, radius):
        x = max(min(center[0], gray.shape[1]-radius-1), radius)
        y = max(min(center[1], gray.shape[0]-radius-1), radius)
        return (x, y)

    def update(self, gray, maxIter=10, drawImg=None):
        if drawImg is not None:
            cv.circle(drawImg, self.center, self.radius, (50,50,50), 1)

        kernel = circularKernel(self.radius*2+1)

        x, y = self._limitCenter(gray, self.center, self.radius)# maybe not needed?

        centerNew, endPatch, i = meanShift(gray, (x, y), kernel, maxIter=maxIter, drawImg=drawImg)
        #centerNew, endPatch, i = maxShift(gray, (x, y), kernel, maxIter=maxIter, drawImg=drawImg)
        #centerNew = self._limitCenter(gray, centerNew, self.radius) # maybe not needed?

        cnt = self._peakSelect(gray, endPatch, centerNew, self.radius, drawImg=drawImg)
        if cnt is None:
            # Failed
            return self.center, self.radius, 0, endPatch

        _, radiusNew = cv.minEnclosingCircle(cnt)
        centerNew = contourCentroid(cnt)

        self.lost = False
       
        self.center = centerNew

        scale = 1
        radiusNew = int(radiusNew*scale)
        self.radius = radiusNew
        self.radius = min(self.radius, self.maxRadius)
        self.radius = max(self.radius, self.minRadius)
        
        """
        radiusRatio = np.inf
        if self.radius > 0:
            radiusRatio = float(radiusNew)/self.radius

        
        if self.lost:
            radius = min(radiusNew, self.maxRadius)
            self.radius = max(radius, self.minRadius)
            self.lost = False
            self.center = centerNew
        elif 0.9 < radiusRatio < 1.1:
            radius = min(radiusNew, self.maxRadius)
            self.radius = max(radius, self.minRadius)
            self.lost = False
            self.center = centerNew
        else:
            #x, y = xNew, yNew
            self.radius = self.radius
            self.lost = True
        """
        if drawImg is not None:
            cv.circle(drawImg, self.center, self.radius, (255,0,255), 1)
            cv.circle(drawImg, self.center, 1, (255,0,255), 1)

        return self.center, self.radius, scale, endPatch

class ThresholdFeatureExtractor:
    # Check this for contour properties
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_properties/py_contour_properties.html
    def __init__(self, featureModel, camera, p=.02, erosionKernelSize=5, maxIter=10, useKernel=True):
        if isinstance(featureModel, int):
            self.featureModel = None
            self.nFeatures = featureModel
        else:
            self.featureModel = featureModel
            self.nFeatures = len(self.featureModel.features)
        self.camera = camera
        
        #self.pHold = AdaptiveThreshold(self.nFeatures, 250, 1, 150)#PercentageThreshold(p)
        self.pHold = PercentageThreshold(p)
        #self.pHold = AdaptiveThreshold(self.nFeatures, 255, 5, 150)#PercentageThreshold(p)
        self.adaOpen = AdaptiveOpen(self.nFeatures, kernelSize=erosionKernelSize, maxIter=maxIter)
        if useKernel:
            self.adaOpen = AdaptiveErodeKernel(self.nFeatures, kernelSize=erosionKernelSize, maxIter=maxIter)
    
    #def regionOfInterest(self, img, estTranslationVector, estRotationVector, wMargin, hMargin):
    def regionOfInterest(self, featurePointsGuess, wMargin, hMargin):
        #featurePointsGuess = self.camera.metersToUV(featurePointsGuess)
        topMost = [None, np.inf]
        rightMost = [-np.inf]
        bottomMost = [None, -np.inf]
        leftMost = [np.inf]
        for p in featurePointsGuess:
            if p[1] < topMost[1]:
                topMost = p

            if p[0] > rightMost[0]:
                rightMost = p

            if p[1] > bottomMost[1]:
                bottomMost = p

            if p[0] < leftMost[0]:
                leftMost = p
        
        topMost[1] -= hMargin
        rightMost[0] += wMargin
        bottomMost[1] += hMargin
        leftMost[0] -= wMargin
        cnt = np.array([topMost, rightMost, bottomMost, leftMost], dtype=np.int32)
        x, y, w, h = cv.boundingRect(cnt)
        roiCnt = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
        #img = cv.rectangle(img, (x-wMargin,y-hMargin), (x+w+wMargin, y+h+hMargin), (255, 255, 255), 10, cv.LINE_AA)

        return roiCnt

    def __call__(self, gray, drawImg=None):
        img = gray.copy()

        #img = cv.GaussianBlur(img,(5,5),0)
        
        # https://docs.opencv.org/4.5.3/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
        #blur = cv.bilateralFilter(img,5,200,50) # smoothing but, keeps edges. Makes the adaOpen faster
        #img = blur
        img = self.pHold.process(img)
        #img = fillContoursOnEdges(img) # All contours connected to the edges are background

        _, contours, hier = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        if drawImg is not None:
            cv.drawContours(drawImg, contours, -1, (0, 0, 255), 3)

        contours.sort(key=contourRatio, reverse=True)
        contours.sort(key=contourPositionY, reverse=True)
        #contours = contourAreaOutliers(contours) # dont want to remove outliers here, remove noise first

        img = self.adaOpen.process(img) # removes noise
        
        _, contoursNew, hier = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        if drawImg is not None: cv.drawContours(drawImg, contoursNew, -1, (0, 255, 0), 3)

        # error: The truth value of an array with more than one element is ambiguous
        #contoursNew = np.array(contoursNew)
        #ds = contourAreaDistanceFromMean(contoursNew)
        #print(ds.shape)
        #print(contoursNew.shape)
        #contoursNew = [cnt for d, cnt in sorted(zip(ds, contoursNew))]
        contoursNew.sort(key=contourRatio, reverse=True)
        contoursNew.sort(key=contourPositionY)
        #contoursNew = contoursMinAreaRemove(contoursNew, minArea=7)
        #contoursNew = contoursNew[:self.nFeatures]

        # contours are sorted by ratio but using the initial shape, not the eroded one
        # conuld use the "opened" ratio aswell?
        points = []
        for cntOld in contours:
            #if len(points) == self.nFeatures: # this removes potential extra contours
            #    break
            for cntNew in contoursNew:
                # Check if new contour is inside old
                area = cv.contourArea(cntNew)
                if area == 0:
                    cx, cy = cntNew[0][0]
                else:
                    M = cv.moments(cntNew)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])

                result = cv.pointPolygonTest(cntOld, (cx,cy), False) 
                if result in (1, 0): # inside or on the contour
                    points.append((cx, cy))
                    ratio = contourRatio(cntNew)
                    if drawImg is not None:
                        cv.drawContours(drawImg, [cntNew], 0, (255, 0, 0), -1) # or maybe draw new??
                        drawInfo(drawImg, (cx+10, cy-10), str(ratio), color=(255, 0, 255))
                    #break We don't break here, the old contour might be two overlapping lights

        #points = points[:self.nFeatures]
        if len(points) == 0:
            return img, []

        return img, points

class ThresholdFeatureExtractorWithInitialization:
    def __init__(self, featureModel, camera, p, erosionKernelSize, maxIter, useKernel, nInitImages):
        self.nInitImages = nInitImages
        self.associatedPointsList = [] # only the nInitImages latest associated points are used
        self.img = None

        
        self._thresholdFeatureExt = ThresholdFeatureExtractor(featureModel=featureModel, 
                                                              camera=camera, 
                                                              p=p, 
                                                              erosionKernelSize=erosionKernelSize, 
                                                              maxIter=maxIter, 
                                                              useKernel=useKernel)
        #self._p = self._thresholdFeatureExt.pHold.p

    @property
    def adaOpen(self):
        return self._thresholdFeatureExt.adaOpen

    @property
    def pHold(self):
        return self._thresholdFeatureExt.pHold

    def __call__(self, gray, drawImg=None):
        #self.img = imgColor.copy()
        self.img = gray.copy()
        res, assPoints = self._thresholdFeatureExt(gray, drawImg=drawImg)
        self.associatedPointsList.insert(0, assPoints)
        self.associatedPointsList = self.associatedPointsList[:self.nInitImages]

        colors = [(255, 0, 0),
                  (0, 255, 0),
                  (0, 0, 255),
                  (255, 0, 255),
                  (255, 255, 0)]

        for assPoints in self.associatedPointsList:
            for color, p in zip(colors, assPoints):
                plotPoints(self.img, [p], color=color)

        self.img = self.adaOpen.img
        return res, self.associatedPointsList[0]

class GradientFeatureExtractor:
    def __init__(self, camera, nFeatures, kernelSize):

        self.nFeatures = nFeatures
        self.camera = camera
        self.kernelSize = kernelSize
        self.pHold = AdaptiveThreshold2(5)
        #self.adaOpen = AdaptiveOpen(len(self.featureModel.features), kernelSize=3, startIterations=1, maxIter=3)

    def __call__(self, gray, estTranslationVec=None, estRotationVec=None):
        res = gray
        ksize = self.kernelSize
        grad_x = cv.Sobel(res, cv.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv.Sobel(res, cv.CV_64F, 0, 1, ksize=ksize)

        abs_grad_x = cv.convertScaleAbs(grad_x, alpha=255/grad_x.max())
        abs_grad_y = cv.convertScaleAbs(grad_y, alpha=255/grad_y.max())
        
        res = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)



        #res = cv.GaussianBlur(res, (3,3), 0)
        #res = cv.blur(res, (5,5))

        #res = self.pHold.process(res)
        #res = self.adaOpen.process(res)
        #res = fillContours(res, 0.7)

        return res

class AdaptiveThreshold2:
    def __init__(self, nFeatures, marginPercentage=0, minArea=1, minRatio=0, thresholdType=cv.THRESH_BINARY):
        """
        try cv.THRESH_OTSU?
        """
        self.nFeatures = nFeatures
        self.threshold = 256
        self.thresInc = 1
        self.thresMin = 0
        self.marginPercentage = marginPercentage # 0-1
        self.minArea = minArea
        self.minRatio = minRatio
        self.thresholdType = thresholdType 
        self.img = np.zeros((10, 10)) # store the last processed image

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    def _calcCandidates(self, contours):
        candidates = []
        for cnt in contours:
            #if cv.contourArea(cnt)+1 > self.minArea and contourRatio(cnt) > self.minRatio:
            if cv.contourArea(cnt) > self.minArea:
                if contourRatio(cnt) > self.minRatio:
                    candidates.append(cnt)

        return candidates

    def _sortLightSources(self, candidates, maxAdditionalCandidates):
        candidates.sort(key=lambda p: p.area, reverse=True)
        candidates = candidates[:self.nFeatures+maxAdditionalCandidates]
        
        return candidates

    def process(self, gray, maxAdditionalCandidates=0, estDSPose=None, roiMargin=None, drawImg=None):

        roiCnt = None
        if estDSPose:
            # PREVIOUS
            #featurePointsGuess = estDSPose.reProject()
            #roiMargin += max([ls.radius for ls in estDSPose.associatedLightSources])
            #(x, y, w, h), roiCnt = regionOfInterest(featurePointsGuess, wMargin=roiMargin, hMargin=roiMargin)
            #roiMask = np.zeros(gray.shape, dtype=np.uint8)
            #cv.drawContours(roiMask, [roiCnt], 0, (255,255,255), -1)
            #gray = cv.bitwise_and(gray, gray, mask=roiMask)

            # NEW
            featurePointsGuess = estDSPose.reProject()
            roiMargin += max([ls.radius for ls in estDSPose.associatedLightSources])
            (x, y, w, h), roiCnt = regionOfInterest(featurePointsGuess, wMargin=roiMargin, hMargin=roiMargin)
            x = max(0, x)
            x = min(gray.shape[1]-1, x)
            y = max(0, y)
            y = min(gray.shape[0]-1, y)

            offset = (x, y)
            gray = gray[y:y+h, x:x+w]

        upper = 256

        img = cv.GaussianBlur(gray.copy(), (3,3),0)
        self.threshold = np.max(img)

        ret, imgTemp = cv.threshold(img, self.threshold, upper, self.thresholdType)
        _, contours, hier = cv.findContours(imgTemp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        candidates = self._calcCandidates(contours)
        
        img = img.copy()
        i = 0
        while True:
            i += 1
            if self.threshold < self.thresMin+self.thresInc:
                break

            if len(candidates) >= self.nFeatures:
                break

            self.threshold -= self.thresInc

            ret, imgTemp = cv.threshold(img, self.threshold, upper, self.thresholdType)

            _, contours, hier = cv.findContours(imgTemp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            candidates = self._calcCandidates(contours)

        print("HATS iterations: {}".format(i))
        print("Threshold: {}".format(self.threshold))

        if self.marginPercentage > 0:
            self.threshold = (1-self.marginPercentage)*self.threshold
            ret, imgTemp = cv.threshold(img, self.threshold, upper, self.thresholdType)
            _, contours, hier = cv.findContours(imgTemp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            candidates = self._calcCandidates(contours)

        self.img = imgTemp

        candidates = [LightSource(cnt+offset, self.threshold) for cnt in candidates]
        candidates = self._sortLightSources(candidates, maxAdditionalCandidates)

        if drawImg is not None:
            for ls in candidates:
                cnt = ls.cnt
                cx, cy = contourCentroid(cnt)
                r = contourRatio(cnt)
                drawInfo(drawImg, (cx+15,cy-15), str(r))
                cv.drawContours(drawImg, [cnt], 0, (255,0,0), -1)
        
        return self.img, candidates, roiCnt

class ModifiedHATS:
    MODE_SIMPLE = "simple"
    MODE_PEAK = "peak"
    MODE_VALLEY = "valley"
    
    def __init__(self, nFeatures, peakMargin=0, minArea=1, minRatio=0, maxIntensityChange=0.7, thresholdType=cv.THRESH_BINARY, blurKernelSize=5, mode="valley", ignorePeakAtMax=False, showHistogram=False):
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
        self.thresholdType = thresholdType

        assert mode in (self.MODE_SIMPLE, self.MODE_PEAK, self.MODE_VALLEY), "Invalid mode '', must be '{}, {}' or '{}'".format(mode, self.MODE_SIMPLE, self.MODE_PEAK, self.MODE_VALLEY)
        self.mode = mode
        self.ignorePeakAtMax = ignorePeakAtMax
        self.showHistogram = showHistogram

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

        # TODO: seems reasonable to have
        img = cv.morphologyEx(img, cv.MORPH_OPEN, self.morphKernel, iterations=1)

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


class AdaptiveThresholdPeak:
    def __init__(self, nFeatures, kernelSize, pMin, pMax, maxIntensityChange, minArea, minCircleExtent, blurKernelSize, ignorePAtMax, maxIter):
        """
        try cv.THRESH_OTSU?
        """
        self.nFeatures = nFeatures
        self.kernelSize = kernelSize
        self.kernel = circularKernel(self.kernelSize)
        self.pMin = pMin
        self.pMax = pMax
        self.img = np.zeros((10, 10), dtype=np.uint8) # store the last processed image
        self.maxIntensityChange = maxIntensityChange

        self.minArea = minArea
        self.minCircleExtent = minCircleExtent

        self.blurKernelSize = blurKernelSize

        self.ignorePAtMax = ignorePAtMax

        self.maxIter = maxIter
        # keep track of number of peak iterations
        self.iterations = 0

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

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
            x = max(0, x)
            x = min(gray.shape[1]-1, x)
            y = max(0, y)
            y = min(gray.shape[0]-1, y)

            offset = (x, y)
            gray = gray[y:y+h, x:x+w]

            minIntensity = min([ls.intensity for ls in estDSPose.associatedLightSources])
            minIntensity = self.maxIntensityChange*minIntensity

            # TODO: not sure if this should be done some how. It can for sure make it worse,
            # but maybe it can be better in some cases 
            #_, gray = cv.threshold(gray, minIntensity, 256, cv.THRESH_TOZERO)

            peakMargin = max([ls.radius for ls in estDSPose.associatedLightSources])

        
        # blurring seems to help for large resolution 
        # test_sessions/171121_straight_test.MP4
        if self.blurKernelSize > 0:
            gray = cv.GaussianBlur(gray, (self.blurKernelSize,self.blurKernelSize), 0)
            
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
                                  drawImg=drawImg,
                                  drawInvalidPeaks=True)

        """
        #TODO: possibly more computationally efficient to use local peaks on the non blured image
        (peakDilationImg, 
        peaksDilationMasked, 
        peakCenters, 
        peakContours,
        iterations) = findNPeaksWithBlur(gray, 
                                  kernel=self.kernel, 
                                  pMin=self.pMin,
                                  pMax=self.pMax, 
                                  n=self.nFeatures+maxAdditionalCandidates,
                                  minThresh=minIntensity,
                                  blurKernelSize=self.blurKernelSize,
                                  # maybe this should be (self.kernelSize-1)/2 instead of peakMargin?
                                  # or maybe it will remove the true candidates in case of weird shapes from "non-peaks"?
                                  margin=peakMargin,
                                  offset=offset,
                                  maxIter=self.maxIter,
                                  drawImg=drawImg,
                                  drawInvalidPeaks=True)
        """

        # TODO: check if offset is correct
        candidates = [LightSource(cnt, gray[pc[1]-offset[1], pc[0]-offset[0]]) for pc, cnt in zip(peakCenters, peakContours)]

        # TODO: This should probably be done
        candidatesNew = []
        removedCandidates = []
        for ls in candidates:
            if ls.area > self.minArea and ls.circleExtent() < self.minCircleExtent:
                removedCandidates.append(ls)
            else:
                candidatesNew.append(ls)
        candidates = candidatesNew
        

        candidates = self._sortLightSources(candidates, maxAdditionalCandidates)
        if drawImg is not None:
            for i, ls in enumerate(candidates):
                #cv.circle(drawImg, ls.center, int(round(ls.radius)), (255,0,255), 2)
                cv.circle(drawImg, ls.center, 2, (255,0,0), 2)
        
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

class LocalMaxHATS:
    def __init__(self, nFeatures, kernelSize, p, maxIntensityChange, showHistogram=False):
        """
        try cv.THRESH_OTSU?
        """
        self.nFeatures = nFeatures
        self.kernelSize = kernelSize
        self.kernel = circularKernel(self.kernelSize)
        self.p = p
        self.img = np.zeros((10, 10), dtype=np.uint8) # store the last processed image
        self.maxIntensityChange = maxIntensityChange
        self.showHistogram = showHistogram

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    def _sortLightSources(self, candidates, maxAdditionalCandidates):
        candidates.sort(key=lambda p: (p.intensity, p.area), reverse=True)
        candidates = candidates[:self.nFeatures+maxAdditionalCandidates]
        
        return candidates

    def _calcCandidates(self, gray, lightSources, offset=(0,0), drawImg=None):
        minRad = 5
        radScale = 4
        newLightSources = []
        for ls in lightSources:
            if ls.radius < minRad:
                rcfRad = self.kernelSize#int(round(ls.radius*radScale))
                rcfRad = max(rcfRad, minRad)
                rcf = RCF(np.array(ls.center)-np.array(offset), rcfRad, gray)

                color = (0,0,255)
                if rcf > 0.01: # RCF threshold
                    newLightSources.append(ls)
                    color = (255,0,0)

                rcf = round(rcf, 3)
                cv.circle(drawImg, ls.center, rcfRad, color, 2)
                drawInfo(drawImg, (ls.center[0]+20, ls.center[1]-20), str(rcf), color) 
            else:
                newLightSources.append(ls)

        return newLightSources

    def process(self, gray, maxAdditionalCandidates=0, estDSPose=None, roiMargin=None, drawImg=None):

        grayOrig = gray.copy()

        offset = (0, 0)
        roiCnt = None
        minLightSourceRadius = 0
        peakMargin = 3
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

            peakMargin = max([ls.radius for ls in estDSPose.associatedLightSources])

        # blurring seems to help for large resolution 
        # test_sessions/171121_straight_test.MP4
        gray = cv.GaussianBlur(gray, (5,5),0)

        localMaxImg = localMax(gray, self.kernel)

        imgGray = localMaxImg.copy()
        
        hist = cv.calcHist([imgGray], [0], None, [256], [0,256])
        hist = hist.ravel()
        histPeaks, _ = signal.find_peaks(-hist)
        histPeaks = list(histPeaks) # valleys

        if len(histPeaks) > 0:
            #if np.max(imgGray) == 255:
            #    self.threshold = 254
            #else:
            self.threshold = histPeaks.pop()-1
        else:
            self.threshold = np.max(imgGray)-1
        
        upper = 256
        ret, imgTemp = cv.threshold(imgGray, self.threshold, upper, cv.THRESH_BINARY)
        _, contours, hier = cv.findContours(imgTemp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        candidates = self._calcCandidates(gray, [LightSource(cnt+offset, self.threshold+1) for cnt in contours], offset, drawImg=drawImg)
        cv.drawContours(imgGray, [ls.cnt-offset for ls in candidates], -1, (0), -1)
        
        i = 0
        while True:
            i += 1

            if len(candidates) >= self.nFeatures+maxAdditionalCandidates:
                break

            if len(histPeaks) > 0:
                self.threshold = histPeaks.pop()-1
            else:
                break

            ret, imgTemp = cv.threshold(imgGray, self.threshold, upper, cv.THRESH_BINARY)
            _, contours, hier = cv.findContours(imgTemp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            lightSources = [LightSource(cnt +offset, self.threshold+1) for cnt in contours]
            newCandidates = self._calcCandidates(gray, lightSources, offset, drawImg=drawImg)
            candidates.extend(newCandidates)
            cv.drawContours(imgGray, [ls.cnt-offset for ls in newCandidates], -1, (0), -1)

        print("Iterations: ", i)

        # TODO: check if offset is correct
        #candidates = [LightSource(ls.cnt + offset, ls.intensity) for ls in candidates]
        

        candidates = self._sortLightSources(candidates, maxAdditionalCandidates)

        if False and drawImg is not None:
            for i, ls in enumerate(candidates):
                cv.drawContours(drawImg, [ls.cnt], 0, (255,0,0), -1)
                #drawInfo(drawImg, (ls.center[0]+20, ls.center[1]-20), str(i), (255,0,0))
                minRad = 5
                radScale = 4
                if ls.radius < minRad:
                    rcfRad = self.kernelSize#int(round(ls.radius*radScale))
                    rcfRad = max(rcfRad, minRad)
                    rcf = RCF(np.array(ls.center)-np.array(offset), rcfRad, gray)
                    rcf = round(rcf, 3)
                    cv.circle(drawImg, ls.center, rcfRad, (0,0,255), 2)
                    drawInfo(drawImg, (ls.center[0]+20, ls.center[1]-20), str(rcf), (255,0,0)) 
                #cv.circle(drawImg, ls.center, int(round(ls.radius)), (255,0,255), 2)

        self.img = imgGray #localMaxImg

        if self.showHistogram:
            plt.cla()
            N = 200
            plt.plot(hist)

            for peak in signal.find_peaks(hist)[0]:
                if peak >= N:
                    if peak == self.threshold+1:
                        peakLine = plt.axvline(peak, ymin=0, ymax=1, c="g", label="peaks")
                    else:   
                        peakLine = plt.axvline(peak, ymin=0, ymax=hist[peak]/max(hist[N:]), c="g", label="peaks")

            for peak in signal.find_peaks(-hist)[0]:
                if peak >= N:
                    if peak == self.threshold+1:
                        valleyLine = plt.axvline(peak, ymin=0, ymax=1, c="r", label="valleys")
                    else:   
                        valleyLine = plt.axvline(peak, ymin=0, ymax=hist[peak]/max(hist[N:]), c="r", label="valleys")

            plt.legend(handles=[peakLine, valleyLine])
            plt.xlim([N, 256])
            plt.ylim([0, max(hist[N:])])
            plt.ylabel("Frequency")
            plt.xlabel("Intensity")
            plt.pause(0.0001)
            
        ########### plot peaks ##############

        return drawImg, candidates, roiCnt

class LoG:
    def __init__(self, nFeatures, kernelSize):
        """
        try cv.THRESH_OTSU?
        """
        self.nFeatures = nFeatures
        self.kernelSize = kernelSize
        self.sigma = (self.kernelSize- 1)/3
        print(self.sigma)
        #self.kernel = circularKernel(self.kernelSize)
        self.img = np.zeros((10, 10), dtype=np.uint8) # store the last processed image

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    def _sortLightSources(self, candidates, maxAdditionalCandidates):
        candidates.sort(key=lambda p: (p.intensity, p.area), reverse=True)
        candidates = candidates[:self.nFeatures+maxAdditionalCandidates]
        
        return candidates

    def process(self, gray, estDSPose=None, roiMargin=None, drawImg=None):
        blurred = cv.GaussianBlur(gray, (self.kernelSize,self.kernelSize), sigmaX=self.sigma, sigmaY=self.sigma) 
        dst = cv.Laplacian(blurred, ddepth=cv.CV_16S, ksize=3)
        dst = cv.convertScaleAbs(dst, alpha=255./dst.max())
        print(dst.max())
        self.img = dst
        return []
        print("--hej--_")
        print(dst.min())
        print(dst.max())
        print("-------")
        absMax = max(dst.max(), abs(dst.min()))
        dst = dst.astype(np.float32)
        dst = dst/absMax
        
        print(dst.min())
        print(dst.max())
        dst += 0.5
        print(dst.min())
        print(dst.max())
        dst = cv.convertScaleAbs(dst, alpha=255./dst.max())
        self.img = dst
        #self.img = dst

        return []

if __name__ == '__main__':
    pass