#!/usr/bin/env python
import time
import cv2 as cv
import numpy as np
import itertools
from lolo_perception.plotting_utils import plotHistogram
import lolo_perception.py_logging as logging

def scaleImage(img, scalePercent):
    """
    Scales the image (up or down) base on scalePercentage.
    Preserves aspect ratio.
    """
    return cv.resize(img, (int(img.shape[1]*scalePercent) , int(img.shape[0]*scalePercent)))


def scaleImageToWidth(img, desiredWidth):
    return scaleImage(img, float(desiredWidth)/img.shape[1])


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


def regionOfInterest(points, wMargin, hMargin, shape=None):
    topMost = [None, np.inf]
    rightMost = [-np.inf]
    bottomMost = [None, -np.inf]
    leftMost = [np.inf]
    for p in points:
        if p[1] < topMost[1]:
            topMost = list(p)

        if p[0] > rightMost[0]:
            rightMost = list(p)

        if p[1] > bottomMost[1]:
            bottomMost = list(p)

        if p[0] < leftMost[0]:
            leftMost = list(p)
    
    topMost[1] -= hMargin
    rightMost[0] += wMargin
    bottomMost[1] += hMargin
    leftMost[0] -= wMargin
    cnt = np.array([topMost, rightMost, bottomMost, leftMost], dtype=np.int32)
    x, y, w, h = cv.boundingRect(cnt)

    if shape is not None:
        if x < 0:
            w += x
            x = 0
        else:
            x = min(shape[1]-1, x)
        if y < 0:
            h += y
            y = 0
        else:
            y = min(shape[0]-1, y)

    roiCnt = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)

    return (x, y, w, h), roiCnt


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

def contourConvexDefects(ls):
    r = cv.minEnclosingCircle(ls.cnt)[1]
    
    if r > 5:
        hull = cv.convexHull(ls.cnt,returnPoints=False)
        defects = cv.convexityDefects(ls.cnt, hull)
        dThresh = 0.2*r
        #cv.circle(drawImg, contourCentroid(cntPeakOffset), int(dThresh), (255,0,255), 1)
    
        if defects is not None:
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                if d/256.0 > dThresh:
                    found = False


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

def localMax(gray, kernel, borderValue=255):
    dilated = cv.dilate(gray, kernel, borderValue=borderValue, iterations=1)
    eqMask = cv.compare(gray, dilated, cv.CMP_EQ)
    localMaxImg = cv.bitwise_and(gray, gray, mask=eqMask)
    return localMaxImg

def localMax2(gray, kernel, borderValue=255):
    dilated = cv.dilate(gray, kernel, borderValue=borderValue, iterations=1)
    eqMask = cv.compare(gray, dilated, cv.CMP_EQ)
    localMaxImg = cv.bitwise_and(gray, gray, mask=eqMask)
    return eqMask, localMaxImg

def localMaxSupressed3(gray, kernel, p):
    seed = gray*p
    seed = gray.astype(np.uint8)
    dilated = cv.dilate(gray, kernel, borderValue=0, iterations=1)
    eqMask = cv.compare(seed, dilated, cv.CMP_GE)
    localMaxImg = cv.bitwise_and(gray, gray, mask=eqMask)
    return localMaxImg

def localMaxSupressed2(gray, kernel, p):
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

def localMaxDiff(gray, kernel, p):
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
    return eqMask, localMaxImg

def localMaxDiff2(gray, localMaxKernel, localMinKernel, p):
    dilated = cv.dilate(gray, localMaxKernel, borderValue=0, iterations=1)
    eroded = cv.erode(gray, localMinKernel, borderValue=0, iterations=1)
    if p < 1:
        # Percentage threshold
        dilatedSupressed = dilated.astype(np.float32)*p
    else:
        # Fixed threshold
        dilatedSupressed = dilated-p
    dilatedSupressed = dilatedSupressed.astype(np.uint8)
    mask1 = cv.compare(gray, dilatedSupressed, cv.CMP_GE)
    mask2 = cv.compare(gray, eroded, cv.CMP_GE)
    localMaxImg = cv.bitwise_and(gray, gray, mask=eqMask)

    return localMaxImg

def localMin(gray, kernel):
    dilated = cv.erode(gray, kernel, borderValue=0, iterations=1)
    eqMask = cv.compare(gray, dilated, cv.CMP_EQ)
    localMinImg = cv.bitwise_and(gray, gray, mask=eqMask)
    return localMinImg


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
    """
    TODO -make efficient using floodfill
    # Flood fill region around target pixel with unique color
    floodfill_mask = img.copy()
    cv2.floodFill(floodfill_mask, None, (col, row), 255)

    # Find contours of unique color
    contours, _ = cv2.findContours(floodfill_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get contour surrounding target pixel
    contour = None
    for c in contours:
        if cv2.pointPolygonTest(c, (col, row), False) >= 0:
            contour = c
            break
    """
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


def findPeakContourFloodFill(gray, center, loDiff=0, upDiff=0, offset=None):
    """
    
    """
    
    floodfillMask = gray.copy()
    mask = np.zeros((gray.shape[0] + 2, gray.shape[1] + 2), dtype=np.uint8)
    #p = 0.97
    #loDiff = int(gray[center[1], center[0]]*(1-p))
    retval, image, mask, rect = cv.floodFill(floodfillMask, mask, center, 255, loDiff=loDiff, upDiff=upDiff, flags=cv.FLOODFILL_FIXED_RANGE)
    mask *= 255
    mask = mask[1:mask.shape[0]-1, 1:mask.shape[1]-1]
    _, contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if offset is None:
        contoursOffset = [cnt for cnt in contours]
    else:
        _, contoursOffset, hier = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE, offset=offset)

    # Get contour surrounding target pixel
    foundCnt = None
    foundCntOffset = None
    if len(contours) > 1:
        raise Exception("This should not happen!")
    
    contour = contours[0]
    contourOffset = contoursOffset[0]
    if withinContour(center[0], center[1], contour):
        foundCnt = contour
        foundCntOffset = contourOffset

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


def localPeakFloodFill(gray, kernel, pMin, pMax, n, minThresh=0, margin=1, ignorePAtMax=True, offset=(0,0), maxIter=100, validCntCB=None, drawImg=None, drawInvalidPeaks=False):
    
    # Calculate local max and threshold (to zero) the local max with the given minThresh
    localMaxImage = localMax(gray, kernel, borderValue=0)
    _, localMaxImage = cv.threshold(localMaxImage, minThresh, 255, cv.THRESH_TOZERO)

    localMaxImageMasked = localMaxImage.copy() # A copy of the local max image that will be "painted black" for each peak iteration
    maxIntensity = np.inf                      # The current max intensity of the masked local max image
    peakCenters = []                            
    peakContours = []

    iterations = 0
    while True:
        if len(peakCenters) >= n:
            logging.debug("Found {} peaks, breaking".format(n))
            break

        if np.max(localMaxImageMasked) < maxIntensity:
            maxIntensity = np.max(localMaxImageMasked)
            if maxIntensity == 0:
                break
            if maxIntensity == 255 and ignorePAtMax:
                # if it is maximum intensity, ignore p
                logging.debug("Ignoring p at max")
                threshold = 254 # TODO: maybe another value is more suitable?
            else:
                if pMax < 1:
                    pTemp = pDecay(.175, pMin, pMax, I=maxIntensity)
                    threshold = int(pTemp*maxIntensity - 1)
                else:
                    threshold = max(maxIntensity-1-pMax, 0)

        elif np.max(localMaxImageMasked) > maxIntensity:
            raise Exception("Something went wrong in local peak,")

        iterations += 1
        # TODO: this could probably be more efficient, extracting all contours at this intensity level at the same time
        maxIndx = np.unravel_index(np.argmax(localMaxImageMasked), gray.shape)
        center = maxIndx[1], maxIndx[0]
        cntPeak, cntPeakOffset = findPeakContourFloodFill(gray, center, maxIntensity-threshold, 255, offset=offset)#, mode=cv.RETR_LIST)

        if cntPeak is None:
            logging.error("Something went wrong...")
            break

        # Check if contour is on edge
        cnts = [cntPeak]
        if maxIntensity != 255:
            # we only remove edge contours if they don't have max value
            cnts = removeContoursOnEdges(localMaxImage, cnts)
        if not cnts:
            # Egge
            if drawImg is not None and drawInvalidPeaks:
                cv.drawContours(drawImg, [cntPeakOffset], 0, (0, 255, 255), 1)
        elif validCntCB is not None and validCntCB(cntPeak, maxIntensity) is False:
            # Invalid contour
            if drawImg is not None and drawInvalidPeaks:
                cv.drawContours(drawImg, [cntPeakOffset], 0, (255, 0, 255), 1)
        else:
            for pc in peakCenters:
                    
                if withinContour(pc[0], pc[1], cntPeakOffset) or nextToContour(pc[0], pc[1], cntPeakOffset, margin=margin): # TODO: this should be checked using the center of the new peak instead
                    # Overlapping
                    if drawImg is not None and drawInvalidPeaks:
                        cv.drawContours(drawImg, [cntPeakOffset], 0, (0, 0, 255), 1)
                    break

            else:
                # Found
                peakCenters.append((center[0]+offset[0], center[1]+offset[1]))
                peakContours.append(cntPeakOffset)
                if drawImg is not None:
                    cv.drawContours(drawImg, [cntPeakOffset], 0, (255, 0, 0), -1)

        localMaxImageMasked = cv.drawContours(localMaxImageMasked, [cntPeak], 0, (0, 0, 0), -1)

        if iterations >= maxIter:
            logging.info("Local Peak maxiter reached")
            break
    
    if drawImg is not None:
        for i, pc in enumerate(peakCenters):
            cv.circle(drawImg, pc, 1, (255,0,255), 1)
            #drawInfo(drawImg, (pc[0]+15,pc[1]-15), str(i+1))

    return localMaxImage, localMaxImageMasked, peakCenters, peakContours, iterations


def localPeak(gray, kernel, pMin, pMax, n, minThresh=0, margin=1, ignorePAtMax=True, offset=(0,0), maxIter=100, validCntCB=None, drawImg=None, drawInvalidPeaks=False):
    
    # Calculate local max and threshold (to zero) the local max with the given minThresh
    localMaxImage = localMax(gray, kernel, borderValue=0)
    _, localMaxImage = cv.threshold(localMaxImage, minThresh, 255, cv.THRESH_TOZERO)

    localMaxImageMasked = localMaxImage.copy() # A copy of the local max image that will be "painted black" for each peak iteration
    maxIntensity = np.inf                      # The current max intensity of the masked local max image
    peakCenters = []                            
    peakContours = []

    iterations = 0
    while True:
        if len(peakCenters) >= n:
            logging.debug("Found {} peaks, breaking".format(n))
            break
        if np.max(localMaxImageMasked) < maxIntensity:
            maxIntensity = np.max(localMaxImageMasked)
            if maxIntensity == 0:
                break
            if maxIntensity == 255 and ignorePAtMax:
                # if it is maximum intensity, ignore p
                logging.debug("Ignoring p at max")
                threshold = 254 # TODO: maybe another value is more suitable?
            else:
                if pMax < 1:
                    pTemp = pDecay(.175, pMin, pMax, I=maxIntensity)
                    threshold = int(pTemp*maxIntensity - 1)
                else:
                    threshold = max(maxIntensity-1-pMax, 0)

            ret, threshImg = cv.threshold(gray, threshold, 256, cv.THRESH_BINARY)
        elif np.max(localMaxImageMasked) > maxIntensity:
            raise Exception("Something went wrong in local peak,")

        iterations += 1
        # TODO: this could probably be more efficient, extracting all contours at this intensity level at the same time
        maxIndx = np.unravel_index(np.argmax(localMaxImageMasked), gray.shape)
        center = maxIndx[1], maxIndx[0]
        cntPeak, cntPeakOffset = findPeakContourAt(threshImg, center, offset=offset)#, mode=cv.RETR_LIST)

        if cntPeak is None:
            logging.error("Something went wrong...")
            break

        # Check if contour is on edge
        cnts = [cntPeak]
        if maxIntensity != 255:
            # we only remove edge contours if they don't have max value
            cnts = removeContoursOnEdges(localMaxImage, cnts)
        if not cnts:
            # Egge
            if drawImg is not None and drawInvalidPeaks:
                cv.drawContours(drawImg, [cntPeakOffset], 0, (0, 255, 255), 1)
        elif validCntCB is not None and validCntCB(cntPeak, maxIntensity) is False:
            # Invalid contour
            if drawImg is not None and drawInvalidPeaks:
                cv.drawContours(drawImg, [cntPeakOffset], 0, (255, 0, 255), 1)
        else:
            for pc in peakCenters:
                    
                if withinContour(pc[0], pc[1], cntPeakOffset) or nextToContour(pc[0], pc[1], cntPeakOffset, margin=margin): # TODO: this should be checked using the center of the new peak instead
                    # Overlapping
                    if drawImg is not None and drawInvalidPeaks:
                        cv.drawContours(drawImg, [cntPeakOffset], 0, (0, 0, 255), 1)
                    break

            else:
                # Found
                peakCenters.append((center[0]+offset[0], center[1]+offset[1]))
                peakContours.append(cntPeakOffset)
                if drawImg is not None:
                    cv.drawContours(drawImg, [cntPeakOffset], 0, (255, 0, 0), -1)

        localMaxImageMasked = cv.drawContours(localMaxImageMasked, [cntPeak], 0, (0, 0, 0), -1)

        if iterations >= maxIter:
            logging.info("Local Peak maxiter reached")
            break
    
    if drawImg is not None:
        for i, pc in enumerate(peakCenters):
            cv.circle(drawImg, pc, 1, (255,0,255), 1)
            #drawInfo(drawImg, (pc[0]+15,pc[1]-15), str(i+1))

    return localMaxImage, localMaxImageMasked, peakCenters, peakContours, iterations


def localPeakWindowed(gray, kernel, pMin, pMax, n, windowRad, minThresh=0, margin=1, ignorePAtMax=True, offset=(0,0), maxIter=10000000, validCntCB=None, drawImg=None, drawInvalidPeaks=False):
    """

    """
    if windowRad == 0:
        return localPeak(gray, kernel, pMin, pMax, n, minThresh, margin, ignorePAtMax, offset, maxIter, validCntCB, drawImg, drawInvalidPeaks)

    peaksDilation = localMax(gray, kernel, borderValue=0) # local max
    _, peaksDilation = cv.threshold(peaksDilation, minThresh, 255, cv.THRESH_TOZERO)

    grayMasked = gray.copy()
    peaksDilationMasked = peaksDilation.copy()
    maxIntensity = np.inf
    peakCenters = []
    peakContours = []

    iterations = 0
    
    while True:
        if len(peakCenters) >= n:
            logging.debug("Found {} peaks, breaking".format(n))
            break

        maxIndx = np.unravel_index(np.argmax(peaksDilationMasked), gray.shape)
        center = maxIndx[1], maxIndx[0]
        maxIntensity = np.max(peaksDilationMasked)

        if maxIntensity == 0:
            break
        if maxIntensity == 255 and ignorePAtMax:
            # if it is maximum intensity, ignore p
            logging.debug("Ignoring p at max")
            threshold = 254 # TODO: maybe another value is more suitable?
        else:
            #pTemp = maxIntensity/255.*(pMax-pMin) + pMin
            #pTemp = pDecay(.1801, pMin, pMax, I=maxIntensity) # with pMax = .98
            pTemp = pDecay(.175, pMin, pMax, I=maxIntensity)
            threshold = int(pTemp*maxIntensity - 1)

        windowRadius = windowRad
        if maxIntensity >= 230:
            windowRadius = 300

        x = center[0]-windowRadius
        y = center[1]-windowRadius
        h = windowRadius*2 - 1
        w = windowRadius*2 - 1
        center = windowRadius, windowRadius
        if x < 0:
            center = center[0]+x, center[1]
            w += x
            x = 0
        if y < 0:
            center = center[0], center[1]+y
            h += y
            y = 0

        windowImg = grayMasked[y:y+h, x:x+w]
        ret, threshImg = cv.threshold(windowImg, threshold, 256, cv.THRESH_BINARY)
        windowOffset = x, y
        totalOffset = (x+offset[0], y+offset[1])

        iterations += 1
        # TODO: this could probably be more efficient, extracting all contours at this intensity level at the same time

        cntPeak, cntPeakOffset = findPeakContourAt(threshImg, center, offset=totalOffset)#, mode=cv.RETR_LIST)

        if False:
            centerTmp = (center[0]+totalOffset[0], center[1]+totalOffset[1])
            tempImg = drawImg.copy()
            xTemp = totalOffset[0]
            yTemp = totalOffset[1]
            cv.rectangle(tempImg, (xTemp, yTemp), (xTemp+w, yTemp+h), color=(255,0,0), thickness=2)
            cv.circle(tempImg, centerTmp, 3, color=(255,0,255))
            cv.imshow("local peak 3", scaleImage(tempImg, 0.3))
            cv.waitKey(0)

        if cntPeak is None:
            logging.error("Something went wrong...")
            break

        # Check if contour is on edge
        cnts = [cntPeak]
        if maxIntensity != 255:
            # we only remove edge contours if they don't have max value
            cnts = removeContoursOnEdges(threshImg, cnts)
        if not cnts:
            # Edge
            if drawImg is not None and drawInvalidPeaks:
                cv.drawContours(drawImg, [cntPeakOffset], 0, (0, 255, 255), 1)
        elif validCntCB is not None and validCntCB(cntPeak) is False:
            if drawImg is not None and drawInvalidPeaks:
                # Invalid contour
                cv.drawContours(drawImg, [cntPeakOffset], 0, (255, 0, 255), 1)
        else:
            for pc in peakCenters:
                    
                if withinContour(pc[0], pc[1], cntPeakOffset) or nextToContour(pc[0], pc[1], cntPeakOffset, margin=margin): # TODO: this should be checked using the center of the new peak instead
                    # Overlapping
                    if drawImg is not None and drawInvalidPeaks:
                        cv.drawContours(drawImg, [cntPeakOffset], 0, (0, 0, 255), 1)
                    break

            else:
                # Found
                peakCenters.append((center[0]+totalOffset[0], center[1]+totalOffset[1]))
                peakContours.append(cntPeakOffset)
                if drawImg is not None:
                    cv.drawContours(drawImg, [cntPeakOffset], 0, (255, 0, 0), -1)

        peaksDilationMasked = cv.drawContours(peaksDilationMasked, [cntPeak], 0, (0, 0, 0), -1, offset=windowOffset)

        if iterations >= maxIter:
            logging.debug("Peak maxiter reached")
            break
    
    if drawImg is not None:
        for i, pc in enumerate(peakCenters):
            cv.circle(drawImg, pc, 1, (255,0,255), 1)
            drawInfo(drawImg, (pc[0]+15,pc[1]-15), str(i+1))

    return peaksDilation, peaksDilationMasked, peakCenters, peakContours, iterations

def findNPeaksLoG(gray, kernel, pMin, pMax, n, minThresh=0, margin=1, ignorePAtMax=True, offset=(0,0), maxIter=10000000, drawImg=None, drawInvalidPeaks=False, validCntCB=None):
    peaksDilation = localMax(gray, kernel) # local max
    _, peaksDilation = cv.threshold(peaksDilation, minThresh, 255, cv.THRESH_TOZERO)

    blobRadius = 7
    sigma = (blobRadius-1.0)/3.0
    ksize = int(round(sigma*6))
    if ksize % 2 == 0:
        ksize += 1
    laplaceKernel = 3
    #blurred = cv.GaussianBlur(gray, (ksize,ksize), sigma)
    log = cv.Laplacian(gray, ddepth=cv.CV_16S, ksize=laplaceKernel)
    log = cv.convertScaleAbs(log, alpha=255./log.max())

    dst = log.astype(np.float32)*peaksDilation.astype(np.float32)
    dst = dst*255./np.max(dst)
    dst = dst.astype(np.uint8)
    indices = np.where(peaksDilation > 255) # 240
    dst[indices] = peaksDilation[indices]

    maxIntensity = np.inf
    dstMasked = dst.copy()

    peakCenters = []
    peakContours = []

    iterations = 0
    while True:
        if np.max(dstMasked) != maxIntensity: # this intensity is just for sorting, doesn't reflect actual intensity
            if len(peakCenters) >= n:
                print("Found {} peaks, breaking".format(n))
                break
            else:
                maxIntensity = np.max(dstMasked)
                if maxIntensity == 0:
                    break

                maxIndx = np.unravel_index(np.argmax(dstMasked), gray.shape)
                intensity = gray[maxIndx[0], maxIndx[1]] # this is the actual intensity
                pTemp = pDecay(.175, pMin, pMax, I=intensity)
                threshold = int(pTemp*intensity - 1)

                ret, threshImg = cv.threshold(gray, threshold, 256, cv.THRESH_BINARY)

        iterations += 1
        # TODO: this could probably be more efficient, extracting all contours at this intensity level at the same time
        maxIndx = np.unravel_index(np.argmax(dstMasked), gray.shape)
        center = maxIndx[1], maxIndx[0]
        cntPeak, cntPeakOffset = findPeakContourAt(threshImg, center, offset=offset)#, mode=cv.RETR_LIST)

        if cntPeak is None:
            print("Something went wrong?!")
            break
        cnts = [cntPeak]
        if intensity != 255:
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

        cv.drawContours(dstMasked, [cntPeak], 0, (0, 0, 0), -1)

        if iterations >= maxIter:
            print("Peak maxiter reached")
            break
    
    if drawImg is not None:
        for i, pc in enumerate(peakCenters):
            cv.circle(drawImg, pc, 1, (255,0,255), 1)
            drawInfo(drawImg, (pc[0]+15,pc[1]-15), str(i+1))

    return dst, dstMasked, peakCenters, peakContours, iterations


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

def featureAssociationSquareImprovedWithFilter(featurePoints, detectedLightSources, p, drawImg=None):
    """
    p - decimal fraction (percentage) threshold
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

    if None in refIdxs:
        logging.error("Association failed. Ref idxs: {}".format(refIdxs))
        raise Exception("Association failed")

    # identify top two and bottom two
    sortedByAscendingY = sorted(detectedLightSources, key=lambda ls: ls.center[1])
    topTwo = sortedByAscendingY[:2]
    bottomTwo = sortedByAscendingY[-2:]

    # sort by x to identify [left, right]
    topLeft, topRight = sorted(topTwo, key=lambda ls: ls.center[0])
    bottomLeft, bottomRight = sorted(bottomTwo, key=lambda ls: ls.center[0])

    # Filter/remove non-square detection
    # TODO: use distance and angle instead
    # TODO: This might not work well when there is a lot of perspective distortion

    # check if width is similar
    wTop = topRight.center[0]-topLeft.center[0]
    wBottom = bottomRight.center[0]-bottomLeft.center[0]
    if abs(wTop-wBottom) > max(wTop, wBottom)*p:
        logging.trace("Ignored non-square")
        return None

    # check if height is similar
    hLeft = bottomLeft.center[1]-topLeft.center[1]
    hRight = bottomRight.center[1]-topRight.center[1]
    if abs(hLeft-hRight) > max(hLeft, hRight)*p:
        logging.trace("Ignored non-square")
        return None

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
            raise Exception("Something went wrong")
    

    return associatedLightSources

def featureAssociation(featurePoints, detectedLightSources, featurePointsGuess=None, drawImg=None):
    """
    Assumes that the orientation of the feature model is 0 around every axis relative to the camera frame
    i.e. x and y values are handled directly in the image plane
    """
    detectedLightSources = list(detectedLightSources)
    detectedPoints = [ls.center for ls in detectedLightSources]
    if len(detectedPoints) < len(featurePoints):
        logging.warn("Not enough detected features given")
        return None

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

    return associatedLightSources
               

if __name__ == '__main__':
    pass