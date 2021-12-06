#!/usr/bin/env python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from perception_utils import projectPoints

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
        return img

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

    def process(self, img):
        # this removes reflections effectively at close distances but reduces range

        imgTemp = img.copy()

        i = self.iterations
        imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_OPEN, self.kernel, iterations=i)
        _, contours, hier = cv.findContours(imgTemp, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        while len(contours) >= self.nFeatures and i < self.maxIter:
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

def drawInfo(img, center, text, color=(255, 0, 0)):
    # font
    font = cv.FONT_HERSHEY_SIMPLEX
    # org
    org = (center[0]-10, center[1]+10) # some displacement to make it look good
    # fontScale
    fontScale = 1
    # Line thickness of 2 px
    thickness = 2
    # Using cv2.putText() method
    image = cv.putText(img, text, org, font, 
                    fontScale, color, thickness, cv.LINE_AA)
    
    return image

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

def removeContoursOnEdges(contours):
    newContours = []
    for cnt in contours:
        leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
        topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
        bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
        if leftmost[0] != 0 and rightmost[0] != 639 and topmost[1] != 0 and bottommost[1] != 479:
            print(leftmost)
            newContours.append(cnt)
        else:
            print("Removed contour on edge")
    return newContours

def colorThreshold(frame):
    global image_msg
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)
    return res

def contourAveragePixelIntensity(cnt, img):
    # https://stackoverflow.com/questions/63456523/opencv-average-intensity-of-contour-figures-in-grayscale-image
    mask = np.zeros(img.shape, np.uint8)
    cv.fillPoly(mask, pts=[cnt], color=(255,255,255))
    mask_contour = mask == 255
    intensity = np.mean(img[mask_contour])
    return intensity

def cnotourCentroid(cnt):
    area = cv.contourArea(cnt)
    if area == 0:
        cx, cy = cnt[0][0]
    else:
        M = cv.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    return cx, cy

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

def weightedCentroid(gray):
    # https://stackoverflow.com/questions/53719588/calculate-a-centroid-with-opencv
    pass

def featureAssociation(featurePoints, detectedPoints, featurePointsGuess=None):
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
        featurePointsGuess = [(maxR*x/maxFR + centerx, maxR*y/maxFR + centery) for x,y,_,_ in featurePoints]
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
    
    return np.array(associatedPoints, dtype=np.float32), featurePointsGuess
    
class ThresholdFeatureExtractor:
    # Check this for contour properties
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_properties/py_contour_properties.html
    def __init__(self, featureModel, camera, p=.02, erosionKernelSize=5, maxIter=10, useKernel=True, drawContours=False):
        if isinstance(featureModel, int):
            self.featureModel = None
            self.nFeatures = featureModel
        else:
            self.featureModel = featureModel
            self.nFeatures = len(self.featureModel.features)
        self.camera = camera
        
        #self.pHold = AdaptiveThreshold(self.nFeatures, 250, 1, 150)#PercentageThreshold(p)
        self.pHold = PercentageThreshold(p)
        self.adaOpen = AdaptiveOpen(self.nFeatures, kernelSize=erosionKernelSize, maxIter=maxIter)
        if useKernel:
            self.adaOpen = AdaptiveErodeKernel(self.nFeatures, kernelSize=erosionKernelSize, maxIter=maxIter)
        self.drawContours = drawContours

    #def regionOfInterest(self, img, estTranslationVector, estRotationVector, wMargin, hMargin):
    def regionOfInterest(self, img, featurePointsGuess, wMargin, hMargin):
        featurePointsGuess = self.camera.metersToUV(featurePointsGuess)
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

    def projectedFeatures(self, estTranslationVec, estRotationVec):
        points = self.featureModel.features[:, :3].copy()
        projPoints, jacobian = cv.projectPoints(points, 
                                                estRotationVec, 
                                                estTranslationVec, 
                                                self.camera.cameraMatrix, 
                                                self.camera.distCoeffs)
        projPoints = np.array([p[0] for p in projPoints])

        return projPoints

    def __call__(self, gray, imgColor, estTranslationVec=None, estRotationVec=None):
        img = gray.copy()

        featurePointsGuess = None
        if estTranslationVec is not None and self.featureModel:
            roiImg = img.copy()
            featurePointsGuess = projectPoints(estTranslationVec, 
                                               estRotationVec, 
                                               self.camera, 
                                               self.featureModel.features.copy())
            roiCnt = self.regionOfInterest(img, featurePointsGuess, wMargin=30, hMargin=30)
            mask = np.zeros(roiImg.shape, np.int8)
            mask = cv.drawContours(mask, [roiCnt], -1, (255), -1)
            cv.drawContours(imgColor, [roiCnt], -1, (0, 255, 0), 3)
            #roiImg = cv.rectangle(roiImg, (x,y), (x+w, y+h), (0, 255, 0), 10, cv.LINE_AA)
            roiImg = cv.bitwise_and(img, img, mask=mask)
            img = roiImg

        #img = cv.GaussianBlur(img,(5,5),0)
        
        # https://docs.opencv.org/4.5.3/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
        #blur = cv.bilateralFilter(img,5,200,50) # smoothing but, keeps edges. Makes the adaOpen faster
        #img = blur
        img = self.pHold.process(img)
        #img = fillContoursOnEdges(img) # All contours connected to the edges are background

        _, contours, hier = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        if self.drawContours: cv.drawContours(imgColor, contours, -1, (0, 0, 255), 3)

        contours.sort(key=contourRatio, reverse=True)
        #contours = contourAreaOutliers(contours) # dont want to remove outliers here, remove noise first

        img = self.adaOpen.process(img) # removes noise
        
        _, contoursNew, hier = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        if self.drawContours: cv.drawContours(imgColor, contoursNew, -1, (0, 255, 0), 3)

        # error: The truth value of an array with more than one element is ambiguous
        #contoursNew = np.array(contoursNew)
        #ds = contourAreaDistanceFromMean(contoursNew)
        #print(ds.shape)
        #print(contoursNew.shape)
        #contoursNew = [cnt for d, cnt in sorted(zip(ds, contoursNew))]
        contoursNew.sort(key=contourRatio, reverse=True)
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
                    if self.drawContours:
                        cv.drawContours(imgColor,[cntNew], 0, (255, 0, 0), -1) # or maybe draw new??
                        drawInfo(imgColor, (cx+10, cy-10), str(ratio), color=(255, 0, 255))
                    #break We don't break here, the old contour might be two overlapping lights

        points = points[:self.nFeatures]
        if len(points) == 0:
            return img, []

        if self.camera: # if no camera defined, we don't associate (need pixelsize to associate)
            points = np.array(points, dtype=np.float32)

            # convert from pixels to meters
            points[:,0] *= self.camera.pixelWidth
            points[:,1] *= self.camera.pixelHeight

            # we need to rotate the points if we want the relative orientation to be different, 
            # otherwise feature association will fail
            points3DAss = self.featureModel.features[:, :3].transpose()
            points3DAss = np.matmul( R.from_euler("XYZ", (0, np.pi, 0)).as_dcm(), points3DAss).transpose()
            
            #points3DAss = self.featureModel.features.copy()
            points3DAss = np.append(points3DAss, np.ones((points3DAss.shape[0], 1)), axis=1)

            associatedPoints, featurePointsGuess = featureAssociation(points3DAss, points, featurePointsGuess)

            for i in range(len(associatedPoints)):
                # convert points to pixels
                px = associatedPoints[i][0] / self.camera.pixelWidth
                py = associatedPoints[i][1] / self.camera.pixelHeight
                fpx = featurePointsGuess[i][0] / self.camera.pixelWidth
                fpy = featurePointsGuess[i][1] / self.camera.pixelHeight
                drawInfo(imgColor, (int(px), int(py)), str(i))
                drawInfo(imgColor, (int(fpx), int(fpy)), str(i), color=(0, 0, 255))
                cv.circle(imgColor, (int(px), int(py)), 2, (255, 0, 0), 3)
            points = associatedPoints

        return img, points

class GradientFeatureExtractor:
    def __init__(self, camera, featureModel):
        if isinstance(featureModel, int):
            self.featureModel = None
            self.nFeatures = featureModel
        else:
            self.featureModel = featureModel
            self.nFeatures = len(self.featureModel.features)
        self.camera = camera

        self.pHold = PercentageThreshold(0.9)
        self.adaOpen = AdaptiveOpen(len(self.featureModel.features), kernelSize=3, startIterations=1, maxIter=3)

    def __call__(self, gray, imgColor, estTranslationVec=None, estRotationVec=None):
        res = gray

        grad_x = cv.Sobel(res, cv.CV_64F, 1, 0, ksize=3)
        grad_y = cv.Sobel(res, cv.CV_64F, 0, 1, ksize=3)

        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)
        res = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        #res = cv.GaussianBlur(res, (3,3), 0)
        res = cv.blur(res, (5,5))

        res = self.pHold.process(res)
        res = self.adaOpen.process(res)
        #res = fillContours(res, 0.7)

        return res, []

if __name__ == '__main__':
    nFeatures = 4
    plt.figure()

    featureExtractor = ThresholdFeatureExtractor(featureModel=self.featureModel, camera=self.camera, p=0.01, erosionKernelSize=5, maxIter=3, useKernel=False)
    cap = cv.VideoCapture(2)
    while True:
        _, imgColor = cap.read()
        gray = cv.cvtColor(imgColor, cv.COLOR_BGR2GRAY)

        hist = cv.calcHist([gray], [0], None, [256], [0, 256])
        plt.cla()
        plt.plot(hist)
        #res, points = extract_features_thres(imgColor.copy(), nFeatures)
        #res3, points = extract_features_kmeans(imgColor.copy(), nFeatures)
        #cv.imshow("gray", res1)
        plt.pause(0.1)
        cv.waitKey(100)