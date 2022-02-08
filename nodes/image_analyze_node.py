#!/usr/bin/env python
import sys
from os.path import isfile, join
from os import listdir
import os
import time
import cv2 as cv
import matplotlib.pyplot as plt
import scipy.misc
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import rospy
import rospkg
import rosbag
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, PoseArray

from lolo_perception.camera_model import Camera
from lolo_perception.perception_ros_utils import readCameraYaml, msgToImagePoints
from lolo_perception.feature_extraction import MeanShiftTracker, LightSourceTracker, RCFS, RCF, findPeakContourAt, circularKernel

# for _analyzeImage
from lolo_perception.feature_extraction import contourCentroid, findNPeaks, AdaptiveThresholdPeak, GradientFeatureExtractor, findAllPeaks, findPeaksDilation, peakThreshold, AdaptiveThreshold2, circularKernel, removeNeighbouringContours, removeNeighbouringContoursFromImg, AdaptiveOpen, maxShift, meanShift, drawInfo, medianContourAreaFromImg, findMaxPeaks, findMaxPeak

import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

def drawErrorCircle(img, errCircle, i, color, font=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1, delta=5):
    (x, y, r) = errCircle
    d = int(1/np.sqrt(2)*r) + delta
    org = (x+d, y+d) # some displacement to make it look good
    cv.putText(img, str(i), org, font, fontScale, color, thickness, cv.LINE_AA)
    cv.circle(img, (x,y), 1, color, 2)
    cv.circle(img, (x,y), r, color, 1)

class ImageLabeler:
    # https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
    def __init__(self):
        self.currentPoint = None
        self.currentRadius = 10
        self.changeErrCircleIdx = None
        self.currentImg = None
        self.i = 0

        self.img = None
        self.errCircles = None

    def _drawInfoText(self, img):
        dy = 15
        yDisplacement = 20
        infoText = "+ - increase size"
        cv.putText(img, infoText, (10,yDisplacement), 1, 1, (255, 255, 255), 1, cv.LINE_AA)
        yDisplacement += dy
        infoText = "- - decrease size"
        cv.putText(img, infoText, (10,yDisplacement), 1, 1, (255, 255, 255), 1, cv.LINE_AA)
        yDisplacement += dy
        infoText = "r - remove last label"
        cv.putText(img, infoText, (10,yDisplacement), 1, 1, (255, 255, 255), 1, cv.LINE_AA)
        yDisplacement += dy
        n = len(self.errCircles)
        if n == 1:
            infoText = "0 - change label"
            cv.putText(img, infoText, (10,yDisplacement), 1, 1, (255, 255, 255), 1, cv.LINE_AA)
            yDisplacement += dy
        elif n > 1:
            infoText = "(0-{}) - change label".format(n-1)
            cv.putText(img, infoText, (10,yDisplacement), 1, 1, (255, 255, 255), 1, cv.LINE_AA)
            yDisplacement += dy
        infoText = "n - next image"
        cv.putText(img, infoText, (10,yDisplacement), 1, 1, (255, 255, 255), 1, cv.LINE_AA)

    def draw(self):
        img = self.img.copy()
        self._drawInfoText(img)

        font = cv.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 2
        delta = 5
        for i, (x, y, r) in enumerate(self.errCircles):
            if i == self.changeErrCircleIdx:
                color = (255, 0, 255)
            else:
                color = (0, 255, 0)
            drawErrorCircle(img, (x, y, r), i, color, font, fontScale, thickness, delta)

        if self.currentPoint:
            color = (0, 0, 255)
            idx = self.changeErrCircleIdx if self.changeErrCircleIdx is not None else len(self.errCircles)
            errCircle = self.currentPoint + (self.currentRadius,)
            drawErrorCircle(img, errCircle, idx, color, font, fontScale, thickness, delta)
            
        self.currentImg = img

    def click(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            if self.currentPoint is None:
                self.currentPoint = (x, y)
                self.currentRadius = 10
            else:
                p = self.currentPoint
                for (x, y, r) in self.errCircles:
                    d = np.sqrt(pow(p[0]-x, 2) + pow(p[1]-y, 2))
                    if d < r+self.currentRadius:
                        if self.changeErrCircleIdx is not None and (x,y,r) == self.errCircles[self.changeErrCircleIdx]:
                            # okay to overlap
                            pass
                        else:
                            print("circles are overlapping")
                            break
                else:
                    if self.changeErrCircleIdx is not None:
                        self.errCircles[self.changeErrCircleIdx] = self.currentPoint + (self.currentRadius,)
                        print("changed error circle {}".format(self.changeErrCircleIdx))
                        self.changeErrCircleIdx = None
                    else:
                        self.errCircles.append(self.currentPoint + (self.currentRadius,))
                        print("added error circle")
        
        elif event == cv.EVENT_MOUSEMOVE:
            self.currentPoint = (x, y)

        # check to see if the left mouse button was released
        elif event == cv.EVENT_LBUTTONUP:
            pass
        
    def label(self, img, imgName, errCircles=None):
        self.img = img
        self.errCircles = errCircles if errCircles else []

        cv.namedWindow(imgName)
        cv.setMouseCallback(imgName, self.click)
        while True:
            self.draw()
            # display the image and wait for a keypress
            cv.imshow(imgName, self.currentImg)
            key = cv.waitKey(1) & 0xFF

            if key == ord("+"): # arrow up
                print("increased size")
                self.currentRadius += 1

            elif key == ord("-"): # arrow down
                print("decreased size")
                self.currentRadius = max(self.currentRadius-1, 0)

            elif key == ord("r"):
                self.errCircles = self.errCircles[:-1]
                self.changeErrCircleIdx = None

            elif key in map(ord, map(str, range(10))):
                idx = key-48
                if idx < len(self.errCircles):
                    print("changing label {}".format(idx))
                    self.changeErrCircleIdx = idx
                elif idx == len(self.errCircles):
                    self.changeErrCircleIdx = None

            elif key == ord("n"):
                break

            else:
                pass
                #print("pressed {}".format(key))
        cv.destroyAllWindows()
        return self.errCircles

def getImagePaths(path):
    # https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    imagePaths = []
    for f in listdir(path):
        filepath = join(path, f)
        filename, file_extension = os.path.splitext(filepath)
        if file_extension in (".jpg", ".png"):
            imagePaths.append(filepath)

    return imagePaths

def saveLabeledImages(datasetPath, labelFile, labeledImgs):
    labelFile = join(datasetPath, labelFile)
    with open(labelFile, "w") as f:
        for imgFile in labeledImgs:
            if labeledImgs[imgFile]:
                labelsText = ["({},{},{})".format(x,y,r) for x,y,r in labeledImgs[imgFile]]
                f.write("{}:{}\n".format(imgFile, labelsText))

def readLabeledImages(datasetPath, labelFile):
    labelFile = join(datasetPath, labelFile)
    if not os.path.isfile(labelFile):
        with open(labelFile, 'w'): pass
        print("Created label file '{}'".format(labelFile))
    labeledImgs = {}
    with open(labelFile, "r") as f:
        for line in f:
            imgPath, labels = line.split(":")
            labels = labels.strip()[1:-1] # remove []
            labels = labels.split(", ")
            labels = [tuple(map(int, s[2:-2].split(","))) for s in labels]
            labeledImgs[imgPath] = labels

    return labeledImgs

def testFeatureExtractor(featExtClass, datasetPath, labelFile):
    imgPaths = getImagePaths(datasetPath)
    labeledImgs = readLabeledImages(datasetPath, labelFile)
    
    for imgPath in labeledImgs:
        img = cv.imread(imgPath)
        print("Reading {}".format(imgPath))
        if img is None:
            print("'{}' not found, removing from labeled data.".format(imgPath))
            labeledImgs[imgPath] = []
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        labels = labeledImgs[imgPath]
        nFeatures = len(labels)

        #featExt = featExtClass(nFeatures, camera=None, p=0.1, useKernel=False)
        featExt = featExtClass(featureModel=5, camera=None, p=0.01, erosionKernelSize=5, maxIter=4, useKernel=False)
        
        _, points = featExt(gray, img.copy())
        _, points = featExt(gray, img)
        points = points[:nFeatures]

        for i, errCircle in enumerate(labels):
            drawErrorCircle(img, errCircle, i, color=(0, 255, 0))

        classifications = {c: [] for c in labels}
        classifiedPoints = []
        for errCircle in labels:
            pointsInCircle = []
            for p in points:
                # check if inside errCircle    
                x1, y1, r = errCircle
                x2, y2 = p
                d = np.sqrt(pow(x2-x1, 2) + pow(y2-y1, 2))
                if d < r:
                    pointsInCircle.append(p)
                    classifiedPoints.append(p)
            classifications[errCircle] = pointsInCircle
            if len(pointsInCircle) == 1:
                cv.circle(img, pointsInCircle[0], 1, (255, 0, 0), 2)
            elif len(pointsInCircle) > 1:
                for p in pointsInCircle:
                    cv.circle(img, p, 1, (0, 140, 255), 2)

        correctlyClassified = True
        for p in points:
            if p not in classifiedPoints:
                cv.circle(img, p, 1, (0, 0, 255), 2)
                correctlyClassified = False

        if correctlyClassified:
            font = cv.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            thickness = 2
            infoText = "{}/{}".format(len(classifiedPoints), len(points))
            cv.putText(img, infoText, (30, 30), font, fontScale, (0, 255, 0), thickness, cv.LINE_AA)
            cv.imshow("bin", featExt.pHold.img)
            cv.imshow("bin open", featExt.adaOpen.img)
            cv.imshow("image", img)
            cv.waitKey(0)

class MeanShiftAnalyzer:
    def __init__(self):
        cv.imshow("mean shift", np.zeros((10,10)))
        cv.setMouseCallback("mean shift", self._click)
        self.msTrackers = []

    def _click(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            for ms in self.msTrackers:
                if np.linalg.norm([x-ms.center[0], y-ms.center[1]]) < ms.radius:
                    self.msTrackers.remove(ms)
                    break
            else:
                self.msTrackers.append(MeanShiftTracker((x,y), radius=10, maxRadius=90, minRadius=10))

    def update(self, img):
        drawImg = img.copy()
        gray = cv.cvtColor(drawImg, cv.COLOR_BGR2GRAY)

        for ms in self.msTrackers:
            ms.update(gray, maxIter=10, drawImg=drawImg)
        cv.imshow("mean shift", drawImg)

class LightSourceTrackAnalyzer:
    def __init__(self):
        cv.imshow("Light source tracking", np.zeros((10,10)))
        cv.setMouseCallback("Light source tracking", self._click)
        self.trackers = []

    def _click(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            for tr in self.trackers:
                if np.linalg.norm([x-tr.center[0], y-tr.center[1]]) < tr.patchRadius:
                    self.trackers.remove(tr)
                    break
            else:
                self.trackers.append(LightSourceTracker((x,y), radius=10, maxPatchRadius=50, minPatchRadius=7))

    def update(self, img):
        drawImg = img.copy()
        gray = cv.cvtColor(drawImg, cv.COLOR_BGR2GRAY)

        for tr in self.trackers:
            tr.update(gray, drawImg=drawImg)
        cv.imshow("Light source tracking", drawImg)

class RCFAnalyzer:
    def __init__(self):
        cv.imshow("RCF analyzer", np.zeros((10,10)))
        cv.setMouseCallback("RCF analyzer", self._click)
        self.coordinates = []
        self.drawImg = None

    def _click(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.coordinates.append((x,y))
            self._draw()

    def _draw(self):
        if self.drawImg is not None:
            drawImg = self.drawImg.copy()
            gray = cv.cvtColor(drawImg, cv.COLOR_BGR2GRAY)
            #plt.clf()
            for center in self.coordinates:
                r, rcfs = RCFS(center, rStart=1, gray=gray, drawImg=drawImg)
                #plt.plot(rcfs)

            #plt.pause(0.001)
            cv.imshow("RCF analyzer", drawImg)

    def update(self, img):
        self.coordinates = []
        self.drawImg = img.copy()
        cv.imshow("RCF analyzer", self.drawImg)

class PeakAnalyzer:
    def __init__(self):
        self.coordinates = []
        self.windowRadius = 3
        self.currentPos = (0,0)
        self.drawImg = None
        self.p = 0.95


        cv.imshow("Peak analyzer p={}".format(self.p), np.zeros((10,10)))
        cv.setMouseCallback("Peak analyzer p={}".format(self.p), self._click)

    def _click(self, event, x, y, flags, param):
        self.currentPos = (x,y)
        if event == cv.EVENT_LBUTTONDOWN:
            drawImg = self.drawImg.copy()
            gray = cv.cvtColor(drawImg, cv.COLOR_BGR2GRAY)

            kernel = circularKernel(int(self.windowRadius*2+1))
            mask = np.zeros(gray.shape, dtype=np.uint8)

            mask[self.currentPos[1]-self.windowRadius:self.currentPos[1]+self.windowRadius+1, 
                 self.currentPos[0]-self.windowRadius:self.currentPos[0]+self.windowRadius+1] = kernel
            
            grayMasked = cv.bitwise_and(gray, gray, mask=mask)
            maxIdx = np.unravel_index(np.argmax(grayMasked), gray.shape)
            maxPos = maxIdx[1], maxIdx[0]
            self.coordinates.append(maxPos)

        self._draw()

    def _draw(self):
        if self.drawImg is not None:
            drawImg = self.drawImg.copy()

            gray = cv.cvtColor(drawImg, cv.COLOR_BGR2GRAY)

            for center in self.coordinates:
                cnt = findPeakContourAt(gray, center, p=self.p)
                cv.circle(drawImg, center, 1, (255,0,255), 1)
                cv.drawContours(drawImg, [cnt], 0, (255,0,0), 1)

            cv.circle(drawImg, self.currentPos, self.windowRadius, (0,255,0), 1)
            cv.imshow("Peak analyzer p={}".format(self.p), drawImg)

    def update(self, img):
        self.coordinates = []
        self.drawImg = img.copy()
        cv.imshow("Peak analyzer p={}".format(self.p), self.drawImg)

class ImageAnalyzeNode:
    def __init__(self, cameraYamlPath=None):
        if cameraYamlPath:
            self.cameraInfoMsg = readCameraYaml(cameraYamlPath)
            projectionMatrix = np.array(self.cameraInfoMsg.P, dtype=np.float32).reshape((3,4))[:,:3]
            
            self.camera = Camera(cameraMatrix=projectionMatrix, 
                                distCoeffs=np.zeros((1,4), dtype=np.float32),
                                projectionMatrix=None,
                                resolution=(self.cameraInfoMsg.height, self.cameraInfoMsg.width))

            #self.camera = Camera(cameraMatrix=np.array(self.cameraInfoMsg.K, dtype=np.float32).reshape((3,3)), 
            #                    distCoeffs=np.array(self.cameraInfoMsg.D, dtype=np.float32),
            #                    projectionMatrix=projectionMatrix,
            #                    resolution=(self.cameraInfoMsg.height, self.cameraInfoMsg.width))
        else:
            self.cameraInfoMsg = None
            self.camera = Camera(cameraMatrix=np.eye(3, dtype=np.float32).reshape((3,3)), 
                                distCoeffs=np.zeros((1,4), dtype=np.float32),
                                resolution=None)

        self.associatedImgPointsMsg = None
        self.analyzeThreshold = 150
        self.bridge = CvBridge()

        self._lButtonPressed = False
        self._coordinatePressed = (0, 0) # (x, y)
        self._roi = (0, 0, 0) # (x, y, size)

        self.rawImgPublisher = rospy.Publisher('lolo_camera/image_raw', Image, queue_size=1)
        self.rectImgPublisher = rospy.Publisher('lolo_camera/image_rect_color', Image, queue_size=1)
        self.camInfoPublisher = rospy.Publisher('lolo_camera/camera_info', CameraInfo, queue_size=1)

        self.associatedImagePointsSubscriber = rospy.Subscriber('lolo_camera/associated_image_points', 
                                                                PoseArray, 
                                                                self._associatedImagePointsCallback)
        
    def _publish(self, img, imgRect):
        """Publish raw image, rect image and camera_info"""

        self.rawImgPublisher.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
        self.rectImgPublisher.publish(self.bridge.cv2_to_imgmsg(imgRect, "bgr8"))
        if self.cameraInfoMsg:
            self.camInfoPublisher.publish(self.cameraInfoMsg)
        else:
            msg = CameraInfo()
            msg.width = img.shape[1]
            msg.height = img.shape[0]
            msg.K = np.eye(3, dtype=np.float32).ravel()
            msg.D = np.zeros((1, 4), dtype=np.float32)
            msg.R = np.eye(3, dtype=np.float32).ravel()
            msg.P = np.eye(3, 4, dtype=np.float32).ravel()
            self.camInfoPublisher.publish(msg)

    def _associatedImagePointsCallback(self, msg):
        self.associatedImgPointsMsg = msg

    def _undistortErrCircles(self, errCircles):
        imgPoints = np.array([(x,y) for x, y, _ in errCircles], dtype=np.float32)
        imgPoints = imgPoints.reshape((len(imgPoints), 1, 2))

        imagePointsUndist = cv.undistortPoints(imgPoints, self.camera.cameraMatrix, self.camera.distCoeffs, P=self.camera.projectionMatrix)
        imagePointsUndist = imagePointsUndist.reshape((len(imagePointsUndist), 2))

        errCirclesUndist = []
        for imgPointUndist, errCircle in zip(imagePointsUndist, errCircles):
            errCirclesUndist.append([int(round(imgPointUndist[0])), 
                                     int(round(imgPointUndist[1])), 
                                     errCircle[2]])

        return errCirclesUndist

    def _click(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self._lButtonPressed = True
            self._coordinatePressed = (x, y)
            
        elif event == cv.EVENT_MOUSEMOVE:
            if self._lButtonPressed is True:
                size = abs(y-self._coordinatePressed[1])
                if size > 0:
                    self._roi = (x, y, size)
                else:
                    self._roi = (0, 0, 0)
                    self._coordinatePressed = (0, 0)          

        # check to see if the left mouse button was released
        elif event == cv.EVENT_LBUTTONUP:
            self._lButtonPressed = False
            size = abs(y-self._coordinatePressed[1])
            if size > 0:
                self._roi = (x, y, size)
            else:
                self._roi = (0, 0, 0)
                self._coordinatePressed = (0, 0)

    def _testProcessImage(self, gray):
        from lolo_perception.feature_extraction import AdaptiveThreshold2, contourCentroid, RCF, drawInfo
        from matplotlib.pyplot import cm

        adaThres = AdaptiveThreshold2(5)
        blur = cv.GaussianBlur(gray.copy(), (5,5),0)
        imgTemp = adaThres.process(blur.copy())
        _, contours, hier = cv.findContours(imgTemp, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        plt.gcf()
        plt.cla()
        imgColor = cv.cvtColor(blur, cv.COLOR_GRAY2BGR)

        # remove areas < 5 pixels
        contours = [cnt for cnt in contours if cv.contourArea(cnt) >= 5]
        # sort by largest area first
        contours = sorted(contours, key=cv.contourArea, reverse=True)

        color = iter(cm.rainbow(np.linspace(0, 1, len(contours))))
        for i, cnt in enumerate(contours):
            centerX, centerY = contourCentroid(cnt)
            _, radius = cv.minEnclosingCircle(cnt)
            rStart = 1 ###
            rEnd = 5
            rcfs = RCF(centerX, centerY, rStart, rEnd, blur, drawImg=imgColor)
            #rcfsOffset = RCF(centerX+5, centerY+5, rStart, rEnd, blur)
            rcf = round(rcfs[-1], 4)

            c = next(color)
            drawInfo(imgColor, (centerX+20, centerY-20), "RCF {}: {}".format(i, rcf), color=(255, 0, 0))
            cv.circle(imgColor, (centerX, centerY), 2, (255, 0, 0), 2)
            cv.circle(imgColor, (centerX, centerY), int(rStart), (255, 0, 0), 1)
            cv.circle(imgColor, (centerX, centerY), int(rEnd), (255, 0, 0), 1)
            plt.plot(rcfs, c=c)
            #plt.plot(rcfsOffset, c=c, marker="^")
        legendLs = []
        for i in range(len(contours)):
            legendLs.extend([i])
        plt.legend(legendLs)


        img = imgColor
        cv.imshow("test", img)
        return img
        kernelSize = 5
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernelSize,kernelSize)) 
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=1)
        return img

    def _analyzeImage(self, imgRect, nFeatures, mask=None):
        gray = cv.cvtColor(imgRect, cv.COLOR_BGR2GRAY)
        cv.imshow("gray", gray)
        return imgRect
                
        
        if mask is None:
            mask = np.ones(imgRect.shape[:2], dtype=np.uint8)

        #nFeatures = 5 ###########################################
        if nFeatures == 0:
            nFeatures = 5

        gray = cv.cvtColor(imgRect, cv.COLOR_BGR2GRAY)
        grayMasked = cv.bitwise_and(gray, gray, mask=mask)
        #testImg = self._testProcessImage(gray.copy())
        #cv.imshow("gray", grayMasked)
        blur3 = cv.GaussianBlur(gray.copy(), (3,3),0)
        blurMasked3 = cv.bitwise_and(blur3, blur3, mask=mask)
        #blur5 = cv.GaussianBlur(gray.copy(), (5,5),0)
        #blurMasked5 = cv.bitwise_and(blur3, blur5, mask=mask)
        #blur11 = cv.GaussianBlur(gray.copy(), (11,11),0)
        #blurMasked11 = cv.bitwise_and(blur11, blur11, mask=mask)
        #cv.imshow("blur", blurMasked)

        

        """
        # peaks dilation
        peaksDilationImg = blurMasked3.copy()
        
        kernelSize = 21
        #kernel1 = circularKernel(kernelSize)
        #peaksDilation = findPeaksDilation(peaksDilationImg, kernel1)
        #print("Freq peaks dilation", 1/elapsed)
        #adaThresh = AdaptiveThresholdPeak(nFeatures, thresholdType=cv.THRESH_BINARY)

        #ret, peaksDilation = cv.threshold(peaksDilation, self.analyzeThreshold, 256, cv.THRESH_TOZERO)
        #cv.imshow("peaks dilation kernel {}".format(kernelSize), peaksDilation)

        # max peak contour
        start = time.time()
        img = peaksDilationImg.copy()
        drawImg = cv.cvtColor(peaksDilationImg.copy(), cv.COLOR_GRAY2BGR)
        #peakCenters, peakContours = findNPeaks(img, kernelSize=kernelSize, p=.95, n=nFeatures, margin=5, drawImg=drawImg)
        elapsed = time.time()-start
        print("Freq max peaks:", 1/elapsed)
        #cv.imshow("max peak contour", drawImg)
        """ 
        """
        # Gradient

        gradFeatExt = GradientFeatureExtractor(None, nFeatures, kernelSize=3)
        gradImg = gradFeatExt(grayMasked)
        
        maxGrad = np.max(gradImg)
        gradImg = cv.GaussianBlur(gradImg, (5,5),0)
        ret, gradImg = cv.threshold(gradImg, self.analyzeThreshold, 256, cv.THRESH_BINARY)
        #cv.imshow("gradient3", gradImg)

        gradFeatExt = GradientFeatureExtractor(None, nFeatures, kernelSize=5)
        gradImg = gradFeatExt(grayMasked)
        ret, gradImg = cv.threshold(gradImg, self.analyzeThreshold, 256, cv.THRESH_BINARY)
        #cv.imshow("gradient5", gradImg)
        """
        
        # Adaptive threshold (HATS)
        minArea = 5
        adaThresh = AdaptiveThreshold2(nFeatures, minArea=minArea, marginPercentage=0.0, thresholdType=cv.THRESH_BINARY)
        adaThreshImg = blurMasked3.copy()
        drawImg = cv.cvtColor(adaThreshImg, cv.COLOR_GRAY2BGR)
        adaThresh.process(adaThreshImg, drawImg=drawImg)
        adaThreshImg = adaThresh.img
        cv.imshow("adaptive threshold min area: {}".format(minArea), adaThreshImg)

        cv.imshow("adaptive threshold min area: {} candidates".format(minArea), drawImg)
        
        """
        ret, data = cv.threshold(peaksDilation, self.analyzeThreshold, 256, cv.THRESH_TOZERO)
        _, contours, hier = cv.findContours(data, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        print("Ncontours", len(contours))
        cv.imshow("peaks dilation blured threshold", data)

        adaThresh = AdaptiveThreshold2(nFeatures, minArea=0, marginPercentage=0.0, thresholdType=cv.THRESH_TOZERO)
        adaThresh.process(peaksDilation)
        peaksDilation = adaThresh.img
        #cv.imshow("peaks dilation blured adaptive threshold", peaksDilation)
        """

        """
        # some other threshold
        data = grayMasked.copy()
        neighborhood_size = 3
        threshold = 10
        data_max = filters.maximum_filter(data, neighborhood_size)

        indices = np.where(data == data_max)
        maxima = np.zeros(blurMasked.shape, np.uint8)
        maxima[indices] = blurMasked[indices]
        
        data_min = filters.minimum_filter(data, neighborhood_size)
        #diff = ((data_max - data_min) > 3)
        diff = (data_min > data_max*0.95)
        maxima[diff == 0] = 0
        maximaOrid = maxima
        cv.imshow("some other threshold", maximaOrid)

        # same but different
        #data = blurMasked.copy()
        data = grayMasked.copy()
        data = AdaptiveThreshold2(nFeatures, minArea=3, marginPercentage=0.05, thresholdType=cv.THRESH_TOZERO).process(data)

        cv.imshow("adaptive threshold min area 3 margin 0.05", data)
        """

        """
        # find best point candidates
        start = time.time()
        data = blurMasked3.copy()
        #minArea = 3
        #minRatio = 0.3
        #marginP = 0
        #adaExt = AdaptiveThreshold2(nFeatures, minArea=minArea, marginPercentage=marginP, minRatio=minRatio, thresholdType=cv.THRESH_BINARY)
        #lightCndidates = adaExt.process(data)
        #data = adaExt.img



        cv.imshow("adaptive feature thingy", data)
        _, contours, hier = cv.findContours(data, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        #lightCndidates = [contourCentroid(cnt) for cnt in contours]
        
        data = cv.cvtColor(data, cv.COLOR_GRAY2BGR)
        for cnt, cent in zip(contours, lightCndidates):
            cv.drawContours(data, [cnt], 0, (255,0,0), 1)
            cv.circle(data, cent, 1, (255,0,0), 1)

        from lolo_perception.pose_estimation import DSPoseEstimator
        from lolo_perception.camera_model import usbCamera720p
        from lolo_perception.feature_model import bigPrototype5
        from lolo_perception.perception_utils import plotPoints, plotAxis
        camera = usbCamera720p
        featureModel = bigPrototype5
        poseEst = DSPoseEstimator(camera, 
                                  featureModel=featureModel,
                                  ignoreRoll=False, 
                                  ignorePitch=False, 
                                  flag=cv.SOLVEPNP_EPNP,#cv.SOLVEPNP_ITERATIVE, 
                                  calcCovariance=False)

        
        lightCndidates.sort(key=lambda p: p[1], reverse=True) # largest y coordinate probably
        maxAdditionalCandidates = 0
        lightCndidates = lightCndidates[:nFeatures+maxAdditionalCandidates] # to reduce computation

        if len(lightCndidates) >= nFeatures:
            ret = poseEst.findBestImagePoints(lightCndidates)
            if ret is not None:
                bestImagePoints, bestTranslationVec, bestRotationVec, bestRms = ret
                plotPoints(data, bestImagePoints, color=(255,0,255), radius=5)
                plotAxis(data, bestTranslationVec, bestRotationVec, camera, np.zeros((3,3)), 0.43)
            else:
                print("noenoeanoaeon")
        else:
            print("not enough features detected")
        elapsed = time.time()-start
        print("Pose est freq {} features: {}".format(len(lightCndidates), 1/elapsed))
        cv.imshow("light candidates", data)
        """
        
        """
        # peak finding
        start = time.time()
        data = grayMasked.copy()
        ret, data = cv.threshold(grayMasked, self.analyzeThreshold, 256, cv.THRESH_TOZERO)
        maximaPrev = None
        for k in (7,):# 5, 7, 9, 11, 15, 21):
            data_max = cv.morphologyEx(data, cv.MORPH_DILATE, np.ones((k,k)))
            #data_min = cv.morphologyEx(data, cv.MORPH_ERODE, np.ones((k,k)))
            diff = (data > data_max*0.999)

            maxima = data.copy()
            maxima[diff == 0] = 0
            
            if maximaPrev is not None:
                indices = np.where(maxima != maximaPrev)
                maxima = np.zeros(data.shape, np.uint8)
                maxima[indices] = data[indices]
            maximaPrev = maxima

        #cv.imshow("data max", data_max)
        cv.imshow("some other thres again isch", maxima)
        elapsed = time.time()-start
        print("Freq", 1/elapsed)
        """
        """
        # max/mean shift
        start = time.time()

        maxShiftImg = grayMasked.copy()
        threshold = np.max(maxShiftImg)*0.8
        _, maxShiftImg = cv.threshold(maxShiftImg, threshold, 256, cv.THRESH_TOZERO)
        cv.imshow("thresholded maxShift", maxShiftImg)
        drawImg = cv.cvtColor(maxShiftImg.copy(), cv.COLOR_GRAY2BGR)
        drawImg2 = cv.cvtColor(maxShiftImg.copy(), cv.COLOR_GRAY2BGR)
        drawImg = drawImg2

        h, w = gray.shape
        radius = 5
        #dilated = cv.dilate(grayMasked, circularKernel(15))
        stride = radius*3
        size = radius*2 +1
        kernel = circularKernel(size)
        centerVotes = {}
        maxIter = stride/radius*2 -1
        for i in range(radius, w-radius, stride):
            for j in range(radius, h-radius, stride):
                mask = maxShiftImg[j-radius:j+radius+1, i-radius:i+radius+1]
                mask = cv.bitwise_and(mask, mask, mask=kernel)
                if np.max(mask) > 0:
                    #i, j = 684, 548
                    cv.circle(drawImg2, (i,j), radius, (50, 50, 50), 1)
                    
                    center, patch, iterations = maxShift(maxShiftImg, (i, j), kernel, maxIter=maxIter, drawImg=drawImg)
                    #center, patch, iterations = meanShift(maxShiftImg, center, kernel, maxIter=2, drawImg=drawImg)
                    
                    if iterations < maxIter:
                        # score is the mean of the final patch
                        score = gray[center[1], center[0]]
                        if score > 0:
                            centerVotes[center] = score
                    #break

        centers = centerVotes.keys()
        centers.sort(key=lambda c: centerVotes[c], reverse=True)
        if centers:
            idx = min(len(centers)-1, nFeatures-1)
            intensityLowest = centerVotes[centers[idx]]
            threshold = intensityLowest*0.8
        
        for i, center in enumerate(centers):
            #if i >= nFeatures and centerVotes[center] < threshold:
            #    break
            cv.circle(drawImg2, center, 0, (255, 0, 255), 2)
            drawInfo(drawImg2, (center[0]+13, center[1]-13), str(round(centerVotes[center],2)), color=(255, 0, 255), fontScale=0.3, thickness=1)

        elapsed = time.time() - start
        print("Freq maxShift: {}".format(1./elapsed))
        cv.imshow("max shift", drawImg2)
        # /max shift
        """
        """
        # max peak kernel est with peaks with dilation
        start = time.time()
        maxPeakImg = grayMasked.copy()
        maxPeakImgColor = cv.cvtColor(maxPeakImg, cv.COLOR_GRAY2BGR)

        _, peaks = findMaxPeaks(maxPeakImg, .95, drawImg=maxPeakImgColor)
        peaks.sort(key=lambda p: cv.minEnclosingCircle(p)[1], reverse=True)
        if peaks:
            for p in peaks:
                maxCircleRadius = cv.minEnclosingCircle(peaks[0])[1]
                center = contourCentroid(p)
                cv.circle(maxPeakImgColor, center, int(maxCircleRadius*10), (255,0,0), 1)
                #print(maxCircleRadius)
            #maxCircleRadius 
            maxCircleRadius = int(maxCircleRadius)
            maxCircleRadius = max(maxCircleRadius, 3)
            maxCircleRadius = min(maxCircleRadius, 11)
            cv.imshow("max peak", maxPeakImgColor)

            peaksDilation = maxPeakImg.copy()
            p = 0.9
            if maxCircleRadius < 5:
                p = 0.6
            #print(p)
            threshold = int(np.max(peaksDilation)*p)
            ret, peaksDilation = cv.threshold(peaksDilation, threshold, 256, cv.THRESH_TOZERO)
            kernel = circularKernel(int(maxCircleRadius)*2+1)
            peaksDilation = findPeaksDilation(peaksDilation, kernel)

            peaksDilation = cv.morphologyEx(peaksDilation, cv.MORPH_DILATE, kernel, iterations=1)#cv.dilate(peaksDilation, kernel)

            elapsed = time.time() - start
            #print("Freq peaks dilation: {}".format(1./elapsed))
            cv.imshow("peaks dilation", peaksDilation)
            # / peaks with dilation
        """
        return grayMasked

        
        # adaptive threshold
        start = time.time()
        adaThresh = maximaOrid.copy()
        adaThresh = AdaptiveThreshold2(nFeatures, minArea=1, marginPercentage=0).process(adaThresh)

        _, contours, hier = cv.findContours(adaThresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)

        contourAreas = [cv.contourArea(cnt) for cnt in contours]
        if len(contours) >= nFeatures:
            lightArea = np.mean(contourAreas[:nFeatures])
        elif len(contours) > 0:
            lightArea = cv.contourArea(contours[0])
        else:
            lightArea = 0
        lightRadius = np.sqrt(lightArea/np.pi)

        elapsed = time.time() - start
        #print("Freq adaThresh: {}".format(1./elapsed))
        
        #cv.imshow("adaptive threshold min area 1", adaThresh)

        # /adaptive threshold


        return grayMasked
        #kernelType = cv.MORPH_RECT #cv.MORPH_ELLIPSE
        neighborhood_size = 20
        threshold = 50
        kernel = circularKernel(5)
        data = cv.dilate(grayMasked, kernel, iterations=1)
        data_max = filters.maximum_filter(data, neighborhood_size)
        maxima = (data == data_max)
        data_min = filters.minimum_filter(data, neighborhood_size)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0

        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        x, y = [], []
        for dy,dx in slices:
            x_center = (dx.start + dx.stop - 1)/2
            x.append(x_center)
            y_center = (dy.start + dy.stop - 1)/2    
            y.append(y_center)
            cv.circle(imgRect, (x_center,y_center), 1, (0, 0, 255), 2)
        
        cv.imshow("local maxima", imgRect)
        
        # gradient
        kernelGrad = circularKernel(5) #cv.getStructuringElement(kernelType, (15,15))
        morphImg = grayMasked
        eroded = cv.erode(morphImg, kernelGrad, iterations=1)
        dilated = cv.dilate(morphImg, kernelGrad, iterations=1)
        indices = np.where(eroded == dilated)
        morphImg = np.zeros(gray.shape, np.uint8)
        morphImg[indices] = gray[indices]

        #morphGrad = cv.morphologyEx(grayMasked, cv.MORPH_GRADIENT, kernelGrad)
        cv.imshow("morph halabalula", morphImg)

        gradFeatExt = GradientFeatureExtractor(None, nFeatures, kernelSize=5)
        
        gradImg = gradFeatExt(gray)
        cv.imshow("gradient", gradImg)
        gradImg = cv.dilate(gradImg, kernelGrad, iterations=1)
        gradImg = AdaptiveThreshold2(nFeatures, minArea=2, thresholdType=cv.THRESH_TOZERO).process(gradImg)
        gradImg = AdaptiveOpen(nFeatures=nFeatures, kernelSize=3, kernelType=cv.MORPH_ELLIPSE, startIterations=1, maxIter=10).process(gradImg)
        #gradImg = removeNeighbouringContoursFromImg(gradImg, minDist=10, key=cv.contourArea)
        

        gradImg = cv.bitwise_and(gradImg, gradImg, mask=mask)
        
        #gradImg = cv.GaussianBlur(gradImg, (5,5),0)
        gradImg = cv.medianBlur(gradImg, 5)
        
        
        #gradImg = findPeaksDilation(gradImg, 50)
        #
        #gradImg = cv.morphologyEx(gradImg, cv.MORPH_OPEN, kernelGrad, iterations=1)
        
        
        #gradImg = cv.morphologyEx(gradImg, cv.MORPH_OPEN, kernelGrad, iterations=1)
        #gradImg = cv.dilate(gradImg, kernelGrad, iterations=1)
        


        return grayMasked

    def analyzeImageOld(self, imgName, frame, labeledImgs):
        labeler = ImageLabeler()
        imgColorRaw = frame.copy()
        imgRectOrig = self.camera.undistortImage(imgColorRaw).astype(np.uint8)

        self.associatedImgPointsMsg = None
        histogramActive = False
        # publish once
        self._publish(imgColorRaw, imgRectOrig)
        while not rospy.is_shutdown():
            imgRect = imgRectOrig.copy()

            # get associated points from subscription
            #rospy.sleep(0.1)
            points = []
            if self.associatedImgPointsMsg:
                points = msgToImagePoints(self.associatedImgPointsMsg)

            # plot predictions and error circles
            tmpFrame = imgRect.copy()
            for p in points:
                cv.circle(tmpFrame, p, 2, (255, 0, 0), 2)

            # get error circles
            errCircles = None
            if imgName in labeledImgs:
                errCircles = labeledImgs[imgName]
            if errCircles:
                for j, ec in enumerate(self._undistortErrCircles(errCircles)):
                    drawErrorCircle(tmpFrame, ec, j, (0, 255, 0))

            # draw roi if it exists
            mask = None
            if self._roi[2] > 0:
                x, y = self._coordinatePressed 
                size = self._roi[2]
                mask = np.zeros(tmpFrame.shape[:2], dtype=np.uint8)
                mask[y-size:y+size, x-size:x+size] = 1
                tmpFrame = cv.bitwise_and(tmpFrame, tmpFrame, mask=mask)
                #imgRect = cv.bitwise_and(imgRect, imgRect, mask=mask)
                cv.circle(tmpFrame, self._coordinatePressed, self._roi[2], (0,255,0), 2)

            cv.imshow("frame", tmpFrame)
            cv.setWindowTitle("frame", imgName)
            cv.setMouseCallback("frame", self._click)

            key = cv.waitKey(1) & 0xFF

            nFeatures = 0
            if errCircles: nFeatures = len(errCircles)
            analyzeImg = self._analyzeImage(imgRect, nFeatures, mask=mask)

            if histogramActive:
                plt.pause(0.00001)

            if key == 0xFF:# ord("p"):
                continue
            elif key == ord("p"):
                print("publishing")
                # publish raw image, rect image and camera_info
                self._publish(imgColorRaw, imgRect)
            elif key == ord('q'):
                return ord("q")
            elif key == ord('h'):
                self._histogram(analyzeImg)
                histogramActive = True
            elif key == ord("+"):
                self.analyzeThreshold += 1
                print("Threshold: {}".format(self.analyzeThreshold))
            elif key == ord("-"):
                self.analyzeThreshold -= 1
                print("Threshold: {}".format(self.analyzeThreshold))
            elif key == ord("r"):
                # set region of interest
                pass
            elif key == ord("l"):
                labels = labeler.label(frame, imgName, errCircles=errCircles)
                if labels:
                    #labeledImgs[join(datasetPath, imgName)] = labels
                    labeledImgs[imgName] = labels
                    cv.imwrite(join(datasetPath, imgName), frame)
                    print("Saved image frame '{}'".format(imgName))
                else:
                    print("No labels registered, image not saved")
            else:
                return key

    def analyzeImage(self, imgName, imgColorRaw, imgRectOrig, labeledImgs, analyzeImg=True):

        self.associatedImgPointsMsg = None
        histogramActive = False
        # publish once
        self._publish(imgColorRaw, imgRectOrig)
        while not rospy.is_shutdown():
            imgRect = imgRectOrig.copy()

            # get associated points from subscription
            #rospy.sleep(0.1)
            points = []
            if self.associatedImgPointsMsg:
                points = msgToImagePoints(self.associatedImgPointsMsg)

            # plot predictions and error circles
            tmpFrame = imgRect.copy()
            for p in points:
                cv.circle(tmpFrame, p, 2, (255, 0, 0), 2)

            # get error circles
            errCircles = None
            if imgName in labeledImgs:
                errCircles = labeledImgs[imgName]
            if errCircles:
                for j, ec in enumerate(self._undistortErrCircles(errCircles)):
                    drawErrorCircle(tmpFrame, ec, j, (0, 255, 0))

            # draw roi if it exists
            mask = None
            if self._roi[2] > 0:
                x, y = self._coordinatePressed 
                size = self._roi[2]
                mask = np.zeros(tmpFrame.shape[:2], dtype=np.uint8)
                mask[y-size:y+size, x-size:x+size] = 1
                tmpFrame = cv.bitwise_and(tmpFrame, tmpFrame, mask=mask)
                #imgRect = cv.bitwise_and(imgRect, imgRect, mask=mask)
                cv.circle(tmpFrame, self._coordinatePressed, self._roi[2], (0,255,0), 2)

            cv.imshow("frame", tmpFrame)
            cv.setWindowTitle("frame", imgName)
            cv.setMouseCallback("frame", self._click)

            key = cv.waitKey(1) & 0xFF

            nFeatures = 0
            if errCircles: nFeatures = len(errCircles)
            analyzeImg = self._analyzeImage(imgRect, nFeatures, mask=mask)

            if histogramActive:
                plt.pause(0.00001)

            if key == ord("q"):
                break
            elif key == 0xFF:# ord("p"):
                continue
            elif key == ord("p"):
                print("publishing")
                # publish raw image, rect image and camera_info
                self._publish(imgColorRaw, imgRect)
            elif key == ord('h'):
                self._histogram(analyzeImg)
                histogramActive = True
            elif key == ord("+"):
                self.analyzeThreshold += 1
                print("Threshold: {}".format(self.analyzeThreshold))
            elif key == ord("-"):
                self.analyzeThreshold -= 1
                print("Threshold: {}".format(self.analyzeThreshold))
            elif key == ord("l"):
                labels = labeler.label(frame, imgName, errCircles=errCircles)
                if labels:
                    #labeledImgs[join(datasetPath, imgName)] = labels
                    labeledImgs[imgName] = labels
                    cv.imwrite(join(datasetPath, imgName), frame)
                    print("Saved image frame '{}'".format(imgName))
                else:
                    print("No labels registered, image not saved")
            else:
                break
            
        return key

    def _anaLyzeImages(self, datasetPath, labelFile, imageGenerator, analyzeImages):
        labeledImgs = readLabeledImages(datasetPath, labelFile)

        cv.imshow("frame", np.zeros((10,10), dtype=np.uint8))

        labeler = ImageLabeler()
        lightSourceAnalyzer = LightSourceTrackAnalyzer()
        rcfAnalyzer = RCFAnalyzer()
        peakAnalyzer = PeakAnalyzer()
        pause = True
        for imgName, frame, i in imageGenerator():
            imgColorRaw = frame.copy()
            imgRect = self.camera.undistortImage(imgColorRaw).astype(np.uint8)
            self._publish(imgColorRaw, imgRect)
            print("Frame " + str(i))
            #lightSourceAnalyzer.update(imgRect)
            #rcfAnalyzer.update(imgRect)
            #peakAnalyzer.update(imgRect)
            if analyzeImages:
                key = self.analyzeImage(imgName, imgColorRaw, imgRect, labeledImgs)
            else:
                while True:
                    cv.imshow("frame", imgRect)
                    cv.setWindowTitle("frame", imgName)
                    #cv.setMouseCallback("frame", self._click)
                    key = cv.waitKey(30)
                    if key == 32:
                        pause = not pause
                    if pause == False or key == ord("q"):
                        break

            if key == ord("q"):
                break

        saveLabeledImages(datasetPath, labelFile, labeledImgs)
        cv.destroyAllWindows()

    def _imageDatasetGenerator(self, datasetPath, labelFile, startFrame):
        imgPaths = getImagePaths(datasetPath)
        imgBasenames = [os.path.basename(imgPath) for imgPath in imgPaths]
        labeledImgs = readLabeledImages(datasetPath, labelFile)

        for imgPath in labeledImgs:
            if imgPath not in imgBasenames:
                print("Labeled image '{}' does not exist in the dataset".format(imgPath))

        unLabeledImgPaths = [imgName for imgName in imgBasenames if imgName not in labeledImgs]
        i = 0
        if unLabeledImgPaths:
            
            # First label unlabeled images
            for imgPath in unLabeledImgPaths:
                i += 1
                if i < startFrame:
                    continue
                imgName = os.path.basename(imgPath)
                img = cv.imread(imgPath)
                yield imgName, img, i
        else:
            # Then label already labeled images
            #while not rospy.is_shutdown():
            for imgName in labeledImgs:
                i += 1
                if i < startFrame:
                    continue
                imgPath = join(datasetPath, imgName)
                img = cv.imread(imgPath)
                yield imgName, img, i

    def _videoImageGenerator(self, videoPath, startFrame):
        cap = cv.VideoCapture(videoPath)
        if (cap.isOpened()== False):
            print("Error opening video stream or file")

        i = 0
        while (cap.isOpened() and not rospy.is_shutdown()):
            ret, frame = cap.read()
            if ret == True:
                i += 1
                if i < startFrame:
                    continue
                # unique name for video frame
                videoFile = os.path.splitext(os.path.basename(videoPath))[0]
                imgName = os.path.splitext(videoFile)[0] + "_frame_" + str(i) + ".png"
                yield imgName, frame, i
            else:
                break
        cap.release()

    def _rosbagImageGenerator(self, rosbagPath, imageRawTopic, startFrame):
        bridge = CvBridge()
        bag = rosbag.Bag(rosbagPath)

        i = 0
        for topic, msg, t in bag.read_messages(topics=imageRawTopic):
            i += 1
            if i < startFrame:
                continue
            if rospy.is_shutdown():
                return

            frame = bridge.imgmsg_to_cv2(msg, 'bgr8')

            # unique name for video frame
            rosbagFile = os.path.splitext(os.path.basename(rosbagPath))[0]
            imgName = os.path.splitext(rosbagFile)[0] + "_msg_" + str(i) + ".png"
            yield imgName, frame, i

        if i == 0:
            print("No image messages with topic '{}' found".format(imageRawTopic))


    def analyzeRosbagImages(self, datasetPath, labelFile, rosbagPath, imageRawTopic, startFrame=1, analyzeImages=True):
        self._anaLyzeImages(datasetPath, labelFile, lambda: self._rosbagImageGenerator(rosbagPath, imageRawTopic, startFrame), analyzeImages)

    def analyzeVideoImages(self, datasetPath, labelFile, videoPath, startFrame=1, analyzeImages=True):
        self._anaLyzeImages(datasetPath, labelFile, lambda: self._videoImageGenerator(videoPath, startFrame), analyzeImages)

    def analyzeImageDataset(self, datasetPath, labelFile, startFrame=1, analyzeImages=True):
        self._anaLyzeImages(datasetPath, labelFile, lambda: self._imageDatasetGenerator(datasetPath, labelFile, startFrame), analyzeImages)

if __name__ == "__main__":
    rospy.init_node("image_analyze_node")

    datasetPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "image_dataset")
    labelFile = "labels.txt"

    # single light source
    imgLabelNode = ImageAnalyzeNode("camera_calibration_data/contour.yaml")
    videoPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "test_sessions/FILE0151.MP4")
    #imgLabelNode.analyzeVideoImages(datasetPath, labelFile, videoPath, startFrame=550)

    imgLabelNode = ImageAnalyzeNode("camera_calibration_data/contour.yaml")
    videoPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "test_sessions/171121/171121_straight_test.MP4")
    #videoPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "test_sessions/271121/271121_5planar_1080p.MP4")
    #imgLabelNode.analyzeVideoImages(datasetPath, labelFile, videoPath, startFrame=311)

    imgLabelNode = ImageAnalyzeNode("camera_calibration_data/usb_camera_720p_sim.yaml")
    rosbagFile = "sim_bag.bag"
    rosbagPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), join("rosbags", rosbagFile))
    imgLabelNode.analyzeRosbagImages(datasetPath, labelFile, rosbagPath, "/lolo/sim/camera_aft/image_color", startFrame=1, analyzeImages=False)

    imgLabelNode = ImageAnalyzeNode("camera_calibration_data/usb_camera_720p_8.yaml")
    rosbagFile = "ice.bag"
    rosbagPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), join("rosbags", rosbagFile))
    #imgLabelNode.analyzeRosbagImages(datasetPath, labelFile, rosbagPath, "lolo_camera/image_raw", startFrame=420, analyzeImages=True)

    imgLabelNode = ImageAnalyzeNode()
    #imgLabelNode.analyzeImageDataset(datasetPath, labelFile, startFrame=1)
    
    #testFeatureExtractor(ThresholdFeatureExtractor, datasetPath, labelFile)