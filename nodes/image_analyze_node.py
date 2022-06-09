#!/usr/bin/env python
import sys
from os.path import isfile, join
from os import listdir
import os
import time
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import rospy
import rospkg
import rosbag
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import PoseArray

from lolo_perception.camera_model import Camera
from lolo_perception.perception_ros_utils import readCameraYaml, msgToImagePoints
from lolo_perception.feature_extraction import contourRatio, MeanShiftTracker, LightSourceTracker, RCFS, RCF, findPeakContourAt, circularKernel, LightSourceTrackInitializer, localMax, localMaxSupressed, localMaxSupressed2, localMaxChange, removeContoursOnEdges
from lolo_perception.perception_utils import plotHistogram, imageROI, regionOfInterest

# for _analyzeImage
from lolo_perception.feature_extraction import ModifiedHATS, findNPeaks2, peakThresholdMin, LoG, contourCentroid, AdaptiveThresholdPeak, GradientFeatureExtractor, findAllPeaks, findPeaksDilation, peakThreshold, AdaptiveThreshold2, circularKernel, removeNeighbouringContours, removeNeighbouringContoursFromImg, AdaptiveOpen, maxShift, meanShift, drawInfo, medianContourAreaFromImg, findMaxPeaks, findMaxPeak

import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

def drawErrorCircle(img, errCircle, i, color, font=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1, delta=5):
    (x, y, r) = errCircle
    d = int(1/np.sqrt(2)*r) + delta
    org = (x+d, y+d) # some displacement to make it look good
    cv.putText(img, str(i), org, font, fontScale, color, thickness, cv.LINE_AA)
    cv.circle(img, (x,y), 0, color, 1)
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

class _LightSourceTrackAnalyzer:
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

class LightSourceTrackAnalyzer:
    def __init__(self):
        cv.imshow("Light source tracking", np.zeros((10,10)))
        cv.setMouseCallback("Light source tracking", self._click)
        self.lsTracker = LightSourceTrackInitializer(radius=50, maxPatchRadius=100, minPatchRadius=55, maxMovement=50)
        self._newTrackerCenters = []
        self._gray = None

    def _click(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            for tr in self.lsTracker.trackers:
                if np.linalg.norm([x-tr.center[0], y-tr.center[1]]) < tr.patchRadius:
                    self.lsTracker.trackers.remove(tr)
                    break
            else:
                self._newTrackerCenters.append((x,y))
                #self.lsTracker.trackers.append(LightSourceTracker((x,y), radius=10, maxPatchRadius=50, minPatchRadius=7))

    def update(self, img):
        drawImg = img.copy()
        self._gray = cv.cvtColor(drawImg, cv.COLOR_BGR2GRAY)

        self.lsTracker.update(self._gray, 
                              newTrackerCenters=list(self._newTrackerCenters), 
                              drawImg=drawImg)

        self._newTrackerCenters = []
        cv.imshow("Light source tracking", drawImg)


class RCFAnalyzer:
    def __init__(self):
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

        if not self._initialized:
            cv.setMouseCallback("RCF analyzer", self._click)
            self._initialized = True

class PeakAnalyzer:
    def __init__(self):
        self.coordinates = []
        self.windowRadius = 20
        self.currentPos = (0,0)
        self.drawImg = None
        self.p = 0.975


        #cv.imshow("Peak analyzer p={}".format(self.p), np.zeros((10,10)))
        self._initialized = False

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
                _, grayThreholded = cv.threshold(gray, gray[center[1], center[0]]*self.p, 256, cv.THRESH_TOZERO)
                cnt, _ = findPeakContourAt(grayThreholded, center)
                #cv.circle(drawImg, center, 0, (255,0,0), 1)
                cv.drawContours(drawImg, [cnt], 0, (255,0,0), -1)
                cv.putText(drawImg, "I: {}".format(gray[center[1], center[0]]), (center[0]+20, center[1]-20), 1, 1, color=(255,0,0))

            cv.circle(drawImg, self.currentPos, self.windowRadius, (0,255,0), 1)
            cv.imshow("Peak analyzer p={}".format(self.p), drawImg)

    def update(self, img):
        self.coordinates = []
        self.drawImg = img.copy()

        cv.imshow("Peak analyzer p={}".format(self.p), self.drawImg)

        if not self._initialized:
            cv.setMouseCallback("Peak analyzer p={}".format(self.p), self._click)
            self._initialized = True

class ImageAnalyzeNode:
    def __init__(self, cameraYamlPath=None):
        if cameraYamlPath:
            self.cameraInfoMsg = readCameraYaml(cameraYamlPath)
            projectionMatrix = np.array(self.cameraInfoMsg.P, dtype=np.float32).reshape((3,4))[:,:3]
            #self.camera = Camera(cameraMatrix=projectionMatrix, 
            #                    distCoeffs=np.zeros((1,4), dtype=np.float32),
            #                    projectionMatrix=None,
            #                    resolution=(self.cameraInfoMsg.height, self.cameraInfoMsg.width))

            self.camera = Camera(cameraMatrix=np.array(self.cameraInfoMsg.K, dtype=np.float32).reshape((3,3)), 
                                distCoeffs=np.array(self.cameraInfoMsg.D, dtype=np.float32),
                                projectionMatrix=projectionMatrix,
                                resolution=(self.cameraInfoMsg.height, self.cameraInfoMsg.width))
        else:
            self.cameraInfoMsg = None
            self.camera = Camera(cameraMatrix=np.eye(3, dtype=np.float32).reshape((3,3)), 
                                distCoeffs=np.zeros((1,4), dtype=np.float32),
                                resolution=(1280, 720)) # arbitrary default value

        self.associatedImgPointsMsg = None
        self.analyzeThreshold = 0
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
        _, thresh = cv.threshold(gray, self.analyzeThreshold, 255, cv.THRESH_BINARY)
        img = thresh
        """
        # LoG
        blobRadius = 10
        sigma = (blobRadius-1.0)/3.0
        ksize = int(round(sigma*6))
        if ksize % 2 == 0:
            ksize += 1
        print("Sigma", sigma)
        print("ksize", ksize)
        blurred = cv.GaussianBlur(gray, (ksize,ksize), sigma)
        dst = cv.Laplacian(blurred, ddepth=cv.CV_16S, ksize=3)
        dst = cv.convertScaleAbs(dst, alpha=255./dst.max())
        _, logThresh = cv.threshold(dst, self.analyzeThreshold, 256, cv.THRESH_BINARY)
        cv.imshow("blurred", blurred)
        cv.imshow("log", dst)
        cv.imshow("log thresh", logThresh)
        #dst[np.where(blurred == 255)] = 255
        """
        return img

    def _histogram(self, gray):
        # 1D histogram
        
        hist = cv.calcHist([gray], [0], None, [256], [0,256])
        hist = hist.ravel()
        #hist = - hist # find valleys

        
        # 2D histogram
        xx, yy = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]
        plt.cla()
        #gray = cv.flip(gray, 1)
        gray = cv.flip(gray, 0)
        plt.contour(yy, xx, gray, level=10)

    def _histogram1D(self, imgs):
        ########### plot peaks ##############
        from scipy import signal
        plt.cla()
        N = 100
        for img in imgs:
            hist = cv.calcHist([img], [0], None, [256], [0,256])
            hist = hist.ravel()
            hist = hist/hist[N:].max()
            
           
            plt.plot(hist)
            peaks, _ = signal.find_peaks(hist)


            #for peak in peaks:
            #    if peak >= N:
            #        plt.axvline(peak, ymin=0, ymax=hist[peak]/max(hist[N:]), c="r")
        plt.axvline(self.analyzeThreshold, ymin=0, ymax=1, c="g")
        plt.xlim([N, 256])
        plt.ylim([0, 1])
        #plt.pause(0.0001)
        ########### plot peaks ##############

    def analyzeImage(self, imgName, imgColorRaw, imgRectOrig, labeledImgs, i, analyzeImg=True):

        self.associatedImgPointsMsg = None

        # publish once
        self._publish(imgColorRaw, imgRectOrig)

        imgSize = imgColorRaw.shape[1], imgColorRaw.shape[0]
        if imgSize[0] > 1280:
            scale = 1280./imgSize[0] # percent of original size
            width = int(imgColorRaw.shape[1] * scale)
            height = int(imgColorRaw.shape[0] * scale)
            imgSize = (width, height)

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


            cv.imshow("frame", cv.resize(tmpFrame, imgSize))
            cv.setWindowTitle("frame", imgName)
            cv.setMouseCallback("frame", self._click)

            key = cv.waitKey(1) & 0xFF

            nFeatures = 0
            if errCircles: nFeatures = len(errCircles)
            
            analyzeImg = self._analyzeImage(imgRect, nFeatures, mask=mask)
            
            if self.histogramActive:
                self._histogram1D([analyzeImg, cv.cvtColor(imgRect, cv.COLOR_BGR2GRAY)])
                plt.pause(0.00001)

            cv.imshow("analyzed image", cv.resize(analyzeImg, imgSize))

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
                self.histogramActive = True
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

    def _anaLyzeImages(self, datasetPath, labelFile, imageGenerator, analyzeImages, displayTracking=True, displayPeak=False, displayRCF=False):
        labeledImgs = readLabeledImages(datasetPath, labelFile)

        cv.imshow("frame", np.zeros((10,10), dtype=np.uint8))

        labeler = ImageLabeler()
        #if displayTracking:
        lightSourceAnalyzer = LightSourceTrackAnalyzer()

        #if displayRCF:
        rcfAnalyzer = RCFAnalyzer()
        #if displayPeak:
        peakAnalyzer = PeakAnalyzer()

        self.histogramActive = False
        pause = True

        for imgName, frame, i in imageGenerator():

            imgColorRaw = frame.copy()
            imgRect = self.camera.undistortImage(imgColorRaw).astype(np.uint8)
            
            print("Frame " + str(i))
            if displayTracking:
                lightSourceAnalyzer.update(imgRect)
            if displayRCF:
                rcfAnalyzer.update(imgRect)
            if displayPeak:
                peakAnalyzer.update(imgRect)
            if analyzeImages:
                key = self.analyzeImage(imgName, imgColorRaw, imgRect, labeledImgs, i)
            else:
                self._publish(imgColorRaw, imgRect)
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

    def _testImage(self, imgName, imgColorRaw, imgRectOrig, labeledImgs, i):

        
        tmpFrame = imgRectOrig.copy()
        
        # get error circles
        if imgName not in labeledImgs:
            raise Exception("Image '{}' is not labeled".format(imgName))
        
        errCircles = labeledImgs[imgName]
        for j, ec in enumerate(self._undistortErrCircles(errCircles)):
            drawErrorCircle(tmpFrame, ec, j, (0, 255, 0))

        key = 0xFF
        elapsed = None
        errors = []
        self.associatedImgPointsMsg = None

        # publish once
        self._publish(imgColorRaw, imgRectOrig)
        start = time.clock()
        while not rospy.is_shutdown():

            # get associated points from subscription
            #rospy.sleep(0.1)
            if self.associatedImgPointsMsg:
                if not elapsed:
                    elapsed = time.clock() - start
                points = msgToImagePoints(self.associatedImgPointsMsg)
                
                for p in points:
                    cv.circle(tmpFrame, p, 2, (255, 0, 0), 2)

                errors = [np.linalg.norm([p[0]-e[0], p[1]-e[1]]) for p, e in zip(points, errCircles)]
                # image points found
                self.associatedImgPointsMsg = None
                break

            cv.imshow("frame", tmpFrame)
            cv.imshow("rect", imgRectOrig)
            cv.setWindowTitle("frame", imgName)
            cv.setMouseCallback("frame", self._click)

            key = cv.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == 0xFF:# ord("p"):
                continue
            elif key == ord("p"):
                print("publishing")
                self._publish(imgColorRaw, imgRectOrig)
                start = time.clock()
            else:
                break
            
        return key, elapsed, errors

    def _testFeatureExtractor(self, datasetPath, labelFile, imageGenerator, featureExtractor=None):
        
        from lolo_perception.perception import Perception
        from lolo_perception.feature_model import FeatureModel
        from lolo_perception.camera_model import Camera
        import timeit

        featureModelYamlPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "feature_models/{}".format("donn_light.yaml"))
        featureModel = FeatureModel.fromYaml(featureModelYamlPath)
        featureExtractor = Perception(Camera(self.camera.projectionMatrix, np.zeros((1,4), dtype=np.float32), resolution=self.camera.resolution),
                                      featureModel)
        featureExtractor.maxAdditionalCandidates = 100000

        labeledImgs = readLabeledImages(datasetPath, labelFile)

        cv.imshow("frame", np.zeros((10,10), dtype=np.uint8))

        undistort =  not np.all(self.camera.distCoeffs == 0)

        allErrors = []
        failedImgs = []
        filedNCandidates = []
        nCandidates = []
        peakIterations = []
        poseEstimationAttempts = []

        timeitNumber = 0
        times = [] # total time
        processTimes = [] # image processing time
        undistortTimes = [] # undistortion time
        covTimes = [] # covariance calc time
        
        estDSPose = None

        for imgName, frame, i in imageGenerator():
            imgColorRaw = frame.copy()

            undistortElapsed = 0.1
            if timeitNumber > 0:
                timitFunc = lambda: self.camera.undistortImage(imgColorRaw).astype(np.uint8)
                undistortElapsed = min(timeit.repeat(timitFunc, repeat=timeitNumber, number=1))

            imgRect = self.camera.undistortImage(imgColorRaw).astype(np.uint8)

            if not undistort:
                imgRect = imgColorRaw.copy()

            print("Frame " + str(i))

            tmpFrame = imgRect.copy()

            # for standard gray conversion, coefficients = [0.114, 0.587, 0.299]
            
            B = 0.164
            G = 0.537
            R = 0.299
            colorCoeffs = None#[B,G,R] # Gives blue channel all the weight
            
            key = 0xFF
            if featureExtractor:
                # get error circles
                if imgName not in labeledImgs:
                    raise Exception("Image '{}' is not labeled".format(imgName))
                
                errCircles = labeledImgs[imgName]
                if undistort:
                    errCircles = self._undistortErrCircles(errCircles)
                for j, ec in enumerate(errCircles):
                    drawErrorCircle(tmpFrame, ec, j, (0, 255, 0))

                imgRectROI = imgRect.copy()

                offset = (0, 0)
                if False:
                    # Manual ROI                
                    (x, y, w, h), roiCnt = regionOfInterest([[u,v] for u,v,r in errCircles], 80, 80)
                    x = max(0, x)
                    x = min(imgRect.shape[1]-1, x)
                    y = max(0, y)
                    y = min(imgRect.shape[0]-1, y)
                    offset = (x,y)
                    #roiMask = np.zeros(imgRect.shape[:2], dtype=np.uint8)
                    #cv.drawContours(roiMask, [roiCnt], 0, (255,255,255), -1)
                    #imgRectROI = cv.bitwise_and(imgRect, imgRect, mask=roiMask)

                    imgRectROI = imgRect[y:y+h, x:x+w]
                    cv.drawContours(tmpFrame, [roiCnt], 0, (0,255,0), 1)


                #for i in range(10):
                #start = time.clock()
                #elapsed = time.clock()-start
                elapsed = 0.1
                if timeitNumber > 0:
                    timitFunc = lambda: featureExtractor.estimatePose(imgRectROI, 
                                                                    estDSPose=estDSPose,
                                                                    estCameraPoseVector=None,
                                                                    colorCoeffs=colorCoeffs)
                    elapsed = min(timeit.repeat(timitFunc, repeat=timeitNumber, number=1))
                
                
                dsPose, poseAquired, candidates, processedImg, poseImg = featureExtractor.estimatePose(imgRectROI, 
                                                                                                        estDSPose=None,
                                                                                                        estCameraPoseVector=None,
                                                                                                        colorCoeffs=colorCoeffs,
                                                                                                        calcCovariance=False)
                
                

                cv.imshow("processed image", processedImg)
                cv.imshow("frame", tmpFrame)
                cv.setWindowTitle("frame", imgName)
                key = cv.waitKey(1)

                if estDSPose and not dsPose:
                    start = time.clock()
                    dsPose, poseAquired, candidates, processedImg, poseImg = featureExtractor.estimatePose(imgRectROI, 
                                                                                                        estDSPose=None,
                                                                                                        estCameraPoseVector=None,
                                                                                                        colorCoeffs=colorCoeffs,
                                                                                                        calcCovariance=False)
                    elapsed = time.clock()-start

                    cv.imshow("processed image", processedImg)
                    cv.imshow("frame", tmpFrame)
                    cv.setWindowTitle("frame", imgName)
                    key = cv.waitKey(1)

                attempts = 0
                if dsPose:
                    attempts = dsPose.attempts
                poseEstimationAttempts.append(attempts)
                peakIterations.append(featureExtractor.peakFeatureExtractor.iterations)

                for ls in candidates:
                    ls.center = ls.center[0]+offset[0], ls.center[1]+offset[1]
                
                covElapsed = 0
                if False and dsPose:
                    if dsPose._covariance is not None:
                        raise Exception("Covariance already calculated")
                    
                    covElapsed = 0.1
                    if timeitNumber > 0:
                        timeitFunc = lambda: dsPose.calcCovariance()
                        covElapsed = min(timeit.repeat(timeitFunc, repeat=timeitNumber, number=1))
                    dsPose.calcCovariance()
                    #for ls in dsPose.associatedLightSources:
                    #    ls.center = ls.center[0]+offset[0], ls.center[1]+offset[1]

                #estDSPose = dsPose
                totElapsed = undistortElapsed + elapsed + covElapsed
                times.append(totElapsed)
                processTimes.append(elapsed)
                undistortTimes.append(undistortElapsed)
                covTimes.append(covElapsed)
                nCandidates.append(len(candidates))

                if dsPose:
                    points = [ls.center for ls in dsPose.associatedLightSources]
                    for p in points:
                        cv.circle(tmpFrame, p, 2, (255, 0, 0), 2)
                    
                    errors = [np.linalg.norm([p[0]-e[0], p[1]-e[1]]) for p, e in zip(points, errCircles)]
                    err = sum(errors)/len(errors)
                    allErrors.append(err)
                else:
                    allErrors.append(None)
                    failedImgs.append(i)
                
                if dsPose:
                    for ls in dsPose.associatedLightSources:
                        if candidates.index(ls) > len(errCircles)-1:
                            filedNCandidates.append(i)
                            break
                else:
                    filedNCandidates.append(i)

                allErrorsTemp = [e for e in allErrors if e is not None]
                if allErrorsTemp:
                    avgError = sum(allErrorsTemp)/len(allErrorsTemp)
                    print("Avg err:", avgError)
                print("Avg time:", sum(times)/len(times))
                print("Avg hz:", 1./ (sum(times)/len(times)))
                
            else:
                cv.imshow("frame", imgRect)
                cv.setWindowTitle("frame", imgName)
                key, elapsed, errors = self._testImage(imgName, imgColorRaw, imgRect, labeledImgs, i)
                if elapsed:
                    totElapsed = undistortElapsed + elapsed
                    err = sum(errors)/len(errors)
                    allErrors.append(err)
                    avgError = (avgError*(i-1) + err)/float(i)
                    avgTime = (avgTime*(i-1) + totElapsed)/float(i)
                    print("Avg err:", avgError)
                    print("Avg hz:", 1./avgTime)

            if key == ord("q"):
                break

        print("Failed images ({}): {}".format(len(failedImgs), failedImgs))
        print("Failed n candidates ({}): {}".format(len(filedNCandidates), filedNCandidates))
        
        nImages = i
        imageNumbers = list(range(1, nImages+1))

        figures = []  

        figures.append(plt.figure())
        plt.xlim(0, nImages+1)
        plt.ylabel("Pixel error")
        plt.xlabel("Image number")
        allErrorsTemp = [e if e is not None else avgError for e in allErrors]
        plt.plot(imageNumbers, allErrorsTemp)
        for i,e in enumerate(allErrors):
            if e is None:
                plt.vlines(i, min(allErrorsTemp), max(allErrorsTemp), color="r")
  
        figures.append(plt.figure())
        plt.xlim(0, nImages+1)
        plt.ylabel("Computation time")
        plt.xlabel("Image number")
        plt.plot(imageNumbers, times)
        #plt.plot(imageNumbers, undistortTimes)
        #plt.plot(imageNumbers, processTimes)
        plt.plot(imageNumbers, covTimes)
        #plt.legend(["Total", "Undistortion", "Image processing", "Covariance calculation"])
        
        figures.append(plt.figure())
        plt.xlim(0, nImages+1)
        plt.ylabel("Frequency")
        plt.xlabel("Image number")
        plt.plot(imageNumbers, 1./np.array(times))

        figures.append(plt.figure())
        plt.xlim(0, nImages+1)
        plt.ylabel("Number of detected light sources")
        plt.xlabel("Image number")
        plt.plot(imageNumbers, nCandidates)

        figures.append(plt.figure())
        plt.xlim(0, nImages+1)
        plt.ylabel("Peak iterations")
        plt.xlabel("Image number")
        plt.plot(imageNumbers, peakIterations) 

        figures.append(plt.figure())
        plt.xlim(0, nImages+1)
        plt.ylabel("Pose estimation attempts")
        plt.xlabel("Image number")
        plt.plot(imageNumbers, poseEstimationAttempts)
        plt.show()
        
        print("Saving images")
        for i, fig in enumerate(figures):
            fig.savefig("fig{}.png".format(i), dpi=fig.dpi, format='png')

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
                # TODO: this is the old version, remove png for the old labels
                #imgName = os.path.splitext(videoFile)[0] + "_frame_" + str(i) + ".png"
                imgName = os.path.splitext(videoFile)[0] + "_" + str(i)
                
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
            #if i%2 == 0: # Hey Aldo! Uncomment to skip some images
            #    continue
            if i < startFrame:
                continue
            if rospy.is_shutdown():
                return

            if msg._type == Image._type:
                frame = bridge.imgmsg_to_cv2(msg, 'bgr8')
            elif msg._type == CompressedImage._type:
                arr = np.fromstring(msg.data, np.uint8)
                frame = cv.imdecode(arr, cv.IMREAD_COLOR)
            else:
                raise Exception("Invalid message type '{}'".format(msg._type))

            
            # unique name for video frame
            rosbagFile = os.path.splitext(os.path.basename(rosbagPath))[0]
            imgName = os.path.splitext(rosbagFile)[0] + "_msg_" + str(i) + ".png"
            yield imgName, frame, i

        if i == 0:
            print("No image messages with topic '{}' found".format(imageRawTopic))


    def analyzeRosbagImages(self, datasetPath, labelFile, rosbagPath, imageRawTopic, startFrame=1, analyzeImages=True):
        self._anaLyzeImages(datasetPath, labelFile, lambda: self._rosbagImageGenerator(rosbagPath, imageRawTopic, startFrame), analyzeImages)

    def analyzeVideoImages(self, datasetPath, labelFile, videoPath, startFrame=1, analyzeImages=True, test=False):
        if test is True:
            self._testFeatureExtractor(datasetPath, labelFile, lambda: self._videoImageGenerator(videoPath, startFrame))
        else:
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
    #imgLabelNode.analyzeVideoImages(datasetPath, labelFile, videoPath, startFrame=550, analyzeImages=False)

    imgLabelNode = ImageAnalyzeNode("camera_calibration_data/contour.yaml")
    #videoPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "test_sessions/171121/171121_straight_test.MP4")
    videoPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "test_sessions/171121/171121_straight_test_reversed.mkv")
    #videoPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "test_sessions/171121/171121_angle_test.MP4")
    #videoPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "test_sessions/FILE0197.MP4")
    #videoPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "test_sessions/271121/271121_5planar_1080p.MP4")
    #imgLabelNode.analyzeVideoImages(datasetPath, labelFile, videoPath, startFrame=350, analyzeImages=False)

    # DoNN dataset
    imgLabelNode = ImageAnalyzeNode("camera_calibration_data/donn_camera.yaml")
    videoPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "image_dataset/dataset_recovery/donn.mp4")
    datasetRecovery = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "image_dataset/dataset_recovery")
    #imgLabelNode.analyzeVideoImages(datasetRecovery, "donn.txt", videoPath, startFrame=1, analyzeImages=True, test=False)

    imgLabelNode = ImageAnalyzeNode("camera_calibration_data/usb_camera_720p_sim.yaml")
    rosbagFile = "sim_bag.bag"
    rosbagPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), join("rosbags", rosbagFile))
    #imgLabelNode.analyzeRosbagImages(datasetPath, labelFile, rosbagPath, "/lolo/sim/camera_aft/image_color", startFrame=1000, analyzeImages=False)

    imgLabelNode = ImageAnalyzeNode("camera_calibration_data/usb_camera_720p_8.yaml")
    rosbagFile = "ice.bag"
    rosbagPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), join("rosbags", rosbagFile))
    #imgLabelNode.analyzeRosbagImages(datasetPath, labelFile, rosbagPath, "lolo_camera/image_raw", startFrame=1200, analyzeImages=False)

    # For Aldo
    cameraYaml = "camera_calibration_data/kristineberg.yaml" # In /camera_calibration_data
    
    # Lab test
    #rosbagFile = "2022-06-08-17-44-05_lab_docking_station.bag" # In /rosbags
    #topic = "/csi_cam_1/camera/image_raw/compressed"
    #startFrame = 700
    
    # Seabed 1
    #rosbagFile = "2022-06-09-12-03-13.bag"
    #topic = "/sam/perception/csi_cam_0/camera/image_raw/compressed"
    #startFrame = 7000
    
    # Seabed 2
    rosbagFile = "2022-06-09-12-10-59.bag"
    topic = "/sam/perception/csi_cam_0/camera/image_raw/compressed"
    startFrame = 1

    rosbagPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), join("rosbags", rosbagFile))
    imgLabelNode = ImageAnalyzeNode(cameraYaml)
    imgLabelNode.analyzeRosbagImages(datasetPath, labelFile, rosbagPath, topic, startFrame=startFrame, analyzeImages=False)