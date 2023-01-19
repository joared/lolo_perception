#!/usr/bin/env python
import sys
import os
import time
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import rospy
import rospkg
import rosbag
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import PoseArray
from lolo_perception.msg import FeatureModel as FeatureModelMsg


import lolo_perception.definitions as loloDefs
from lolo_perception.definitions import makeAbsolute
from lolo_perception.utils import Timer
from lolo_perception.camera_model import Camera
from lolo_perception.feature_model import FeatureModel
from lolo_perception.perception_ros_utils import readCameraYaml, yamlToFeatureModelMsg, msgToImagePoints
from lolo_perception.image_processing import contourRatio, LightSourceTrackInitializer, LightSourceTracker, RCFS, RCF, findPeakContourAt, circularKernel, localMax, localMaxSupressed, localMaxSupressed2, localMaxDiff, localMaxDiff2, removeContoursOnEdges
from lolo_perception.perception_utils import plotHistogram, imageROI, regionOfInterest, plotFPS, scaleImageToWidth
from lolo_perception.image_dataset import ImageDataset
from lolo_perception.perception import Perception

from lolo_perception.pose_estimation import calcPoseCovarianceFixedAxis

# for _analyzeImage
from lolo_perception.image_processing import findNPeaks2, peakThresholdMin, contourCentroid, findAllPeaks, findPeaksDilation, peakThreshold, circularKernel, removeNeighbouringContours, removeNeighbouringContoursFromImg, maxShift, meanShift, drawInfo, medianContourAreaFromImg, findMaxPeaks, findMaxPeak

class Trackbar:
    def __init__(self, name, wName, value, maxvalue, minValue=0):
        self.name = name
        self.wName = wName
        self.value = value
        self.maxValue = maxvalue
        self.minValue = minValue
        cv.createTrackbar(name, wName, value, maxvalue, self.setPos)

        self._hasChanged = True

    def hasChanged(self):
        return self._hasChanged

    def setMaxVal(self, value):
        if value != self.maxValue and value >= self.minValue:
            self.maxValue = value
            self._hasChanged = True

    def setPos(self, value):
        if value != self.value:
            if value >= self.minValue:
                self.value = value
            else:
                self.value = self.minValue
            self._hasChanged = True

    def update(self):
        if self._hasChanged:
            # Opencv does not update the trackbar max value unless the position has changed.
            # This is a workaround.
            cv.setTrackbarMax(self.name, self.wName, self.maxValue)
            value = self.value
            cv.setTrackbarPos(self.name, self.wName, 0 if self.value != 0 else 1)
            cv.imshow(self.wName, np.zeros((1,1), dtype=np.uint8))
            cv.setTrackbarPos(self.name, self.wName, value)
            self._hasChanged = False


def drawErrorCircle(img, errCircle, i, color, font=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2, delta=5):
    (x, y, r) = errCircle
    d = int(1/np.sqrt(2)*r) + delta
    org = (x+d, y+d) # some displacement to make it look good
    cv.putText(img, str(i), org, font, fontScale, color, thickness, cv.LINE_AA)
    cv.circle(img, (x,y), 0, color, 1)
    cv.circle(img, (x,y), r, color, 1)

class ImageLabeler:
    # https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
    def __init__(self, debug=False):
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
        cv.putText(img, "+ - increase size", (10,yDisplacement), 1, 1, (255, 255, 255), 1, cv.LINE_AA)
        yDisplacement += dy
        cv.putText(img, "- - decrease size", (10,yDisplacement), 1, 1, (255, 255, 255), 1, cv.LINE_AA)
        yDisplacement += dy
        cv.putText(img, "r - remove last label", (10,yDisplacement), 1, 1, (255, 255, 255), 1, cv.LINE_AA)
        yDisplacement += dy
        n = len(self.errCircles)
        if n == 1:
            cv.putText(img, "0 - change label", (10,yDisplacement), 1, 1, (255, 255, 255), 1, cv.LINE_AA)
            yDisplacement += dy
        elif n > 1:
            cv.putText(img, "(0-{}) - change label".format(n-1), (10,yDisplacement), 1, 1, (255, 255, 255), 1, cv.LINE_AA)
            yDisplacement += dy
        cv.putText(img, "n - next image", (10,yDisplacement), 1, 1, (255, 255, 255), 1, cv.LINE_AA)
        yDisplacement += dy
        cv.putText(img, "q - exit labeling tool", (10,yDisplacement), 1, 1, (255, 255, 255), 1, cv.LINE_AA)
        

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
                            print("Circles are overlapping")
                            break
                else:
                    if self.changeErrCircleIdx is not None:
                        self.errCircles[self.changeErrCircleIdx] = self.currentPoint + (self.currentRadius,)
                        #print("changed error circle {}".format(self.changeErrCircleIdx))
                        self.changeErrCircleIdx = None
                    else:
                        self.errCircles.append(self.currentPoint + (self.currentRadius,))
                        #print("added error circle")
        
        elif event == cv.EVENT_MOUSEMOVE:
            self.currentPoint = (x, y)

        # check to see if the left mouse button was released
        elif event == cv.EVENT_LBUTTONUP:
            pass
        
    def label(self, imgName, img, errCircles=None):
        self.img = img
        self.errCircles = errCircles if errCircles else []

        cv.setMouseCallback(imgName, self.click)
        while True:
            self.draw()

            # display the image and wait for a keypress
            cv.imshow(imgName, self.currentImg)
            key = cv.waitKey(1) & 0xFF

            if key == ord("+"):
                #print("increased size")
                self.currentRadius += 1

            elif key == ord("-"):
                #print("decreased size")
                self.currentRadius = max(self.currentRadius-1, 0)

            elif key == ord("r"):
                self.errCircles = self.errCircles[:-1]
                self.changeErrCircleIdx = None

            elif key in map(ord, map(str, range(10))):
                idx = key-48
                if idx < len(self.errCircles):
                    #print("changing label {}".format(idx))
                    self.changeErrCircleIdx = idx
                elif idx == len(self.errCircles):
                    self.changeErrCircleIdx = None

            elif key in (ord("n"), ord("q")):
                break

            else:
                break
                #print("pressed {}".format(key))
        return key, self.errCircles


def testFeatureExtractor(featExtClass, datasetPath, labelFile):
    
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

class LightSourceTrackAnalyzer:
    def __init__(self):
        cv.imshow("Light source tracking", np.zeros((10,10)))
        cv.setMouseCallback("Light source tracking", self._click)
        self.lsTracker = LightSourceTrackInitializer(radius=50, maxPatchRadius=100, minPatchRadius=55, maxMovement=1000)
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
    def __init__(self, datasetDir):
        datasetDir = makeAbsolute(datasetDir, loloDefs.IMAGE_DATASET_DIR)
        self.dataset = ImageDataset(datasetDir)

        # Load the camera msg and the camera. If the path is relative we use 
        # loloDefs to find the absolute path.
        self.cameraYaml = self.dataset.metadata["camera_yaml"]
        cameraYamlAbs = makeAbsolute(self.cameraYaml, loloDefs.CAMERA_CONFIG_DIR)
        self.cameraInfoMsg = readCameraYaml(cameraYamlAbs)
        self.camera = Camera.fromYaml(cameraYamlAbs)

        # Load the feature model msg and the feature model. If the path is relative we use 
        # loloDefs to find the absolute path.
        self.featureModelYaml = self.dataset.metadata["feature_model_yaml"]
        featureModelYamlAbs = makeAbsolute(self.featureModelYaml, loloDefs.FEATURE_MODEL_CONFIG_DIR)
        self.featureModelMsg = yamlToFeatureModelMsg(featureModelYamlAbs)
        self.featureModel = FeatureModel.fromYaml(featureModelYamlAbs)

        self.associatedImgPointsMsg = None
        self.analyzeThreshold = 0
        self.bridge = CvBridge()

        self.rawImgPublisher = rospy.Publisher('lolo_camera/image_raw', Image, queue_size=1)
        self.rectImgPublisher = rospy.Publisher('lolo_camera/image_rect_color', Image, queue_size=1)
        self.camInfoPublisher = rospy.Publisher('lolo_camera/camera_info', CameraInfo, queue_size=1)
        self.featureModelPublisher = rospy.Publisher('feature_model', FeatureModelMsg, queue_size=1)

        self.associatedImagePointsSubscriber = rospy.Subscriber('lolo_camera/associated_image_points', 
                                                                PoseArray, 
                                                                self._associatedImagePointsCallback)
        
    def _publish(self, img, imgRect):
        """Publish raw image, rect image and camera_info"""

        self.rawImgPublisher.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
        self.rectImgPublisher.publish(self.bridge.cv2_to_imgmsg(imgRect, "bgr8"))
        self.camInfoPublisher.publish(self.cameraInfoMsg)
        self.featureModelPublisher.publish(self.featureModelMsg)
        # if self.cameraInfoMsg:
        #     self.camInfoPublisher.publish(self.cameraInfoMsg)
        # else:
        #     msg = CameraInfo()
        #     msg.width = img.shape[1]
        #     msg.height = img.shape[0]
        #     msg.K = np.eye(3, dtype=np.float32).ravel()
        #     msg.D = np.zeros((1, 4), dtype=np.float32)
        #     msg.R = np.eye(3, dtype=np.float32).ravel()
        #     msg.P = np.eye(3, 4, dtype=np.float32).ravel()
        #     self.camInfoPublisher.publish(msg)

    def _associatedImagePointsCallback(self, msg):
        self.associatedImgPointsMsg = msg

    def _undistortErrCircles(self, errCircles):
        imgPoints = np.array([(x,y) for x, y, _ in errCircles], dtype=np.float32)
        undistImgPoints = self.camera.undistortPoints(imgPoints)
        errCirclesUndist = [(p[0], p[1], errCirc[2]) for p, errCirc in zip(undistImgPoints, errCircles)]
        # imgPoints = imgPoints.reshape((len(imgPoints), 1, 2))

        # imagePointsUndist = cv.undistortPoints(imgPoints, self.camera.cameraMatrix, self.camera.distCoeffs, P=self.camera.projectionMatrix)
        # imagePointsUndist = imagePointsUndist.reshape((len(imagePointsUndist), 2))

        # errCirclesUndist = []
        # for imgPointUndist, errCircle in zip(imagePointsUndist, errCircles):
        #     errCirclesUndist.append([int(round(imgPointUndist[0])), 
        #                              int(round(imgPointUndist[1])), 
        #                              errCircle[2]])

        return errCirclesUndist

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

        trackingConfig = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "tracking_config/{}".format("tracking_configuration_small.yaml"))
        featureModelYamlPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "feature_models/{}".format("donn_light.yaml"))
        featureModel = FeatureModel.fromYaml(featureModelYamlPath)
        camera = Camera(self.camera.projectionMatrix, np.zeros((1,4), dtype=np.float32), resolution=self.camera.resolution)
        featureExtractor = Perception.create(trackingConfig, camera, featureModel)

        labeledImgs = readLabeledImages(datasetPath, labelFile)

        cv.imshow("frame", np.zeros((10,10), dtype=np.uint8))

        undistort = not np.all(self.camera.distCoeffs == 0)

        allErrors = []
        failedImgs = []
        filedNCandidates = []
        filedNCandidates2 = []
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
                if True:
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
                                                                                                        colorCoeffs=colorCoeffs)
                
                

                cv.imshow("processed image", processedImg)
                cv.imshow("pose image", poseImg)
                cv.imshow("frame", tmpFrame)
                cv.setWindowTitle("frame", imgName)
                key = cv.waitKey(1)
                
                """
                # When using predicted ROI 
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
                """
                attempts = 0
                if dsPose:
                    attempts = dsPose.attempts
                poseEstimationAttempts.append(attempts)
                peakIterations.append(featureExtractor.lightSourceDetector.localPeak.iterations)

                for ls in candidates:
                    ls.center = ls.center[0]+offset[0], ls.center[1]+offset[1]
                
                covElapsed = 0
                if dsPose:
                    covElapsed = 0.1
                    if timeitNumber > 0:
                        def covCalc():
                            calcPoseCovarianceFixedAxis(dsPose.camera, 
                                                        dsPose.featureModel, 
                                                        dsPose.translationVector, 
                                                        dsPose.rotationVector, 
                                                        dsPose.pixelCovariances)
                        #def covCalc():
                        timeitFunc = covCalc
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
                    if len(candidates) > len(errCircles):
                        filedNCandidates2.append(i)

                    for ls in dsPose.associatedLightSources:
                        if candidates.index(ls) > len(errCircles)-1:
                            filedNCandidates.append(i)
                            break
                else:
                    filedNCandidates.append(i)
                    filedNCandidates2.append(i)

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
        print("Failed n candidates2 ({}): {}".format(len(filedNCandidates2), filedNCandidates2))
        
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
        plt.plot(imageNumbers, undistortTimes)
        plt.plot(imageNumbers, processTimes)
        plt.plot(imageNumbers, covTimes)
        plt.legend(["Total", "Undistortion", "Image processing", "Covariance calculation"])
        
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


    def _testImage2(self, imgName, imgColorRaw, imgRectOrig, labeledImgs, i):
        tmpFrame = imgRectOrig.copy()
        
        errCircles = labeledImgs[i]
        for j, ec in enumerate(self._undistortErrCircles(errCircles)):
            drawErrorCircle(tmpFrame, ec, j, (0, 255, 0))

        key = 0xFF
        elapsed = None
        errors = []
        self.associatedImgPointsMsg = None

        # publish once
        self._publish(imgColorRaw, imgRectOrig)
        start = time.time()
        while not rospy.is_shutdown():

            # get associated points from subscription
            #rospy.sleep(0.1)
            if self.associatedImgPointsMsg:
                if not elapsed:
                    elapsed = time.time() - start
                points = msgToImagePoints(self.associatedImgPointsMsg)
                
                for p in points:
                    cv.circle(tmpFrame, p, 2, (255, 0, 0), 2)

                errors = [np.linalg.norm([p[0]-e[0], p[1]-e[1]]) for p, e in zip(points, errCircles)]
                # image points found
                self.associatedImgPointsMsg = None
                break

            cv.imshow("frame", tmpFrame)
            #cv.imshow("rect", imgRectOrig)
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
                start = time.time()
            else:
                break
            
        return key, elapsed, errors

    def _testTrackerOnImage(self, img, tracker, labels=None, estDSPose=None):
        result = {"est_pose": None,
                  "elapsed": 0,
                  "candidates": [],
                  "detected_points": [],
                  "errors": [],
                  "norm_error": 0,
                  "success": False if labels else None}

        timer = Timer("Elapsed")

        timer.start()
        dsPose, _, candidates, _, _ = tracker.estimatePose(img, estDSPose=estDSPose)
        timer.stop()
        result["est_pose"] = dsPose
        result["elapsed"] = timer.elapsed()

        result["candidates"] = [ls.center for ls in candidates]
        
        if dsPose:
            result["detected_points"] = [ls.center for ls in dsPose.associatedLightSources]

        if not dsPose or not labels:
            return result

        # Only if labels are given and a pose was detected
        result["errors"] = [np.linalg.norm([p[0]-e[0], p[1]-e[1]]) for p, e in zip(result["detected_points"], labels)]
        result["norm_error"] = sum(result["errors"])/len(result["errors"])
        
        return result


    def testTracker(self, trackingYaml, lebelUnlabelImages=False):
        trackingYaml = makeAbsolute(trackingYaml, loloDefs.TRACKING_CONFIG_DIR)
        tracker = Perception.create(trackingYaml, self.camera, self.featureModel)

        results = []
        estDSPose = None

        wName = "frame"
        cv.namedWindow(wName)

        for imgIdx in range(len(self.dataset)):
            frame = self.dataset.loadImage(imgIdx)
            imgName = self.dataset.idxToImageName(imgIdx)
            labels = self.dataset.getLabels(imgIdx)
            imgColorRaw = frame.copy()

            imgRect = imgColorRaw
            if self.dataset.metadata["is_raw_images"]:
                imgRect = self.camera.undistortImage(imgColorRaw).astype(np.uint8)

            result = self._testTrackerOnImage(imgRect, tracker, labels, estDSPose)
            estDSPose = result["est_pose"]
            results.append(result)

            tmpFrame = imgRect.copy()
            if labels:
                for i, c in enumerate(labels): 
                    drawErrorCircle(tmpFrame, c, i, (0,255,0))

            if result["detected_points"]:
                for i, p in enumerate(result["detected_points"]):
                    drawErrorCircle(tmpFrame, (p[0], p[1], 0), i, (255,0,0), thickness=3)
            else:
                for i, p in enumerate(result["candidates"]):
                    drawErrorCircle(tmpFrame, (p[0], p[1], 0), i, (255,0,0), thickness=3)

            cv.setWindowTitle(wName, imgName)
            cv.imshow(wName, tmpFrame)

            key = cv.waitKey(0)            

            if key == ord("q"):
                break

            successes = [r["success"] for r in results]
            normError = np.average([r["norm_error"] for r in results])
            print("Successfull detections {}/{}".format(successes.count(True), successes.count(False)))
            print("Avg pixel error: {}".format(normError))
        #print("Failed n candidates ({}): {}".format(len(filedNCandidates), filedNCandidates))
        #print("Failed n candidates2 ({}): {}".format(len(filedNCandidates2), filedNCandidates2))
        return
        nImages = len(labeledImages)
        imageNumbers = list(range(nImages))

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
        plt.legend(["Total"])
        
        figures.append(plt.figure())
        plt.xlim(0, nImages+1)
        plt.ylabel("FPS")
        plt.xlabel("Image number")
        plt.plot(imageNumbers, 1./np.array(times))

        figures.append(plt.figure())
        plt.xlim(0, nImages+1)
        plt.ylabel("Number of detected light sources")
        plt.xlabel("Image number")
        plt.plot(imageNumbers, nCandidates)
        
        print("Saving images")
        for i, fig in enumerate(figures):
            fig.savefig("fig{}.png".format(i), dpi=fig.dpi, format='png')

        cv.destroyAllWindows()


    def _click(self, event, x, y, flags, imgIdx):
        """
        Callback when the image is clicked. Prints info about the dataset
        and the batch status of its image loader (for debug purposes).
        """
        if event == cv.EVENT_LBUTTONDOWN:
            self.dataset.printInfo()
            self.dataset.loader.printBatchStatus(imgIdx)


    def analyzeDataset(self):
        
        datasetDir = self.dataset._datasetDir
        dataset = self.dataset #dataset = ImageDataset(datasetDir)

        # Times the playback fps. The playback fps is displayed in the image.
        # When loading images from the dataset, a fps drop might occur and this timer 
        # gives visual feedback when this happens.
        fpsTimer = Timer("fps", nAveragSamples=20)
        # Just for the timer to record at least one sample to allow calling avg() before the first stop().
        fpsTimer.start()
        fpsTimer.stop()

        # Timer to keep track of how much we have to sleep to uphold the desired playback fps.
        rateTimer = Timer("rate")
        
        imgIdx = 0
        play = False
        labelerActive = False

        wName = "frame"
        cv.namedWindow(wName)
        cv.setMouseCallback(wName, self._click, param=imgIdx)
        
        # Using some custom trackbars to enable dynamic update 
        # of the trakcbar when only maxval is changed 
        # (the opencv trackbars does not really allow this).
        imageTrackbar = Trackbar("Image", wName, imgIdx, 1)
        sourceFPSTrackbar = Trackbar("Source FPS", wName, dataset.metadata["fps"], 60, minValue=1)
        playbackTrackbar = Trackbar("Playback FPS", wName, dataset.metadata["fps"], 60, minValue=1)

        labeler = ImageLabeler()
        while True:
            
            fpsTimer.start()
            rateTimer.start()

            # loadImage() initializes the loader that reads and save the images in batches
            # based on the current imgIdx that is loaded.
            img = dataset.loadImage(int(round(imgIdx)))

            # Update the trackbars
            if imageTrackbar.hasChanged():
                imgIdx = imageTrackbar.value
            imageTrackbar.setPos(int(round(imgIdx)))
            imageTrackbar.setMaxVal(max(0, len(dataset)-1))
            imageTrackbar.update()
            sourceFPSTrackbar.update()
            playbackTrackbar.update()
            dataset.metadata["fps"] = sourceFPSTrackbar.value

            imgColorRaw = img.copy()
            
            # Rectify the image if it is not already
            imgRect = imgColorRaw.copy()
            if dataset.metadata["is_raw_images"]:
                imgRect = self.camera.undistortImage(imgColorRaw).astype(np.uint8)

            imgTemp = imgRect.copy()
            labels = dataset.getLabels(imgIdx)

            if labelerActive:
                if labelerActive:
                    play = False
                    key, labels = labeler.label(wName, imgColorRaw, errCircles=labels)
                    if key == ord("q"):
                        key = ord("l")
                    dataset.addLabels(labels, imgIdx)
            else:
                if labels:
                    # If the image has associated labels, draw them.
                    labelsTemp = labels
                    if dataset.metadata["is_raw_images"]:
                        labelsTemp = self._undistortErrCircles(labelsTemp)
                    for i,l in enumerate(labelsTemp): 
                        drawErrorCircle(imgTemp, l, i, (0,255,0))

                # Scale the display image to a fixed width. TODO: should not be hard coded
                imgTemp = scaleImageToWidth(imgTemp, 1280)
                # Plot playback fps as reference (may drop when the images are loaded)
                imgTemp = plotFPS(imgTemp, 1./fpsTimer.avg())
                # Draw the status of the loader
                imgTemp = dataset.loader.drawBatchStatus(imgTemp, int(round(imgIdx)))

                cv.setWindowTitle(wName, "{}: {}".format(os.path.basename(datasetDir), dataset.idxToImageName(int(round(imgIdx)))))
                cv.imshow(wName, imgTemp)

                key = cv.waitKey(1)

            if play:
                # When play == True, imgIdx is increamented based on source and playback fps
                imgIdx += float(sourceFPSTrackbar.value)/playbackTrackbar.value
                imgIdx = min(len(dataset)-1, imgIdx)
                if imgIdx == len(dataset)-1:
                    play = False
                self._publish(imgColorRaw, imgRect)
            else:
                # If play == False, the user increments imgIdx manually.
                # The images are published every time imgIdx changes (except when using the trackbar)
                if key in (ord("+"), ord("n"), 83): # +, n or right/up arrow
                    imgIdx += 1
                    imgIdx = min(len(dataset)-1, imgIdx)
                    self._publish(imgColorRaw, imgRect)
                elif key in (ord("-"), 81): # - or left/down arrow
                    imgIdx -= 1
                    imgIdx = max(0, imgIdx)
                    self._publish(imgColorRaw, imgRect)
                elif key == ord("p"):
                    self._publish(imgColorRaw, imgRect)

            if key == ord("q"):
                break
            elif key == 32: # spacebar
                play = not play
            elif key == ord("s"):
                # Saves the metadata and labels
                dataset.save()
                print("Saved dataset")
            elif key == ord("z"):
                dataset.zipImages()
            elif key == ord("l"):
                labelerActive = not labelerActive
                if not labelerActive:
                    cv.setMouseCallback(wName, self._click, param=imgIdx)
            else:
                pass
                #print("key:", key)

            rateTimer.stop()
            time.sleep(max(0, 1./playbackTrackbar.value - rateTimer.elapsed()))
            fpsTimer.stop()

        cv.destroyAllWindows()


    @classmethod
    def createDataset(cls, datasetDir, imageGenerator, cameraYaml, featureModelYaml, isRawImages, fps, startIdx=0, endIdx=None):
        """
        Creates a dataset that is comptible with ImageAnalyzeNode.analyzeDataset().
        ImageAnalyzeNode.analyzeDataset() needs the following metadata to execute:
        cameraYaml - path to the camera yaml
        featureModelYaml - path to the feature model yaml
        fps - the fps of the source recording
        isRawImages - if the images provided by imageGenerator is raw (not rectified/undistorted). 
        """
        datasetDir = makeAbsolute(datasetDir, loloDefs.IMAGE_DATASET_DIR)
        
        metadata = {"camera_yaml": cameraYaml,
                    "feature_model_yaml": featureModelYaml,
                    "fps": fps,
                    "is_raw_images": isRawImages}

        return ImageDataset.create(datasetDir, imageGenerator, metadata, startIdx, endIdx)


    @classmethod
    def createDatasetFromRosbag(cls, datasetDir, rosbagPath, imageRawTopic, cameraYaml, featureModelYaml, isRawImages, fps, startIdx=0, endIdx=None):
        rosbagPath = makeAbsolute(rosbagPath, loloDefs.ROSBAG_DIR)
        g = cls.rosbagImageGenerator(rosbagPath, imageRawTopic)
        cls.createDataset(datasetDir, g, cameraYaml, featureModelYaml, isRawImages, fps, startIdx, endIdx)


    @classmethod
    def createDatasetFromVideo(cls, datasetDir, videoPath, cameraYaml, featureModelYaml, isRawImages, fps, startIdx=0, endIdx=None):
        videoPath = makeAbsolute(videoPath, loloDefs.VIDEO_DIR)
        g = cls.videoImageGenerator(videoPath)
        cls.createDataset(datasetDir, g, cameraYaml, featureModelYaml, isRawImages, fps, startIdx, endIdx)


    @staticmethod
    def videoImageGenerator(videoPath):
        cap = cv.VideoCapture(videoPath)
        if (cap.isOpened()== False):
            print("Error opening video stream or file")

        i = 0
        while (cap.isOpened() and not rospy.is_shutdown()):
            ret, frame = cap.read()
            if ret == True:
                i += 1
                
                yield frame
            else:
                break
        cap.release()


    @staticmethod
    def rosbagImageGenerator(rosbagPath, imageRawTopic):
        bridge = CvBridge()
        bag = rosbag.Bag(rosbagPath)

        i = 0
        for _, msg, _ in bag.read_messages(topics=imageRawTopic):
            i += 1
            
            if rospy.is_shutdown():
                return

            if msg._type == Image._type:
                frame = bridge.imgmsg_to_cv2(msg, 'bgr8')
            elif msg._type == CompressedImage._type:
                arr = np.fromstring(msg.data, np.uint8)
                frame = cv.imdecode(arr, cv.IMREAD_COLOR)
            else:
                raise Exception("Invalid message type '{}'".format(msg._type))

            yield frame

        if i == 0:
            print("No image messages with topic '{}' found".format(imageRawTopic))


if __name__ == "__main__":

    # single light source
    # imgLabelNode = ImageAnalyzeNode("config/camera_calibration_data/contour.yaml")
    # videoPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "datasets/FILE0151.MP4")
    # imgLabelNode.analyzeVideoImages(datasetPath, labelFile, videoPath, startFrame=550, analyzeImages=False)

    # imgLabelNode = ImageAnalyzeNode("config/camera_calibration_data/contour.yaml")
    # videoPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "datasets/171121/171121_straight_test.MP4")
    # videoPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "datasets/171121/171121_straight_test_reversed.mkv")
    # videoPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "datasets/171121/171121_angle_test.MP4")
    # videoPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "datasets/FILE0197.MP4")
    # videoPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "datasets/271121/271121_5planar_1080p.MP4")
    # imgLabelNode.analyzeVideoImages(datasetPath, labelFile, videoPath, startFrame=100, analyzeImages=False, waitForFeatureExtractor=False) #350

    # DoNN dataset
    # imgLabelNode = ImageAnalyzeNode("config/camera_calibration_data/donn_camera.yaml")
    # videoPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "image_dataset/dataset_recovery/donn.mp4")
    # datasetRecovery = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "image_dataset/dataset_recovery")
    # imgLabelNode.analyzeVideoImages(datasetRecovery, "donn.txt", videoPath, startFrame=1, analyzeImages=True, test=True)

    # imgLabelNode = ImageAnalyzeNode("config/camera_calibration_data/usb_camera_720p_sim.yaml")
    # rosbagFile = "sim_bag.bag"
    # rosbagPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), join("rosbags", rosbagFile))
    # imgLabelNode.analyzeRosbagImages(datasetPath, labelFile, rosbagPath, "/lolo/sim/camera_aft/image_color", startFrame=1000, analyzeImages=False)

    # imgLabelNode = ImageAnalyzeNode("config/camera_calibration_data/usb_camera_720p_8.yaml")
    # rosbagFile = "ice.bag"
    # rosbagPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), join("rosbags", rosbagFile))
    # imgLabelNode.analyzeRosbagImages(datasetPath, labelFile, rosbagPath, "lolo_camera/image_raw", startFrame=1200, analyzeImages=False, waitForFeatureExtractor=False) # 1200

    # imgLabelNode = ImageAnalyzeNode("config/camera_calibration_data/usb_camera_720p_8.yaml")
    # rosbagFile = "test_session_5mm_led_prototype/pos_1.bag"
    # rosbagPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), join("rosbags", rosbagFile))
    #imgLabelNode.analyzeRosbagImages(datasetPath, labelFile, rosbagPath, "/lolo_camera/image_raw", startFrame=1, analyzeImages=False, waitForFeatureExtractor=False) # 1200

    # For Aldo
    # cameraYaml = "config/camera_calibration_data/kristineberg.yaml" # In /camera_calibration_data
    
    # Lab test
    #rosbagFile = "2022-06-08-17-44-05_lab_docking_station.bag" # In /rosbags
    #topic = "/csi_cam_1/camera/image_raw/compressed"
    #startFrame = 700
    
    # Seabed 1
    #rosbagFile = "2022-06-09-12-03-13.bag"
    #topic = "/sam/perception/csi_cam_0/camera/image_raw/compressed"
    #startFrame = 7000
    
    # Seabed 2
    #rosbagFile = "2022-06-09-12-10-59.bag"
    #topic = "/sam/perception/csi_cam_0/camera/image_raw/compressed"
    #startFrame = 1

    # Side cameras
    #rosbagFile = "2022-06-09-19-44-50.bag"
    
    #cameraYaml = "config/camera_calibration_data/csi_cam_2.yaml"
    #topic = "/sam/perception/csi_cam_2/camera/image_raw/compressed"

    # rosbagFile = "kristineberg.bag"
    # cameraYaml = "camera_calibration_data/csi_cam_1.yaml"
    # topic = "/sam/perception/csi_cam_1/camera/image_raw/compressed"
    # startFrame = 0 #5900 #8800

    import argparse

    rospy.init_node("image_analyze_node")

    parser = argparse.ArgumentParser(description="Analyze images from datasets that can be created from rosbags or videos.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    loloDir = rospkg.RosPack().get_path("lolo_perception")
    datasetsDir = os.path.join(loloDir, "image_datasets")

    parser.add_argument('dataset_dir', 
                        help="The directory of the dataset to be analyzed/created." \
                             "If the path is relative, the dataset is loaded/saved in {}.\n" \
                             "If the directory exists, other arguments are ignored and the node will start".format(loloDefs.IMAGE_DATASET_DIR))
    parser.add_argument("-test", action="store_true", help="Using this flag, the tracker specified by the tracking yaml will be tested on the available labels in the dataset.")
    parser.add_argument("-tracking_yaml", help="Tracking configuration of the tracker to be tested. Relative path starts at {}".format(loloDefs.TRACKING_CONFIG_DIR))
    parser.add_argument("-file", help="Rosbag (.bag) or video (see opencv supported formats) file path to generate images from to create the dataset.")
    parser.add_argument("-topic", help="If -file is a rosbag, the image topic has to be given.")
    parser.add_argument("-camera_yaml", default="Undefined", help="Path to the camera yaml. Relative path starts from {}".format(loloDefs.CAMERA_CONFIG_DIR))
    parser.add_argument("-feature_model_yaml", default="Undefined", help="Path to the feature model yaml. Relative path starts from {}".format(loloDefs.FEATURE_MODEL_CONFIG_DIR))
    parser.add_argument("-fps", default=30, help="Frames per second of the source recording.")
    parser.add_argument("-is_raw_images", default=True, help="Indicates if the recorded images are raw. Currently this has to be true.")
    parser.add_argument("-start", default=0, type=int,  help="Start image index of the source recording to save the dataset.")
    parser.add_argument("-end", type=int, help="End image index of the source recording to save the dataset.")
    parser.add_argument("-print_info", action="store_true", help="Prints all available datasets and exits.")
    
    args = parser.parse_args()
    
    if args.print_info:
        datasets = []
        for d in os.listdir(loloDefs.IMAGE_DATASET_DIR):
            try:
                dataset = ImageDataset(os.path.join(loloDefs.IMAGE_DATASET_DIR, d))
            except:
                print("Failed to open dataset '{}'".format(d))
                print("")
            else:
                datasets.append(dataset)
            
        for dataset in datasets:
            dataset.printInfo()
            print("")

        exit()

    if os.path.isdir(makeAbsolute(args.dataset_dir, loloDefs.IMAGE_DATASET_DIR)):
        analyzer = ImageAnalyzeNode(args.dataset_dir)
        if args.test:
            if not args.tracking_yaml:
                parser.error("-tracking_yaml has to be specified to run a test.")
            
            analyzer.testTracker(args.tracking_yaml)
        else:
            analyzer.analyzeDataset()
    else:
        if not args.file:
            parser.error("The dataset does not exist and cannot create a new if -file is not given.")
        isRosbag = os.path.splitext(args.file)[1] == ".bag"
        if isRosbag:
            # ROsbag
            if not args.topic:
                parser.error("Require --topic when -file is a rosbag.")

            ImageAnalyzeNode.createDatasetFromRosbag(args.dataset_dir, 
                                                     args.file, 
                                                     args.topic, 
                                                     args.camera_yaml, 
                                                     args.feature_model_yaml, 
                                                     args.is_raw_images, 
                                                     args.fps, 
                                                     startIdx=args.start, 
                                                     endIdx=args.end)
        else:
            # Assume video
            ImageAnalyzeNode.createDatasetFromVideo(args.dataset_dir, 
                                                    args.file,
                                                    args.camera_yaml, 
                                                    args.feature_model_yaml, 
                                                    args.is_raw_images, 
                                                    args.fps, 
                                                    startIdx=args.start, 
                                                    endIdx=args.end)
