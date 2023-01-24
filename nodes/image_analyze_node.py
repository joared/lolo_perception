#!/usr/bin/env python
import sys
import os
import time
import cv2 as cv
import numpy as np
import rospy
import rosbag
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import PoseArray
from lolo_perception.msg import FeatureModel as FeatureModelMsg

import lolo_perception.definitions as loloDefs
from lolo_perception.definitions import makeAbsolute
from lolo_perception.utils import Timer
from lolo_perception.camera_model import Camera
from lolo_perception.feature_model import FeatureModel
from lolo_perception.ros_utils import readCameraYaml, yamlToFeatureModelMsg, msgToImagePoints
from lolo_perception.plotting_utils import plotFPS
from lolo_perception.image_processing import scaleImageToWidth
from lolo_perception.image_dataset import ImageDataset
from lolo_perception.tracker import Tracker

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


    def _associatedImagePointsCallback(self, msg):
        self.associatedImgPointsMsg = msg


    def _undistortErrCircles(self, errCircles):
        imgPoints = np.array([(x,y) for x, y, _ in errCircles], dtype=np.float32)
        undistImgPoints = self.camera.undistortPoints(imgPoints)
        errCirclesUndist = [(p[0], p[1], errCirc[2]) for p, errCirc in zip(undistImgPoints, errCircles)]

        return errCirclesUndist


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


    def testTracker(self, trackingYaml, labelUnlabelImages=False):
        trackingYaml = makeAbsolute(trackingYaml, loloDefs.TRACKING_CONFIG_DIR)
        tracker = Tracker.create(trackingYaml, self.camera, self.featureModel)

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
        Creates a dataset that is compatible with ImageAnalyzeNode.analyzeDataset().
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
        """
        Generates images from a video file using opencv VideoCapture.
        """
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
    def rosbagImageGenerator(rosbagPath, imageTopic):
        """
        Generates images from a RosBag with the given image topic. The images can be either of type "Image" or "CompressedImage"
        """
        bridge = CvBridge()
        bag = rosbag.Bag(rosbagPath)

        i = 0
        for _, msg, _ in bag.read_messages(topics=imageTopic):
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
            print("No image messages with topic '{}' found".format(imageTopic))


if __name__ == "__main__":

    import argparse

    rospy.init_node("image_analyze_node")

    parser = argparse.ArgumentParser(description="Analyze images from datasets that can be created from rosbags or videos.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Directory of the dataset to be opened (directory exists) or created (the directory should not exist and will be created)
    parser.add_argument('dataset_dir', 
                        help="The directory of the dataset to be analyzed/created." \
                             "If the path is relative, the dataset is loaded/saved in {}.\n" \
                             "If the directory exists, other arguments are ignored and the node will start".format(loloDefs.IMAGE_DATASET_DIR))
    
    # Needed for creating a new dataset    
    parser.add_argument("-file", help="Rosbag (.bag) or video (see opencv supported formats) file path to generate images from to create the dataset.")
    parser.add_argument("-topic", help="If -file is a rosbag, the image topic has to be given.")
    
    # Metadata that is saved for the new dataset
    parser.add_argument("-camera_yaml", default="Undefined", help="Path to the camera yaml. Relative path starts from {}".format(loloDefs.CAMERA_CONFIG_DIR))
    parser.add_argument("-feature_model_yaml", default="Undefined", help="Path to the feature model yaml. Relative path starts from {}".format(loloDefs.FEATURE_MODEL_CONFIG_DIR))
    parser.add_argument("-fps", default=30, help="Frames per second of the source recording.")
    parser.add_argument("-is_raw_images", default=True, help="Indicates if the recorded images are raw. Currently this has to be true.")
    parser.add_argument("-start", default=0, type=int,  help="Start image index of the source recording to save the dataset.")
    parser.add_argument("-end", type=int, help="End image index of the source recording to save the dataset.")
    
    # Testing your specified tracker
    parser.add_argument("-test", action="store_true", help="Using this flag, the tracker specified by the tracking yaml will be tested on the available labels in the dataset.")
    parser.add_argument("-tracking_yaml", help="Tracking configuration of the tracker to be tested. Relative path starts at {}".format(loloDefs.TRACKING_CONFIG_DIR))
    
    # Prints the available datasets
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
            # Rosbag
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
