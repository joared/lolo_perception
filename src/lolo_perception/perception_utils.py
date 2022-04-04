import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R


def plotPoseImageInfo(poseImg,
                      dsPose,
                      camera,
                      featureModel,
                      poseAquired,
                      validOrientationRange,
                      roiCnt=None,
                      roiCntUpdated=None,
                      progress=0):

    validOrientation = False
    if dsPose:
        validYaw, validPitch, validRoll = dsPose.validOrientation(*validOrientationRange)
        validOrientation = validYaw and validPitch and validRoll

    if poseAquired:
        axisColor = None
        roiColor = (0, 255, 0)
        if not validOrientation:
            axisColor = (0, 0, 255)
            roiColor = (0, 0, 255)
        if roiCnt is not None:
            cv.drawContours(poseImg, [roiCnt], -1, (100,100,100), 3)
        if roiCntUpdated is not None:
            cv.drawContours(poseImg, [roiCntUpdated], -1, roiColor, 3)
            
            cv.putText(poseImg, 
                       "#{}".format(dsPose.detectionCount), 
                       (roiCntUpdated[0][0]-10, roiCntUpdated[0][1]-10), 
                       cv.FONT_HERSHEY_SIMPLEX, 
                       fontScale=1, 
                       thickness=2, 
                       color=(0,255,0))

            cv.putText(poseImg, 
                       "#{}".format(dsPose.attempts), 
                       (roiCntUpdated[1][0]-20, roiCntUpdated[1][1]-10), 
                       cv.FONT_HERSHEY_SIMPLEX, 
                       fontScale=1, 
                       thickness=2, 
                       color=(0,0,255))

            # mahanalobis distance meter
            mahaDistRatio = dsPose.mahaDist/dsPose.mahaDistThresh

            """
            xStart = roiCntUpdated[2][0]+8
            yStart = roiCntUpdated[2][1]
            xEnd = xStart
            yEnd = roiCntUpdated[1][1]
            
            cv.line(poseImg, (xStart, yStart), (xEnd, int(mahaDistRatio*(yEnd - yStart)) + yStart), (0,0,255), 10)
            """

            xStart = roiCntUpdated[2][0]+3
            yStart = roiCntUpdated[2][1]+2
            xEnd = xStart + 10
            yEnd = roiCntUpdated[1][1]
            
            cv.rectangle(poseImg, (xStart, yStart), (xEnd, int(mahaDistRatio*(yEnd - yStart)) + yStart), color=(0,0,255), thickness=-1)
        if dsPose:
            plotAxis(poseImg, 
                    dsPose.translationVector, 
                    dsPose.rotationVector, 
                    camera, 
                    featureModel.features, 
                    featureModel.maxRad, # scaling for the axis shown
                    color=axisColor,
                    thickness=5) 
    if dsPose:
        """
        # "Cone angle"
        coneX = np.rad2deg(np.arctan(abs(dsPose.translationVector[0]) / abs(dsPose.translationVector[2])))
        coneY = np.rad2deg(np.arctan(abs(dsPose.translationVector[1]) / abs(dsPose.translationVector[2])))
        org = 10, camera.resolution[0]-10
        cv.putText(poseImg, 
                    "Cone angle x: {}, y: {}".format(round(coneX), round(coneY)), 
                    org, 
                    cv.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, 
                    thickness=2, 
                    color=(0,0,255))
        """

        plotPoseInfo(poseImg, 
                    dsPose.translationVector, 
                    dsPose.rotationVector,
                    yawColor=(0,255,0) if validYaw else (0,0,255),
                    pitchColor=(0,255,0) if validPitch else (0,0,255),
                    rollColor=(0,255,0) if validRoll else (0,0,255))

        plotPosePoints(poseImg, 
                    dsPose.translationVector, 
                    dsPose.rotationVector, 
                    camera, 
                    featureModel.features, 
                    color=(0, 0, 255))
        plotPoints(poseImg, [ls.center for ls in dsPose.associatedLightSources], (255, 0, 0), radius=5)

    plotCrosshair(poseImg, camera)
    

    # progress bar for light source tracker and aquiring pose
    xStart = camera.resolution[1]-30
    yStart = camera.resolution[0]-10
    xEnd = xStart
    yEnd = yStart - 200
    cv.line(poseImg, (xStart, yStart), (xEnd, yEnd), (255,255,255), 15)
    cv.line(poseImg, (xStart, yStart), (xEnd, int(progress*(yEnd - yStart)) + yStart), (0,255,0), 10)

def plotAxis(img, translationVector, rotationVector, camera, points, scale, color=None, thickness=2, opacity=1):
    points = points[:, :3].copy()

    zDir, _ = cv.projectPoints(np.array([(0.0, 0.0, scale)]), 
                               rotationVector, 
                               translationVector, 
                               camera.cameraMatrix, 
                               camera.distCoeffs)

    yDir, _ = cv.projectPoints(np.array([(0.0, scale, 0.0)]), 
                               rotationVector, 
                               translationVector, 
                               camera.cameraMatrix, 
                               camera.distCoeffs)

    xDir, _ = cv.projectPoints(np.array([(scale, 0.0, 0.0)]), 
                               rotationVector, 
                               translationVector, 
                               camera.cameraMatrix, 
                               camera.distCoeffs)

    center, _ = cv.projectPoints(np.array([(0.0, 0.0, 0.0)]), 
                                 rotationVector, 
                                 translationVector, 
                                 camera.cameraMatrix, 
                                 camera.distCoeffs)

    center = center[0][0][0], center[0][0][1]   
    for d, c in zip((xDir, yDir, zDir), ((0,0,255*opacity), (0,255*opacity,0), (255*opacity,0,0))):
        cx = center[0]
        cy = center[1]
        point1 = (int(round(cx)), int(round(cy)))
        point2 = (int(round(d[0][0][0])), int(round(d[0][0][1])))
        if color is not None:
            cv.line(img, point1, point2, tuple((c*opacity for c in color)), thickness)
        else:
            cv.line(img, point1, point2, c, thickness)

def plotPosePoints(img, translationVector, rotationVector, camera, points, color):
    projPoints = projectPoints(translationVector, rotationVector, camera, points)
    plotPoints(img, projPoints, color, radius=5)

def plotPosePointsWithReprojection(img, translationVector, rotationVector, camera, objPoints, imagePoints, color):
    projPoints = projectPoints(translationVector, rotationVector, camera, objPoints)
    plotPosePoints(img, translationVector, rotationVector, camera, objPoints, color)
    rpErrNorms, rms = reprojectionError(translationVector, rotationVector, camera, objPoints, imagePoints)
    rms = round(rms, 2)
    cv.putText(img, "RMS: {}".format(rms), (img.shape[1]-150, 15), cv.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=color)
    for imgP, projP, rp in zip(imagePoints, projPoints, rpErrNorms):
        imgX, imgY = int( round(imgP[0]) ), int( round(imgP[1]) )
        rp = round(np.linalg.norm(rp), 2)
        cv.putText(img, "err: {}".format(rp), (imgX+15, imgY-15), cv.FONT_HERSHEY_SIMPLEX, 0.5, color=color)
        
        projX, projY = int( round(projP[0]) ), int( round(projP[1]) )
        cv.line(img, (imgX, imgY), (projX, projY), color=color)

def plotPoints(img, points, color, radius=1):
    for p in points:
        x = int( round(p[0]) )
        y = int( round(p[1]) )
        cv.circle(img, (x,y), radius, color, -1)

def projectPoints(translationVector, rotationVector, camera, objPoints):
    projPoints, _ = cv.projectPoints(objPoints, 
                                        rotationVector, 
                                        translationVector, 
                                        camera.cameraMatrix, 
                                        camera.distCoeffs)
    projPoints = np.array([p[0] for p in projPoints])
    return projPoints

def plotPoseInfo(img, translationVector, rotationVector, yawColor=(0,255,0), pitchColor=(0,255,0), rollColor=(0,255,0)):
    distance = np.linalg.norm(translationVector)
    yaw, pitch, roll = R.from_rotvec(rotationVector).as_euler("YXZ")
    yaw, pitch, roll = np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll)
    org = (20, 25)
    
    if distance < 2:
        unit = "cm"
        distance = round(distance*100, 2)
        translationVector = translationVector.copy()
        translationVector *= 100
    else:
        unit = "m"
        distance = round(distance, 2)

    inc = 25
    cv.putText(img, "Range: {} {}".format(distance, unit), org, cv.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    org = (org[0], org[1]+inc)
    cv.putText(img, "X: {} {}".format(round(translationVector[0], 2), unit), org, cv.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    org = (org[0], org[1]+inc)
    cv.putText(img, "Y: {} {}".format(round(translationVector[1], 2), unit), org, cv.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    org = (org[0], org[1]+inc)
    cv.putText(img, "Z: {} {}".format(round(translationVector[2], 2), unit), org, cv.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    org = (org[0], org[1]+inc)
    cv.putText(img, "Yaw: {} deg".format(round(yaw, 2)), org, cv.FONT_HERSHEY_SIMPLEX, 1, color=yawColor, thickness=2, lineType=cv.LINE_AA)
    org = (org[0], org[1]+inc)
    cv.putText(img, "Pitch: {} deg".format(round(pitch, 2)), org, cv.FONT_HERSHEY_SIMPLEX, 1, color=pitchColor, thickness=2, lineType=cv.LINE_AA)
    org = (org[0], org[1]+inc)
    cv.putText(img, "Roll: {} deg".format(round(roll, 2)), org, cv.FONT_HERSHEY_SIMPLEX, 1, color=rollColor, thickness=2, lineType=cv.LINE_AA)


def plotCrosshair(img, camera, color=(0, 0, 255)):
    center, = projectPoints(np.array([0., 0., 1.]), 
                            np.array([0., 0., 0.]), 
                            camera, 
                            np.array([[0., 0., 0.]]))

    cx = int(round(center[0]))
    cy = int(round(center[1]))
    size = 10
    cv.line(img, (cx - size, cy), (cx + size, cy), color, 2)
    cv.line(img, (cx, cy - size), (cx, cy + size), color, 2)


def reprojectionError(translationVector, rotationVector, camera, objPoints, imagePoints):
    """

    https://github.com/strawlab/image_pipeline/blob/master/camera_calibration/nodes/cameracheck.py
    """
    projPoints = projectPoints(translationVector, rotationVector, camera, objPoints)

    reprojErrs = imagePoints - projPoints
    rms = np.sqrt(np.sum(reprojErrs ** 2)/np.product(reprojErrs.shape))

    return reprojErrs, rms

def undistortImage(img, camera):
    # How to undistort image: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera.cameraMatrix, 
                                                     camera.distCoeffs, 
                                                     (w,h), 
                                                     0, 
                                                     (w,h))

    imgRect = cv.undistort(img, camera.cameraMatrix, camera.distCoeffs, None, newcameramtx)
    x, y, w, h = roi
    imgRect = imgRect[y:y+h, x:x+w]
    return imgRect, newcameramtx

class PoseAndImageUncertaintyEstimator:
    """
    Utility class to use when evaluating uncertainty of pose and image points.
    Make sure all the samples are correct (watch out for outliers)
    """
    def __init__(self, nImagePoints, nSamples=100):
        self.associatedImagePoints = [[] for _ in range(nImagePoints)]
        self.poseVecs = [] # list of pose vectors [tx, ty, tz, rx, ry, rz]
        self.poseVecsEuler = [] # list of pose vectors [tx, ty, tz, ax, ay, az] (a_ in degrees)
        self.camPoseVecs = []
        self.camPoseVecsEuler = []
        self.nSamples = nSamples

    def add(self, 
            translationVec, 
            rotationVec,
            camTranslationVec,
            camRotationVec, 
            assImgPoints):

        self.poseVecs.insert(0, list(translationVec) + list(rotationVec))
        self.poseVecs = self.poseVecs[:self.nSamples]

        self.camPoseVecs.insert(0, list(camTranslationVec) + list(camRotationVec))
        self.camPoseVecs = self.camPoseVecs[:self.nSamples]

        self.poseVecsEuler.insert(0, list(translationVec) + list(R.from_rotvec(rotationVec).as_euler("YXZ", degrees=True)))
        self.poseVecsEuler = self.poseVecsEuler[:self.nSamples]
        self.camPoseVecsEuler.insert(0, list(camTranslationVec) + list(R.from_rotvec(camRotationVec).as_euler("YXZ", degrees=True)))
        self.camPoseVecsEuler = self.camPoseVecsEuler[:self.nSamples]

        for i, p in enumerate(assImgPoints):
            self.associatedImagePoints[i].insert(0, p)
            self.associatedImagePoints[i] = self.associatedImagePoints[i][:self.nSamples]

    def calcCovariance(self):
        if len(self.poseVecs) > 1:
            poseCov = np.cov(self.poseVecs, rowvar=False)
            camPoseCov = np.cov(self.camPoseVecs, rowvar=False)
        else:
            poseCov = np.zeros((6,6))
            camPoseCov = np.zeros((6,6))

        imageCovs = []
        for points in self.associatedImagePoints:
            imageCovs.append(np.cov(points, rowvar=False))

        return poseCov, camPoseCov, imageCovs

    def calcCovarianceEuler(self):
        if len(self.poseVecs) > 1:
            poseCov = np.cov(self.poseVecsEuler, rowvar=False)
            camPoseCov = np.cov(self.camPoseVecsEuler, rowvar=False)
        else:
            poseCov = np.zeros((6,6))
            camPoseCov = np.zeros((6,6))

        return poseCov, camPoseCov

    def calcAverage(self):
        # This only works for small differences in rotation and 
        # when rotations are not near +-pi
        poseAvg = np.mean(self.poseVecs, axis=0)
        camPoseAvg = np.mean(self.camPoseVecs, axis=0)
        imageAvgs = []
        for points in self.associatedImagePoints:
            imageAvgs.append(np.mean(points, axis=0))

        return poseAvg, camPoseAvg, imageAvgs

class ImagePointsAverageAndCovarianceEstimator:
    """
    Utility class to use when evaluating uncertainty of pose and image points.
    Make sure all the samples are correct (watch out for outliers)
    """
    def __init__(self, nImagePoints, nSamples=100):
        self.associatedImagePoints = [[] for _ in range(nImagePoints)]
        self.nSamples = nSamples

    def add(self, assImgPoints):
        for i, p in enumerate(assImgPoints):
            self.associatedImagePoints[i].insert(0, p)
            self.associatedImagePoints[i] = self.associatedImagePoints[i][:self.nSamples]

    def calcCovariance(self):
        imageCovs = []
        for points in self.associatedImagePoints:
            imageCovs.append(np.cov(points, rowvar=False))

        return imageCovs

    def calcAverage(self):
        imageAvgs = []
        for points in self.associatedImagePoints:
            imageAvgs.append(np.mean(points, axis=0))

        return imageAvgs
