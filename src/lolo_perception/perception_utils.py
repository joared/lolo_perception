
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy import signal

def plotPoseImageInfo(poseImg,
                      titleText,
                      dsPose,
                      camera,
                      featureModel,
                      poseAquired,
                      validOrientationRange,
                      poseEstMethod,
                      roiCnt=None,
                      roiCntUpdated=None,
                      progress=0,
                      fixedAxis=False):


    cv.putText(poseImg, 
               titleText, 
               (int(poseImg.shape[1]/2.5), 45), 
               cv.FONT_HERSHEY_SIMPLEX, 
               2, 
               color=(255,0,255), 
               thickness=2, 
               lineType=cv.LINE_AA)


    cv.putText(poseImg, 
               poseEstMethod, 
               (int(poseImg.shape[1]/2.), 70), 
               cv.FONT_HERSHEY_SIMPLEX, 
               1, 
               color=(0,255,255), 
               thickness=2, 
               lineType=cv.LINE_AA)

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
                       "{}/{}".format(dsPose.attempts, dsPose.combinations), 
                       (roiCntUpdated[3][0]-10, roiCntUpdated[3][1]+30), 
                       cv.FONT_HERSHEY_SIMPLEX, 
                       fontScale=1, 
                       thickness=2, 
                       color=(255,0,255))



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
                    thickness=5,
                    fixedAxis=fixedAxis) 
            
            # Some reprojection uncertainty stuff
            cv.putText(poseImg, 
                      "RMSE: {} < {}".format(round(dsPose.rmse, 2), round(dsPose.rmseMax, 2)), 
                      (20, 200), 
                      cv.FONT_HERSHEY_SIMPLEX, 
                      .8, 
                      color=(0,255,0), 
                      thickness=2, 
                      lineType=cv.LINE_AA)

            cv.putText(poseImg, 
                      "RMSE certainty: {}".format(round(1-dsPose.rmse/dsPose.rmseMax, 2)), 
                      (20, 220), 
                      cv.FONT_HERSHEY_SIMPLEX, 
                      .8, 
                      color=(0,255,0), 
                      thickness=2, 
                      lineType=cv.LINE_AA)

            cv.putText(poseImg, 
                      "Err certainty: {}".format(round(dsPose.reprErrMinCertainty(), 2)), 
                      (20, 240), 
                      cv.FONT_HERSHEY_SIMPLEX, 
                      .8, 
                      color=(0,255,0), 
                      thickness=2, 
                      lineType=cv.LINE_AA)
            
            

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
        #plotMaxReprojection(poseImg, dsPose)
        plotErrorEllipses(poseImg, dsPose)
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
        plotPoints(poseImg, [ls.center for ls in dsPose.associatedLightSources], (255, 0, 0), radius=1)

    plotCrosshair(poseImg, camera)
    

    # progress bar for light source tracker and aquiring pose
    xStart = camera.resolution[1]-30
    yStart = camera.resolution[0]-10
    xEnd = xStart
    yEnd = yStart - 200
    cv.line(poseImg, (xStart, yStart), (xEnd, yEnd), (255,255,255), 15)
    cv.line(poseImg, (xStart, yStart), (xEnd, int(progress*(yEnd - yStart)) + yStart), (0,255,0), 10)

def plotErrorEllipses(img, dsPose, color=(0,0,255), displayReferenceSphere=False):
    for center, pixelCov in zip(dsPose.reProject(), dsPose.pixelCovariances):
        plotErrorEllipse(img, center, pixelCov, confidence=dsPose.chiSquaredConfidence, color=color)

    if displayReferenceSphere:
        X, Y, Z = dsPose.translationVector
        fx = dsPose.camera.cameraMatrix[0, 0]
        fy = dsPose.camera.cameraMatrix[1, 1]
        r = dsPose.featureModel.maxRad
        confidence = dsPose.chiSquaredConfidence
        sigmaScale = 2*np.sqrt(confidence)
        sigma = 2*r/sigmaScale
        sigmaU2 = sigma**2*fx**2*(X**2 + Z**2)/Z**4
        sigmaV2 = sigma**2*fy**2*(Y**2 + Z**2)/Z**4
        sigmaUV2 = sigma**2*fx*fy*X*Y/Z**4
        pixelCovariance = np.array([[sigmaU2, sigmaUV2], 
                                    [sigmaUV2, sigmaV2]])

        projPoint = projectPoints(dsPose.translationVector, dsPose.rotationVector, dsPose.camera, np.array([[0., 0., 0.]]))[0]
        plotErrorEllipse(img, projPoint, pixelCovariance, confidence=confidence)

def plotErrorEllipse(img, center, pixelCovariance, confidence=5.991, color=(0,0,255)):
    # confidence - sqrt(confidence) in std direction
    lambdas, vs = np.linalg.eig(pixelCovariance)
    l1, l2 = lambdas # variances
    v1, v2 = vs
    if l2 < 0.25 or l2 < 0.25:
        print("Variance really low!!!!!!!!!!!!!!!!!!!!!")

    if l1 > l2:
        #dir = -1
        angle = np.rad2deg(np.arctan2(v1[1], v1[0]))
        major = int(round(np.sqrt(confidence*l1))) # has to be int for some reason
        minor = int(round(np.sqrt(confidence*l2))) # has to be int for some reason
    else:
        #dir = 1
        angle = np.rad2deg(np.arctan2(v2[1], v2[0]))
        major = int(round(np.sqrt(confidence*l2))) # has to be int for some reason
        minor = int(round(np.sqrt(confidence*l1))) # has to be int for some reason

    center = int(round(center[0])), int(round(center[1]))
    for l, v in zip((l1, l2), (v1, v2)):
        v = v[0], -v[1]
        l = np.sqrt(confidence*l)
        end = center[0]+v[0]*l, center[1]+v[1]*l
        end = int(round(end[0])), int(round(end[1]))
        cv.line(img, center, end, color=color)
    
    cv.ellipse(img, center, (major, minor), -angle, 0, 360, color=color)

def plotMaxReprojection(img, dsPose, color=(0,0,255)):
    for center, maxErr in zip(dsPose.reProject(), dsPose.reprErrorsMax):

        w = int(round(maxErr[0]*2))
        h = int(round(maxErr[1]*2))
        x = int(round(center[0]-maxErr[0]))
        y = int(round(center[1]-maxErr[1]))

        cv.rectangle(img, (x,y), (x+w, y+h), color, 1)

def plotAxis(img, translationVector, rotationVector, camera, points, scale, color=None, thickness=2, opacity=1, fixedAxis=False):
    points = points[:, :3].copy()

    if fixedAxis:
        points = R.from_rotvec(rotationVector).apply(points)
        rotationVector = rotationVector*0

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
    for d, c in zip((zDir, yDir, xDir), ((255*opacity,0,0), (0,255*opacity,0), (0,0,255*opacity))):
        cx = center[0]
        cy = center[1]
        point1 = (int(round(cx)), int(round(cy)))
        point2 = (int(round(d[0][0][0])), int(round(d[0][0][1])))
        if color is not None:
            cv.line(img, point1, point2, tuple((c*opacity for c in color)), thickness)
        else:
            cv.line(img, point1, point2, c, thickness)

def plotVector(img, vector, translationVector, rotationVector, camera, color=(255,0,0), thickness=2, opacity=1):
    dir, _ = cv.projectPoints(np.array([vector]), 
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
    cx = center[0]
    cy = center[1]
    point1 = (int(round(cx)), int(round(cy)))
    point2 = (int(round(dir[0][0][0])), int(round(dir[0][0][1])))
    cv.line(img, point1, point2, tuple((c*opacity for c in color)), thickness)

def plotPosePoints(img, translationVector, rotationVector, camera, points, color):
    projPoints = projectPoints(translationVector, rotationVector, camera, points)
    plotPoints(img, projPoints, color, radius=1)

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
        if radius == 1:
            cv.circle(img, (x,y), radius, color, 1)
        else:
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

def plotHistogram(img, N, showPeaks=False, showValleys=False, highLightFirstPeakValley=False, highlightPeak=None, facecolor=None, limitAxes=True, alpha=.4):
    hist = cv.calcHist([img], [0], None, [256], [0,256])
    hist = hist.ravel()
    plt.plot(hist, c=facecolor)
    plt.fill_between(range(len(hist)), hist, alpha=alpha, interpolate=True, facecolor=facecolor)

    if showPeaks or showValleys or highLightFirstPeakValley or highlightPeak is not None:
        handles = []

        peakLine = None
        peaks = signal.find_peaks(hist)[0]
        for i, peak in enumerate(peaks):
            if peak >= N:
                if highlightPeak is not None and peak == highlightPeak+1 or highLightFirstPeakValley and i==len(peaks)-1:
                    peakLine = plt.axvline(peak, ymin=0, ymax=1, c="g", label="peak")
                elif showPeaks:   
                    peakLine = plt.axvline(peak, ymin=0, ymax=hist[peak]/max(hist[N:]), c="g", label="peak")
        if peakLine is not None:
            handles.append(peakLine)

        valleyLine = None
        valleys = signal.find_peaks(-hist)[0]
        for j, peak in enumerate(valleys):
            if peak >= N:
                if highlightPeak is not None and peak == highlightPeak+1 or highLightFirstPeakValley and j==len(valleys)-1:
                    valleyLine = plt.axvline(peak, ymin=0, ymax=1, c="orange", label="valley")
                elif showValleys:   
                    valleyLine = plt.axvline(peak, ymin=0, ymax=hist[peak]/max(hist[N:]), c="orange", label="valley")
        if valleyLine is not None:
            handles.append(valleyLine)

        plt.legend(handles=handles)

    if limitAxes:
        plt.xlim([N, 256])
        plt.ylim([0, max(hist[N:])])
    
    plt.ylabel("Frequency")
    plt.xlabel("Intensity")

def regionOfInterest(featurePointsGuess, wMargin, hMargin):
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

    return (x, y, w, h), roiCnt

def imageROI(img, imgPoints, margin):
    (x, y, w, h), roiCnt = regionOfInterest([[u,v] for u,v in imgPoints], margin, margin)
    x = max(0, x)
    x = min(img.shape[1]-1, x)
    y = max(0, y)
    y = min(img.shape[0]-1, y)
    offset = (x,y)
    #roiMask = np.zeros(imgRect.shape[:2], dtype=np.uint8)
    #cv.drawContours(roiMask, [roiCnt], 0, (255,255,255), -1)
    #imgRectROI = cv.bitwise_and(imgRect, imgRect, mask=roiMask)

    imgROI = img[y:y+h, x:x+w]

    return imgROI, roiCnt, (x, y, w, h)

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
