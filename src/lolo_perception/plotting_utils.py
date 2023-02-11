
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy import signal
from lolo_perception.pose_estimation import projectPoints, reprojectionError

def plotPoseImageInfo(poseImg,
                      titleText,
                      dsPose,
                      camera,
                      featureModel,
                      poseAquired,
                      validOrientationRange,
                      poseEstMethod,
                      piChartArgs,  
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

    if dsPose:
        axisColor = None
        roiColor = (0, 255, 0)
        if not validOrientation:
            axisColor = (0, 0, 255)
            roiColor = (0, 0, 255)
        if roiCnt is not None:
            cv.drawContours(poseImg, [roiCnt], -1, (100,100,100), 3)
        if roiCntUpdated is not None:
            
            drawProgressROI(poseImg, progress, roiCntUpdated, roiColor)
            
            # Number of consecutive estimated poses
            cv.putText(poseImg, 
                       "#{}".format(dsPose.detectionCount), 
                       (roiCntUpdated[0][0]-10, roiCntUpdated[0][1]-10), 
                       cv.FONT_HERSHEY_SIMPLEX, 
                       fontScale=1, 
                       thickness=2, 
                       color=(0,255,0))

            # Number of attempts
            cv.putText(poseImg, 
                       "{}/{}".format(dsPose.attempts, dsPose.combinations), 
                       (roiCntUpdated[1][0]-70, roiCntUpdated[1][1]-10), 
                       cv.FONT_HERSHEY_SIMPLEX, 
                       fontScale=1, 
                       thickness=2, 
                       color=(255,0,255))

            # Warning if light sources are overlapping
            if any([ls.isOverlapping() for ls in dsPose.associatedLightSources]):
                cv.putText(poseImg, 
                           "Overlapping!", 
                           (roiCntUpdated[3][0]+10, roiCntUpdated[3][1]-15), 
                           cv.FONT_HERSHEY_SIMPLEX, 
                           fontScale=1, 
                           thickness=2, 
                           color=(0,0,255))

            # mahanalobis distance meter
            mahaDistRatio = 1
            if dsPose.mahaDist is not None and dsPose.mahaDistThresh:
                mahaDistRatio = dsPose.mahaDist/dsPose.mahaDistThresh

            xStart = roiCntUpdated[2][0]+3
            yStart = roiCntUpdated[2][1]+2
            xEnd = xStart + 10
            yEnd = roiCntUpdated[1][1]
            
            cv.rectangle(poseImg, 
                         (xStart, yStart), 
                         (xEnd, int(mahaDistRatio*(yEnd - yStart)) + yStart), 
                         color=(0, 0, 255) if dsPose.mahaDist is not None else (0, 140, 255), 
                         thickness=-1)

        if poseAquired and dsPose: # redundancy
            plotAxis(poseImg, 
                    dsPose.translationVector, 
                    dsPose.rotationVector, 
                    camera, 
                    featureModel.features, 
                    featureModel.maxRad, # scaling for the axis shown
                    color=axisColor,
                    thickness=5,
                    fixedAxis=fixedAxis) 

            plotPoseInfo(poseImg, 
                         dsPose.translationVector, 
                         dsPose.rotationVector,
                         yawColor=(0,255,0) if validYaw else (0,0,255),
                         pitchColor=(0,255,0) if validPitch else (0,0,255),
                         rollColor=(0,255,0) if validRoll else (0,0,255))

            # TODO: plot uncertainty based on reprojection error

    if dsPose:
        #plotMaxReprojection(poseImg, dsPose)
        
        plotErrorEllipses(poseImg, 
                          dsPose,
                          displayReferenceSphere=True)
        
        plotPosePoints(poseImg, 
                    dsPose.translationVector, 
                    dsPose.rotationVector, 
                    camera, 
                    featureModel.features, 
                    color=(0, 0, 255))
        plotPoints(poseImg, [ls.center for ls in dsPose.associatedLightSources], (255, 0, 0), radius=1)

    plotCrosshair(poseImg, camera)

    piChartSize = 35
    plotPiChart(poseImg, (int(poseImg.shape[1]*4/5), 45+piChartSize*2), piChartSize, **piChartArgs)

    return poseImg

def plotPoseImageInfoSimple(poseImg, dsPose, poseAquired, roiCnt, progress):
    if dsPose:
        if poseAquired:
            plotAxis(poseImg, 
                    dsPose.translationVector, 
                    dsPose.rotationVector, 
                    dsPose.camera, 
                    dsPose.featureModel.features, 
                    dsPose.featureModel.maxRad,
                    color=None,
                    thickness=5,
                    fixedAxis=False)
        if roiCnt is not None:
            drawProgressROI(poseImg, progress, roiCnt, (255,255,255))

    return poseImg

def drawProgressROI(poseImg, progress, roiCnt, roiColor, clockwise=True):
    """
    roiCnt[0][0/1] - top left x/y
    roiCnt[1][0/1] - top right x/y
    roiCnt[2][0/1] - bottom right x/y
    roiCnt[3][0/1] - bottom left x/y
    """
    if progress == 1:
        cv.drawContours(poseImg, [roiCnt], -1, roiColor, 3)
        return

    cv.drawContours(poseImg, [roiCnt], -1, (200,200,200), 7)

    w = roiCnt[1][0] - roiCnt[0][0]
    h = roiCnt[3][1] - roiCnt[0][1]
    tot = 2*w + 2*h
    w_norm = w/float(tot)
    h_norm = h/float(tot)

    progressTemp = progress
    for i, l in enumerate([w_norm, h_norm, w_norm, h_norm]):
        p1 = roiCnt[i][0], roiCnt[i][1]
        if i == 3:
            p2 = roiCnt[0][0], roiCnt[0][1]
        else:
            p2 = roiCnt[i+1][0], roiCnt[i+1][1]

        if progressTemp < l:
            if i == 0:
                p2 = p1[0]+progressTemp*tot, p2[1]
            elif i == 1:
                p2 = p2[0], p1[1]+progressTemp*tot
            elif i == 2:
                p2 = p1[0]-progressTemp*tot, p2[1]
            else:
                p2 = p2[0], p1[1]-progressTemp*tot

        p2 = int(p2[0]), int(p2[1])
        dimColor = (roiColor[0]*0.6, roiColor[1]*0.6, roiColor[2]*0.6)
        cv.line(poseImg, p1, p2, dimColor, 3)
        progressTemp -= l

        if progressTemp < 0:
            cv.circle(poseImg, p2, 15, roiColor, -1)
            return
        

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

def plotErrorEllipse(img, center, pixelCovariance, confidence=5.991, color=(0,0,255), displayAxis=False):
    # confidence - sqrt(confidence) in std direction
    lambdas, vs = np.linalg.eig(pixelCovariance)
    l1, l2 = lambdas # variances
    v1, v2 = vs
    if l2 < 0.25 or l2 < 0.25:
        #print("Variance really low!!!!!!!!!!!!!!!!!!!!!")
        pass

    if l1 > l2:
        #dir = -1
        angle = np.rad2deg(np.arctan2(v1[1], v1[0]))
        major = np.sqrt(confidence*l1) # has to be int for some reason
        minor = np.sqrt(confidence*l2) # has to be int for some reason
    else:
        #dir = 1
        angle = np.rad2deg(np.arctan2(v2[1], v2[0]))
        major = np.sqrt(confidence*l2) # has to be int for some reason
        minor = np.sqrt(confidence*l1) # has to be int for some reason

    center = int(round(center[0])), int(round(center[1]))

    if displayAxis:
        for l, v in zip((l1, l2), (v1, v2)):
            v = v[0], -v[1]
            l = np.sqrt(confidence*l)
            end = center[0]+v[0]*l, center[1]+v[1]*l
            end = int(round(end[0])), int(round(end[1]))
            cv.line(img, center, end, color=color)
    
    cv.ellipse(img, center, (int(round(major)), int(round(minor))), -angle, 0, 360, color=color)
    
    return major, minor, -angle

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
    cv.putText(img, "Range: {} {}".format(distance, unit), org, cv.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 0), thickness=1, lineType=cv.LINE_AA)
    org = (org[0], org[1]+inc)
    cv.putText(img, "X: {} {}".format(round(translationVector[0], 2), unit), org, cv.FONT_HERSHEY_SIMPLEX, .8, color=(0, 255, 0), thickness=1, lineType=cv.LINE_AA)
    org = (org[0], org[1]+inc)
    cv.putText(img, "Y: {} {}".format(round(translationVector[1], 2), unit), org, cv.FONT_HERSHEY_SIMPLEX, .8, color=(0, 255, 0), thickness=1, lineType=cv.LINE_AA)
    org = (org[0], org[1]+inc)
    cv.putText(img, "Z: {} {}".format(round(translationVector[2], 2), unit), org, cv.FONT_HERSHEY_SIMPLEX, .8, color=(0, 255, 0), thickness=1, lineType=cv.LINE_AA)
    
    thickness = 1
    width = 200
    org = (org[0], org[1]+inc+5)
    plotOrientationBar(img, org, "Yaw", (0,255,0), yaw, 90, width=width, height=inc, thickness=thickness)
    org = (org[0], org[1]+inc+thickness)
    plotOrientationBar(img, org, "Pitch", (0,0,255), pitch, 90, width=width, height=inc, thickness=thickness)
    org = (org[0], org[1]+inc+thickness)
    plotOrientationBar(img, org, "Roll", (255,0,0), roll, 90, width=width, height=inc, thickness=thickness)


def plotOrientationBar(img, org, text, color, val, valRange, width, height, thickness=1):
    text = str(round(val,2))
    font = cv.FONT_HERSHEY_SIMPLEX
    fontScale = 0.8
    textThickness = 1
    textSize = cv.getTextSize(text, font, fontScale, textThickness)
    textWidth = textSize[0][0]
    textHeight = textSize[0][1]
    cv.rectangle(img, 
                (org[0],org[1]),
                (org[0]+width, org[1]-height),
                color=color, 
                thickness=thickness)
    center = (org[0]+int(width/2.0), org[1])
    barWidth = int(val/float(valRange)*width/2.0)
    cv.rectangle(img, 
                (center[0],center[1]),
                (center[0]+barWidth, center[1]-height),
                color=color, 
                thickness=-1)

    textOrg = (org[0]+int((width-textWidth)/2.0), org[1]-int((height-textHeight)/2.0))
    cv.putText(img, "{:7<s}".format(text), textOrg, font, fontScale, color=(255,255,255), thickness=textThickness, lineType=cv.LINE_AA)
    #textOrg = (org[0], org[1]-int(height/5.0))
    #cv.putText(img, "{:7<s} {:4=+.1f} deg".format(text, round(val, 2)), textOrg, cv.FONT_HERSHEY_SIMPLEX, .5, color=(255,255,255), thickness=1, lineType=cv.LINE_AA)

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


def plotPiChart(img, center, size=20, **kwargs):

    colors = ((255, 0, 0),
              (0, 255, 0),
              (0, 0, 255),
              (255, 255, 0),
              (255, 0, 255),
              (0, 255, 255))

    assert len(colors) >= len(kwargs), "Cannot create pi chart with more values than {} < {}".format(len(colors), len(kwargs))

    tot = sum(kwargs.values())

    cv.rectangle(img, 
                (center[0], center[1]+size), 
                (center[0]+size*6, center[1]-size), 
                color=(100,100,100), 
                thickness=-1)
    cv.circle(img, center, int(size*1.1), (100,100,100), -1)
    startAngle = -90
    for i, (k, c) in enumerate(zip(kwargs, colors)):
        val = kwargs[k]
        angle = val/float(tot) * 360
        cv.ellipse(img, center, (size, size), 0, startAngle, startAngle+angle, c, -1)
        margin = 10
        cv.putText(img, 
                   k, 
                   (center[0] + size + margin, center[1] - int(size - size/(len(kwargs)-1) - 2*size*i/len(kwargs))), 
                   cv.FONT_HERSHEY_SIMPLEX, 
                   .7, 
                   color=c, 
                   thickness=1, 
                   lineType=cv.LINE_AA)

        startAngle += angle

def plotFPS(img, fps, fpsVirtual=None):
    cv.putText(img, 
                "FPS {}".format(round(fps, 1)), 
                (int(img.shape[1]*4/5), 25), 
                cv.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                color=(0,255,0), 
                thickness=2, 
                lineType=cv.LINE_AA)

    if fpsVirtual is not None:
        cv.putText(img, 
                   "Virtual FPS {}".format(round(fpsVirtual, 1)), 
                   (int(img.shape[1]*4/5), 45), 
                   cv.FONT_HERSHEY_SIMPLEX, 
                   0.7, 
                   color=(0,255,0), 
                   thickness=2, 
                   lineType=cv.LINE_AA)

    return img


if __name__ == "__main__":
    pass