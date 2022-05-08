import numpy as np
import cv2 as cv
import time
from scipy.spatial.transform import Rotation as R

def adjustToCenter(center, p):
    return (p[0]+center[0], p[1]+center[1])

def toValidPixel(point):
    return (int(round(point[0])), int(round(point[1])))

def calcErrPoint(p, deltaE):
    px, pz = p[0], p[1]
    pNorm = np.linalg.norm(p)

    # TODO: if pNorm >> deltaE, np.arcsin(...) can be neglected
    if px < 0:
        #beta = np.arctan2(-px, -pz)
        #theta = -beta + np.arcsin( deltaE/pNorm )
        beta = np.arctan2(px, pz) + np.pi
        theta = -beta + np.arcsin( deltaE/pNorm )
    
    else:
        beta = np.arctan2(px, pz)
        theta = -beta - np.arcsin( deltaE/pNorm )

    deltaEX = deltaE*np.cos(theta)
    deltaEZ = deltaE*np.sin(theta)
    return (p[0]+deltaEX, p[1]+deltaEZ)

def maxReprojectionError(point, f, deltaE):
    newPoint = calcErrPoint(point, deltaE)
    reprojErr = f*point[0]/float(point[1])- f*newPoint[0]/float(newPoint[1])
    return reprojErr

def plot2DReprojection(img, f, fScale, deltaE, center, points, calcErrMethod):
    for point in points:
        deltaEVec = calcErrMethod(point, deltaE)

        newPoint = deltaEVec
        
        rx1 = fScale*float(f)*point[0]/point[1], fScale*f
        rx2 = fScale*float(f)*newPoint[0]/newPoint[1], fScale*f

        # detected point
        point = adjustToCenter(center, point)
        cv.circle(img, point, 1, (0,0,255), 1)
        cv.circle(img, point, int(round(deltaE)), (0,0,255/2), 1)
        cv.line(img, center, point, color=(0,0,255/2))

        # reprojection point
        rx1 = adjustToCenter(center, rx1)
        rx1 = toValidPixel(rx1)
        cv.line(img, (rx1[0], rx1[1]-5), (rx1[0], rx1[1]+5), color=(0,0,255))

        # worst true point
         #(int(round(newPoint[0])), int(round(newPoint[1])))
        newPoint = adjustToCenter(center, newPoint)
        newPoint = toValidPixel(newPoint)
        cv.circle(img, newPoint, 1, (0,255,0), 1)
        cv.line(img, center, newPoint, color=(0,255/2,0))

        # reprojection new point
        rx2 = adjustToCenter(center, rx2)
        rx2 = toValidPixel(rx2)
        cv.line(img, (rx2[0], rx2[1]-5), (rx2[0], rx2[1]+5), color=(0,255,0))

def plotCamera(img, center, f, fScale, xMax):
    imgPlane = adjustToCenter(center, (-fScale*xMax/2, f*fScale)), adjustToCenter(center, (fScale*xMax/2-1, f*fScale))
    imgPlane = toValidPixel(imgPlane[0]), toValidPixel(imgPlane[1])
    cv.line(img, imgPlane[0], imgPlane[1], color=(0,0,255))
    focalLine = center, toValidPixel((center[0], center[1]+f*fScale))
    cv.line(img, focalLine[0], focalLine[1], color=(255,0,0))

def plotReprojection(points, cameraResolution, f, deltaE, rangeMeter, fScale=1./30):
    #zMax = (img.shape[0]-1)
    maxPixel = cameraResolution[1] # TODO: fx only considered, fy should be too
    center = maxPixel/2, 0

    img = np.zeros((cameraResolution[0], cameraResolution[1], 3), dtype=np.uint8)
    plotCamera(img, center, f, fScale, maxPixel)

    meter = cameraResolution[0]/rangeMeter
    deltaE *= meter
    points *= meter
    points = [toValidPixel(p) for p in points]
    plot2DReprojection(img, f, fScale, deltaE, center, points, calcErrPoint)

    img = cv.flip(img, 0)
    return img
    

###############


def NEW_calcPoseReprojectionRMSEThreshold(translationVec, rotationVec, camera, featureModel, showImg=False):
    rotMat = R.from_rotvec(rotationVec).as_dcm()

    reprErrs = []
    pointsX = []
    for fp in featureModel.features:
        point3D = np.matmul(rotMat, fp.transpose()) + translationVec.transpose()
        pointsX.append((point3D[0], point3D[2]))
        
        maxReprojErrX = maxReprojectionError((point3D[0], point3D[2]), camera.cameraMatrix[0, 0], deltaE=featureModel.uncertainty)
        maxReprojErrY = maxReprojectionError((point3D[1], point3D[2]), camera.cameraMatrix[1, 1], deltaE=featureModel.uncertainty)
        #reprErrs.append(np.linalg.norm([maxReprojErrX, maxReprojErrY]))

        reprErrs.append(max(abs(maxReprojErrX), abs(maxReprojErrY)))

    reprErrs = np.array(reprErrs)
    maxRMSE = np.sum(reprErrs)/len(reprErrs)

    if showImg:
        rangeMeter = featureModel.maxRad*40 # cover 40*max rad of the feature model
        img = plotReprojection(np.array(pointsX), 
                                camera.resolution, 
                                camera.cameraMatrix[0, 0], 
                                featureModel.uncertainty, 
                                rangeMeter=rangeMeter,
                                fScale=1./20)
        img = cv.flip(img, 0)
        
        maxPixel = camera.resolution[1]
        meter = camera.resolution[0]/rangeMeter
        center = maxPixel/2, 0

        center2D = translationVec[0]*meter, translationVec[2]*meter
        center2D = toValidPixel(adjustToCenter(center, center2D))
        # x-axis
        xAxis = translationVec + np.matmul(rotMat, [featureModel.maxX, 0, 0])
        xAxis = xAxis[0]*meter, xAxis[2]*meter
        xAxis = toValidPixel(adjustToCenter(center, xAxis))
        cv.line(img, center2D, xAxis, color=(0,0,255))

        yAxis = translationVec + np.matmul(rotMat, [0, featureModel.maxX, 0])
        yAxis = yAxis[0]*meter, yAxis[2]*meter
        yAxis = toValidPixel(adjustToCenter(center, yAxis))
        cv.line(img, center2D, yAxis, color=(0,255,0))

        zAxis = translationVec + np.matmul(rotMat, [0, 0, featureModel.maxX])
        zAxis = zAxis[0]*meter, zAxis[2]*meter
        zAxis = toValidPixel(adjustToCenter(center, zAxis))
        cv.line(img, center2D, zAxis, color=(255,0,0))

        img = cv.flip(img, 0)
        cv.imshow("reprojection", img)
        cv.waitKey(1)

    return maxRMSE

def calcPoseReprojectionThresholds(translationVec, rotationVec, camera, featureModel):
    rotMat = R.from_rotvec(rotationVec).as_dcm()

    reprErrs = []
    pointsX = []
    for fp in featureModel.features:
        point3D = np.matmul(rotMat, fp.transpose()) + translationVec.transpose()
        pointsX.append((point3D[0], point3D[2]))
        
        maxReprojErrX = maxReprojectionError((point3D[0], point3D[2]), camera.cameraMatrix[0, 0], deltaE=featureModel.uncertainty)
        maxReprojErrY = maxReprojectionError((point3D[1], point3D[2]), camera.cameraMatrix[1, 1], deltaE=featureModel.uncertainty)
        reprErrs.append([abs(maxReprojErrX), abs(maxReprojErrY)])

    reprErrs = np.array(reprErrs)
    return reprErrs

def calcPoseReprojectionRMSEThreshold(translationVec, rotationVec, camera, featureModel, showImg=False):
    rotMat = R.from_rotvec(rotationVec).as_dcm()

    reprErrs = []
    pointsX = []
    for fp in featureModel.features:
        point3D = np.matmul(rotMat, fp.transpose()) + translationVec.transpose()
        pointsX.append((point3D[0], point3D[2]))
        
        maxReprojErrX = maxReprojectionError((point3D[0], point3D[2]), camera.cameraMatrix[0, 0], deltaE=featureModel.uncertainty)
        maxReprojErrY = maxReprojectionError((point3D[1], point3D[2]), camera.cameraMatrix[1, 1], deltaE=featureModel.uncertainty)
        reprErrs.append([maxReprojErrX, maxReprojErrY])

    reprErrs = np.array(reprErrs)
    maxRMSE = np.sqrt(np.sum(reprErrs**2)/np.product(reprErrs.shape))

    if showImg:
        rangeMeter = featureModel.maxRad*40 # cover 40*max rad of the feature model
        img = plotReprojection(np.array(pointsX), 
                                camera.resolution, 
                                camera.cameraMatrix[0, 0], 
                                featureModel.uncertainty, 
                                rangeMeter=rangeMeter,
                                fScale=1./20)
        img = cv.flip(img, 0)
        
        maxPixel = camera.resolution[1]
        meter = camera.resolution[0]/rangeMeter
        center = maxPixel/2, 0

        center2D = translationVec[0]*meter, translationVec[2]*meter
        center2D = toValidPixel(adjustToCenter(center, center2D))
        # x-axis
        xAxis = translationVec + np.matmul(rotMat, [featureModel.maxX, 0, 0])
        xAxis = xAxis[0]*meter, xAxis[2]*meter
        xAxis = toValidPixel(adjustToCenter(center, xAxis))
        cv.line(img, center2D, xAxis, color=(0,0,255))

        yAxis = translationVec + np.matmul(rotMat, [0, featureModel.maxX, 0])
        yAxis = yAxis[0]*meter, yAxis[2]*meter
        yAxis = toValidPixel(adjustToCenter(center, yAxis))
        cv.line(img, center2D, yAxis, color=(0,255,0))

        zAxis = translationVec + np.matmul(rotMat, [0, 0, featureModel.maxX])
        zAxis = zAxis[0]*meter, zAxis[2]*meter
        zAxis = toValidPixel(adjustToCenter(center, zAxis))
        cv.line(img, center2D, zAxis, color=(255,0,0))

        img = cv.flip(img, 0)
        cv.imshow("reprojection", img)
        cv.waitKey(1)

    return maxRMSE

if __name__ == "__main__":
    cameraResolution = (720, 1280)
    fx = 1406
    fy = 1411
    deltaE = 0.008545
    deltaE = 0.5
    points = np.array([(-deltaE/2, 6), (deltaE/2, 6)])

    for p in points:
        reprErr = maxReprojectionError(p, fx, deltaE)
        print("Reprojection error: {}".format(reprErr))

    img = plotReprojection(points, cameraResolution, fx, deltaE, rangeMeter=10, fScale=1./20)

    imgTemp = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    zeroImg = imgTemp*0
    imgTemp[np.where(imgTemp == zeroImg)] = 255
    imgTemp = cv.cvtColor(imgTemp, cv.COLOR_GRAY2BGR)
    #img = cv.bitwise_or(img, imgTemp)

    cv.imshow("reprojection", img)
    #cv.imshow("temp", imgTemp)
    cv.waitKey(0)