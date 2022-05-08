import cv2 as cv
import numpy as np

def convertImagePoints(imagePoints, K, KNew):
    u, v = imagePoints

    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    fxNew = KNew[0,0]
    fyNew = KNew[1,1]
    cxNew = KNew[0,2]
    cyNew = KNew[1,2]

    uNew = (u-cx)/fx*fxNew + cxNew
    vNew = (v-cy)/fy*fyNew + cyNew
    
    return (int(round(uNew)), int(round(vNew))) 

def distortImage(img, K, D, imgSize):
    # distortion maps
    # I think we use K here, not P
    mapx, mapy = cv.initUndistortRectifyMap(K, D, np.eye(3), K, imgSize, m1type=cv.CV_32FC1)

    padMinX = int(-mapx.min())
    padMaxX = int(np.ceil(mapx.max() - imgSize[0]))

    padMinY = int(-mapy.min())
    padMaxY = int(np.ceil(mapy.max() - imgSize[1]))

    # tried this for barrel distortion, seems to work for padded distorted image 
    # but something is wrong for the undistorted image using K
    padMinX = max(0, padMinX)
    padMaxX = max(0, padMaxX)
    padMinY = max(0, padMinY)
    padMaxY = max(0, padMaxY)

    distImg = np.zeros(img.shape, dtype=np.uint8)
    distImgPadded = np.zeros((img.shape[0]+padMinY+padMaxY, img.shape[1]+padMinX+padMaxX, img.shape[2]), dtype=np.uint8)

    for u in range(imgSize[0]):
        for v in range(imgSize[1]):
            uDist = mapx[v, u]
            vDist = mapy[v, u]
            uDist = int(round(uDist))
            vDist = int(round(vDist))
            if uDist > 0 and uDist < distImg.shape[1] and vDist > 0 and vDist < distImg.shape[0]:
                distImg[vDist, uDist] = img[v, u]

            uDist += padMinX
            vDist += padMinY
            if uDist > 0 and uDist < distImgPadded.shape[1] and vDist > 0 and vDist < distImgPadded.shape[0]:
                distImgPadded[vDist, uDist] = img[v, u]


    # draw rectangle in padded distorted image indicating what the real distorted image is
    cv.rectangle(distImgPadded, (padMinX, padMinY), (distImg.shape[1]+padMaxX, distImg.shape[0]+padMaxY), color=(0,255,0), thickness=2)

    return distImg, distImgPadded

def getCircleImage():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)

    N = 50
    for i in range(N):
        for j in range(N):
            u = i*img.shape[1]/float(N)
            u = int(round(u))

            v = j*img.shape[0]/float(N)
            v = int(round(v))

            cv.circle(img, (u,v), 5, (255,255,255), -1)
            
    return img

if __name__ == "__main__":
    img = np.zeros((720, 1280), dtype=np.uint8)

    N = 50
    for i in range(N+1):
        # vertical lines
        cv.line(img, (0, (img.shape[0]-1)*i/N), (img.shape[1]-1, (img.shape[0]-1)*i/N), color=(255), thickness=5)

        # horisontal lines
        cv.line(img, ((img.shape[1]-1)*i/N, 0), ((img.shape[1]-1)*i/N, img.shape[0]-1), color=(255), thickness=5)

    img = np.ones((720, 1280, 3), dtype=np.uint8)
    img[:, :, 0] = 255
    img[:, :, 2] = 255
    img = cv.imread("/home/joar/LoLo/lolo_ws/src/lolo_perception/image_dataset/ice_msg_597.png")
    #img = getCircleImage()

    K = np.array([[1400., 0., 639.],
                [0., 1400., 359.], 
                [0., 0.,      1]])

    D = np.array([1., 0, 0., 0.])



    imgSize = img.shape[1], img.shape[0]
    distImg, distImgPadded = distortImage(img, K, D, imgSize)

    # alpha = 0
    alpha0P, alpha0ROI = cv.getOptimalNewCameraMatrix(K, D, imgSize, 0, imgSize)
    imgUndist1 = cv.undistort(distImg, K, D, newCameraMatrix=alpha0P)
    print("alpha0 roi", alpha0ROI)
    # alpha = 1
    alpha1P, alpha1ROI = cv.getOptimalNewCameraMatrix(K, D, imgSize, 1, imgSize)
    print("alpha1 roi", alpha1ROI)
    alpha1ROI = (0, 0, imgSize[0], imgSize[1])
    imgUndist2 = cv.undistort(distImg, K, D, newCameraMatrix=alpha1P)


    # don't use new camera matrix
    imgUndist3 = cv.undistort(distImg, K, D, newCameraMatrix=K)

    rectangleColor = (0,255,0)
    for undistImg, alpha, roi in ((imgUndist1, 0, alpha0ROI), (imgUndist2, 1,alpha1ROI)):
        cv.putText(undistImg, "alpha = {}".format(alpha), (roi[0], roi[1]+20), cv.FONT_HERSHEY_SIMPLEX, 0.7, rectangleColor, 2)
        cv.rectangle(undistImg, tuple(roi[:2]), (roi[0]+roi[2], roi[1]+roi[3]), color=rectangleColor, thickness=2)

    alpha0ROIK = [0, 0, 0, 0]
    alpha0ROIK[:2] = convertImagePoints(alpha0ROI[:2], alpha0P, K)
    wuv, huv = convertImagePoints((alpha0ROI[0]+alpha0ROI[2], alpha0ROI[1]+alpha0ROI[3]), alpha0P, K)
    alpha0ROIK[2] = wuv-alpha0ROIK[0]
    alpha0ROIK[3] = huv-alpha0ROIK[1]

    alpha1ROIK = [0, 0, 0, 0]
    alpha1ROIK[:2] = convertImagePoints(alpha1ROI[:2], alpha1P, K)
    wuv, huv = convertImagePoints((alpha1ROI[0]+alpha1ROI[2]-1, alpha1ROI[1]+alpha1ROI[3]-1), alpha1P, K)
    alpha1ROIK[2] = wuv-alpha1ROIK[0]+1
    alpha1ROIK[3] = huv-alpha1ROIK[1]+1

    imgUndist3Alpha = imgUndist3.copy()
    for alpha, roi in ((0, alpha0ROIK), (1, alpha1ROIK)):
        cv.putText(imgUndist3Alpha, "alpha = {}".format(alpha), (roi[0], roi[1]+20), cv.FONT_HERSHEY_SIMPLEX, 0.7, rectangleColor, 2)
        cv.rectangle(imgUndist3Alpha, tuple(roi[:2]), (roi[0]+roi[2], roi[1]+roi[3]), color=rectangleColor, thickness=2)


    print("Kamera matrix", K)
    print("New camera matrix, alpha=0", alpha0P)
    print("New camera matrix, alpha=1", alpha1P)

    cv.imshow("image", img)
    cv.imshow("distorted image", distImg)
    cv.imshow("distorted image padded", cv.resize(distImgPadded, imgSize))
    cv.imshow("undistorted image, alpha=0", imgUndist1)
    cv.imshow("undistorted image, alpha=1", imgUndist2)
    cv.imshow("undistorted image, K", imgUndist3)
    cv.imshow("undistorted image, K with alpha rectangles", imgUndist3Alpha)
    cv.waitKey(0)
    cv.destroyAllWindows()