from configparser import Interpolation
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
    
    # inverse distortion maps
    # i.e undistortion maps
    #D = np.array([1., 0., 0., 0.])
    
    P, roi = cv.getOptimalNewCameraMatrix(K, D, imgSize, 1, imgSize)

    mapx, mapy = cv.initUndistortRectifyMap(K, D, np.eye(3), P, imgSize, m1type=cv.CV_32FC1)
    newImgSize = getNewImageSize(mapx, mapy, imgSize)

    
    k1 = D[0]
    k1Prime = k1/(1.+k1)

    mapx, mapy = cv.initUndistortRectifyMap(K, np.array([-k1Prime, 0., 0., 0.]), np.eye(3), P, newImgSize, m1type=cv.CV_32FC1)
    imgDist = cv.remap(img, mapx, mapy, interpolation=cv.INTER_LINEAR)

    return imgDist

def getNewImageSize(mapx, mapy, imgSize):
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

    return (imgSize[0]+padMinX+padMaxX, imgSize[1]+padMinY+padMaxY)

def getImagePadding(mapx, mapy, imgSize):
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

    return padMinX, padMaxX, padMinY, padMaxY

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
    img = np.ones((720, 1280), dtype=np.uint8)*255

    N = 30
    for i in range(N+1):
        # vertical lines
        cv.line(img, (0, (img.shape[0]-1)*i/N), (img.shape[1]-1, (img.shape[0]-1)*i/N), color=(0), thickness=2)

        # horisontal lines
        cv.line(img, ((img.shape[1]-1)*i/N, 0), ((img.shape[1]-1)*i/N, img.shape[0]-1), color=(0), thickness=2)

    #img = np.ones((720, 1280, 3), dtype=np.uint8)
    #img[:, :, 0] = 255
    #img[:, :, 2] = 255
    #img = cv.imread("/home/joar/LoLo/lolo_ws/src/lolo_perception/image_dataset/ice_msg_597.png")
    #img = getCircleImage()

    K = np.array([[1400., 0., 639.],
                [0., 1400., 359.], 
                [0., 0.,      1]])

    k1 = 0.5
    k1Prime = k1/(1.+k1) # not right
    print(k1)
    print(k1Prime)

    D = np.array([1., 0., 0., 0.])


    imgSize = img.shape[1], img.shape[0]

    imgDist = distortImage(img, K, D, imgSize)

    imgUndist = cv.undistort(imgDist, K, k1*D)

    # undistorted image represents distorted image
    #imgDist = cv.undistort(distImg, K, -D, newCameraMatrix=K) 

    cv.imshow("image", img)

    cv.imshow("distorted", imgDist)
    cv.imshow("undistorted", imgUndist)
    cv.waitKey(0)
    cv.destroyAllWindows()