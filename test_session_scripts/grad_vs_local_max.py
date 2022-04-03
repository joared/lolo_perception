#!/usr/bin/env python
import cv2 as cv
import numpy as np

if __name__ == "__main__":
    from lolo_perception.feature_extraction import GradientFeatureExtractor, findPeaksDilation, circularKernel

    realImg = True
    if realImg:
        img = cv.imread("/home/joar/LoLo/lolo_ws/src/lolo_perception/image_dataset/171121_straight_test_frame_714.png")
        #img = cv.imread("/home/joar/LoLo/lolo_ws/src/lolo_perception/image_dataset/171121_straight_test_frame_542.png")
    else:
        img = cv.imread("/home/joar/LoLo/lolo_ws/src/lolo_perception/image_dataset/sim_bag_msg_1.png_screenshot_28.03.2022.png")
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #gradImg = cv.medianBlur(gradImg, 5)

    # ROI
    if realImg:
        x, y, w, h = 890, 820, 200, 185 # real
        xCrop, yCrop, wCrop, hCrop = 500, 200, 1000, 880
    else:
        x, y, w, h = 517, 347, 100, 90 # sim
        xCrop, yCrop, wCrop, hCrop = 0, 0, gray.shape[1], gray.shape[0]

    roiMask = np.zeros(gray.shape, dtype=np.uint8)
    cv.rectangle(roiMask, (x, y), (x+w, y+h), (255), -1)

    g = GradientFeatureExtractor(None, 4, kernelSize=3)
    gradImg = g(gray)
    gradThresh = np.max(gradImg)*0.2
    _, gradImgThresh = cv.threshold(gradImg, gradThresh, 256, cv.THRESH_TOZERO)
    

    kernel = circularKernel(11)
    localMaxImg = findPeaksDilation(gray, kernel)
    localMaxThresh = 230
    _, localMaxImgThresh = cv.threshold(localMaxImg, localMaxThresh, 256, cv.THRESH_TOZERO)


    # convert gray to BGR
    gradImg = cv.cvtColor(gradImg, cv.COLOR_GRAY2BGR)
    gradImgThresh = cv.cvtColor(gradImgThresh, cv.COLOR_GRAY2BGR)
    localMaxImg = cv.cvtColor(localMaxImg, cv.COLOR_GRAY2BGR)
    localMaxImgThresh = cv.cvtColor(localMaxImgThresh, cv.COLOR_GRAY2BGR)
    
    # roi
    cv.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
    cv.rectangle(gradImg, (x, y), (x+w, y+h), (0,255,0), 2)
    cv.rectangle(gradImgThresh, (x, y), (x+w, y+h), (0,255,0), 2)
    cv.rectangle(localMaxImg, (x, y), (x+w, y+h), (0,255,0), 2)
    cv.rectangle(localMaxImgThresh, (x, y), (x+w, y+h), (0,255,0), 2)

    gradImgROI = gradImgThresh[y:y+h, x:x+w]
    localMaxImgROI = localMaxImgThresh[y:y+h, x:x+w]
    imgROI = img[y:y+h, x:x+w]

    # crop images
    gray = gray[yCrop:yCrop+hCrop, xCrop:xCrop+wCrop]
    gradImg = gradImg[yCrop:yCrop+hCrop, xCrop:xCrop+wCrop]
    gradImgThresh = gradImgThresh[yCrop:yCrop+hCrop, xCrop:xCrop+wCrop]
    localMaxImg = localMaxImg[yCrop:yCrop+hCrop, xCrop:xCrop+wCrop]
    localMaxImgThresh = localMaxImgThresh[yCrop:yCrop+hCrop, xCrop:xCrop+wCrop]
    img = img[yCrop:yCrop+hCrop, xCrop:xCrop+wCrop]

    # paint colored image with grad info
    imgPaintedGrad = imgROI.copy()
    indices = np.where(gradImgROI > gradThresh)
    gradImgTemp = gradImgROI.copy()
    gradImgTemp[:, :, :2] = 0
    gradImgTemp[:, :, 2] = 255
    imgPaintedGrad[indices] = gradImgTemp[indices]

    # paint colored image with local max info
    imgPaintedLocalMax = imgROI.copy()
    indices = np.where(localMaxImgROI > localMaxThresh)
    localMaxImgTemp = localMaxImgROI.copy()
    localMaxImgTemp[:, :, 1:3] = 0
    localMaxImgTemp[:, :, 0] = 255
    imgPaintedLocalMax[indices] = localMaxImgTemp[indices]

    #if realImg:
    #    resolution = (854, 480)
    #else:
    resolution = (img.shape[1], img.shape[0])

    print("Grad thresh", gradThresh)
    print("Local thresh", localMaxThresh)

    # show full image
    cv.imshow("gradient", cv.resize(gradImg, resolution))
    cv.imshow("local max", cv.resize(localMaxImg, resolution, cv.INTER_LANCZOS4))
    cv.imshow("img", cv.resize(img, resolution))

    # show full thresholded image
    cv.imshow("gradient thresh", cv.resize(gradImgThresh, resolution))
    cv.imshow("local max thresh", cv.resize(localMaxImgThresh, resolution, cv.INTER_LANCZOS4))

    # show rois
    cv.imshow("gradient roi", gradImgROI)
    cv.imshow("local max roi", localMaxImgROI)
    cv.imshow("img roi grad", imgPaintedGrad)
    cv.imshow("img roi local max", imgPaintedLocalMax)
    

    #cv.imshow("gray", cv.resize(gray, resolution))

    #cv.imshow("roi", cv.resize(roiMask, resolution))
    
    cv.waitKey(0)
