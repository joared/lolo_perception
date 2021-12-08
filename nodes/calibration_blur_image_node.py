#!/usr/bin/env python
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image
import time

class BlurImageNode:
    def __init__(self, kernelSize):
        self.kernelSize = kernelSize
        self.imageMsg = None
        self.imgSubsciber = rospy.Subscriber('usb_cam/image_raw', Image, self.imgCallback)
        self.imgPublisher = rospy.Publisher('usb_cam/image_raw_blurred', Image, queue_size=1)
        self.bridge = CvBridge()

    def imgCallback(self, msg):
        self.imageMsg = msg

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.imageMsg:
                tStart = time.time()
                try:
                    imgColor = self.bridge.imgmsg_to_cv2(self.imageMsg, 'bgr8')
                except CvBridgeError as e:
                    print(e)
                else:
                    kernel = np.ones((self.kernelSize, self.kernelSize),np.float32)/(self.kernelSize*self.kernelSize)
                    blurred = cv.filter2D(imgColor,-1,kernel)
                    self.imgPublisher.publish(self.bridge.cv2_to_imgmsg(blurred, "bgr8"))
                    
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('blur_image_node')
    blurImgNode = BlurImageNode(kernelSize=7)
    blurImgNode.run()