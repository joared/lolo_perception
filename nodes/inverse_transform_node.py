#!/usr/bin/env python
import rospy
import tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

class InverseTransformNode:
    def __init__(self, trans1Name, trans2Name, referenceFrame=None, bufferLength=500, movingErrAverage=1):
        self.trans1Name = trans1Name
        self.trans2Name = trans2Name
        self.referenceFrame = referenceFrame if referenceFrame else trans1Name

        self.movingErrAverage = movingErrAverage

        self.bufferLength = bufferLength
        
        self.poses1 = [] # [[x, y, z, ax, ay, az], ...] # euler angles with YXZ rotation order
        self.poses2 = [] # [[x, y, z, ax, ay, az], ...] # euler angles with YXZ rotation order
        self.diffs = [] # # [[diffx, diffy, diffz, diffax, diffay, diffaz], ...] # euler angles with YXZ rotation order

        self.listener = tf.TransformListener()

    def plotTranslation(self, subPlotRows, subPlotCols, startIdx):
        titles = ["X", "Y", "Z"]
        colors = ["r", "g", "b"]
        for i, (c, title) in enumerate(zip(colors, titles)):
            plt.subplot(subPlotRows, subPlotCols, 2*i+1 + startIdx)
            plt.cla()
            plt.gca().set_ylabel(title)

            plt.plot([p[i] for p in self.poses1], color=c)
            plt.plot([p[i] for p in self.poses2], "-o", color=c)

            plt.subplot(subPlotRows, subPlotCols, 2*i+2 + startIdx)
            plt.cla()
            if i == 0:
                plt.title("Average error, {} samples".format(self.movingErrAverage))
            
            diffs = [d[i] for d in self.diffs]
            if self.movingErrAverage > 1:
                n = self.movingErrAverage
                diffs = list(np.convolve(diffs, np.ones(n)/n, mode="same"))

            # center error around 0
            yAbsMax = max(diffs, key=abs)
            marginP = 0.1
            yAbsMax += yAbsMax*marginP
            plt.gca().set_ylim(ymin=-yAbsMax, ymax=yAbsMax)

            # zeros reference line
            plt.plot(np.zeros(len(diffs)), "--", color="gray")
            plt.plot(diffs, color=c)

    def plotOrientation(self, subPlotRows, subPlotCols, startIdx):
        titles = ["Roll", "Pitch", "Yaw"]
        colors = ["r", "g", "b"]
        for i, (c, title) in enumerate(zip(colors, titles)):
            plt.subplot(subPlotRows, subPlotCols, 2*i+1 + startIdx)
            plt.cla()
            plt.gca().set_ylabel(title)

            plt.plot([p[3+i] for p in self.poses1], color=c)
            plt.plot([p[3+i] for p in self.poses2], "-o", color=c)

            plt.subplot(subPlotRows, subPlotCols, 2*i+2 + startIdx)
            plt.cla()
            if i == 0:
                plt.title("Average error, {} samples".format(self.movingErrAverage))
            
            diffs = [d[3+i] for d in self.diffs]
            if self.movingErrAverage > 1:
                n = self.movingErrAverage
                diffs = list(np.convolve(diffs, np.ones(n)/n, mode="same"))

            # center error around 0
            yAbsMax = max(diffs, key=abs)
            marginP = 0.1
            yAbsMax += yAbsMax*marginP
            plt.gca().set_ylim(ymin=-yAbsMax, ymax=yAbsMax)

            # zeros reference line
            plt.plot(np.zeros(len(diffs)), "--", color="gray")
            plt.plot(diffs, color=c)

    def plotPose2(self):
        titles = ["X", "Y", "Z"]
        colors = ["r", "g", "b"]
        subPlotRows = 6
        subPlotCols = 2
        for i, (c, title) in enumerate(zip(colors, titles)):
            plt.subplot(subPlotRows, subPlotCols, 2*i+1)
            plt.cla()
            plt.gca().set_ylabel(title)

            plt.plot([p[i] for p in self.poses1], color=c)
            plt.plot([p[i] for p in self.poses2], "-o", color=c)

            plt.subplot(subPlotRows, subPlotCols, 2*i+2)
            plt.cla()
            if i == 0:
                plt.title("Average error, {} samples".format(self.movingErrAverage))
            
            diffs = [d[i] for d in self.diffs]
            if self.movingErrAverage > 1:
                n = self.movingErrAverage
                diffs = list(np.convolve(diffs, np.ones(n)/n, mode="same"))

            # center error around 0
            yAbsMax = max(diffs, key=abs)
            marginP = 0.1
            yAbsMax += yAbsMax*marginP
            plt.gca().set_ylim(ymin=-yAbsMax, ymax=yAbsMax)

            # zeros reference line
            plt.plot(np.zeros(len(diffs)), "--", color="gray")
            plt.plot(diffs, color=c)

        # plot orientation
        titles = ["Roll", "Pitch", "Yaw"]
        colors = ["r", "g", "b"]
        for i, (c, title) in enumerate(zip(colors, titles)):
            plt.subplot(subPlotRows, subPlotCols, 2*i+1 + subPlotRows)
            plt.cla()
            plt.gca().set_ylabel(title)

            plt.plot([p[3+i] for p in self.poses1], color=c)
            plt.plot([p[3+i] for p in self.poses2], "-o", color=c)

            plt.subplot(subPlotRows, subPlotCols, 2*i+2 + subPlotRows)
            plt.cla()
            
            diffs = [d[3+i] for d in self.diffs]
            if self.movingErrAverage > 1:
                n = self.movingErrAverage
                diffs = list(np.convolve(diffs, np.ones(n)/n, mode="same"))

            # center error around 0
            yAbsMax = max(diffs, key=abs)
            marginP = 0.1
            yAbsMax += yAbsMax*marginP
            plt.gca().set_ylim(ymin=-yAbsMax, ymax=yAbsMax)

            # zeros reference line
            plt.plot(np.zeros(len(diffs)), "--", color="gray")
            plt.plot(diffs, color=c)

    def plotPose(self):
        colors = ["r", "g", "b"]

        # plot translation
        plt.subplot(2, 2, 1)
        plt.cla()
        plt.gca().set_ylabel("Translation X, Y, Z")
        for i, c in enumerate(colors):
            plt.plot([p[i] for p in self.poses1], color=c)
            plt.plot([p[i] for p in self.poses2], "-o", color=c)
            
        # plot orientation
        plt.subplot(2, 2, 3)
        plt.cla()
        plt.gca().set_ylabel("Orientation X, Y, Z")
        for i, c in enumerate(colors):
            plt.plot([p[3+i] for p in self.poses1], color=c)
            plt.plot([p[3+i] for p in self.poses2], "-o", color=c)


    def transToPose(self, transl, rot):
        euler = R.from_quat(rot).as_euler("YXZ")
        pose = list(transl) + [euler[2], euler[1], euler[0]] # [x, y, z, roll, pitch yaw]
        return np.array(pose)

    def run(self):
        rate = rospy.Rate(100)
        plt.figure()
        while not rospy.is_shutdown():
            timeStamp = rospy.Time(0)
            pose1 = None
            pose2 = None
            try:
                transl1, rot1 = self.listener.lookupTransform(self.referenceFrame, 
                                                              self.trans1Name, 
                                                              timeStamp)
            except:
                rospy.loginfo_throttle(2, "Transform 1 not found")
            else:
                pose1 = self.transToPose(transl1, rot1)

            try:
                transl2, rot2 = self.listener.lookupTransform(self.referenceFrame, 
                                                              self.trans2Name, 
                                                              timeStamp)
            except:
                rospy.loginfo_throttle(2, "Transform 2 not found")
            else:
                pose2 = self.transToPose(transl2, rot2)

            if not self.poses1 and (pose1 is None or pose2 is None):
                rospy.loginfo_throttle(2, "Waiting for first transformation")
            else:
                if pose1 is None:
                    pose1 = np.array([0.]*6)
                if pose2 is None:
                    pose2 = np.array([0.]*6)

                self.poses1.append(pose1)
                self.poses2.append(pose2)
                self.diffs.append(pose1-pose2)

                self.poses1 = self.poses1[-self.bufferLength:]
                self.poses2 = self.poses2[-self.bufferLength:]
                self.diffs = self.diffs[-self.bufferLength:]
         
                self.plotPose2()
                plt.pause(0.001)

            rate.sleep()

        

if __name__ == '__main__':
    rospy.init_node("inverse_transform_node")
    invTransNode = InverseTransformNode("docking_station/feature_model_link", 
                                        "docking_station/feature_model_estimated_link", 
                                        referenceFrame="lolo_camera_link", 
                                        bufferLength=50,
                                        movingErrAverage=5)

    invTransNode.run()
