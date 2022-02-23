#!/usr/bin/env python
import rospy
import tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

class InverseTransformNode:
    def __init__(self, trans1Name, trans2Name, referenceFrame=None, bufferLength=500):
        self.trans1Name = trans1Name
        self.trans2Name = trans2Name
        self.referenceFrame = referenceFrame if referenceFrame else trans1Name

        self.bufferLength = bufferLength
        
        self.poses1 = [] # [[x, y, z, ax, ay, az], ...] # euler angles with YXZ rotation order
        self.poses2 = [] # [[x, y, z, ax, ay, az], ...] # euler angles with YXZ rotation order
        self.diffs = [] # # [[diffx, diffy, diffz, diffax, diffay, diffaz], ...] # euler angles with YXZ rotation order

        self.listener = tf.TransformListener()

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
                
                for i, c in enumerate(["r", "g", "b"]):
                    plt.subplot(3, 2, 2*i+1)
                    plt.cla()
                    plt.plot([p[i] for p in self.poses1], color=c)
                    plt.plot([p[i] for p in self.poses2], "-o", color=c)
                    plt.subplot(3, 2, 2*i+2)
                    plt.cla()
                    plt.plot([d[i] for d in self.diffs], color=c)

                plt.pause(0.001)
            rate.sleep()

if __name__ == '__main__':
    rospy.init_node("inverse_transform_node")
    invTransNode = InverseTransformNode("docking_station/feature_model_link", 
                                        "docking_station/feature_model_estimated_link", 
                                        referenceFrame="lolo_camera_link", 
                                        bufferLength=10)

    invTransNode.run()
