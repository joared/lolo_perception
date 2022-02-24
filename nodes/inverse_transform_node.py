#!/usr/bin/env python
import rospy
import tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for python 2.7
from scipy.spatial.transform import Rotation as R

from lolo_simulation.coordinate_system import CoordinateSystemArtist

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
        plt.figure(self.fig.number)
        titles = ["X", "Y", "Z"]
        colors = ["r", "g", "b"]
        subPlotRows = 6
        subPlotCols = 2

        # remove the first smaples used when averaging errors
        poses1 = np.array(self.poses1[self.movingErrAverage-1:])
        if len(poses1) < self.movingErrAverage:
            return
        poses2 = np.array(self.poses2[self.movingErrAverage-1:])

        diffs = np.array(self.diffs)

        # use degrees
        poses1[:, 3:] = np.rad2deg(poses1[:, 3:])
        poses2[:, 3:] = np.rad2deg(poses2[:, 3:])
        diffs[:, 3:] = np.rad2deg(diffs[:, 3:])

        if self.movingErrAverage > 1:
            n = self.movingErrAverage
            newDiffs = np.zeros(poses1.shape)
            for i in range(6):
                d = np.convolve(diffs[:, i], np.ones(n)/n, mode="valid")
                newDiffs[:, i] = d
            diffs = newDiffs

        marginP = 0.1
        translationDiffs = diffs[:, :3] #[d[:3] for d in self.diffs]
        yAbsMaxTransl = max(np.max(translationDiffs), np.min(translationDiffs), key=abs)
        yAbsMaxTransl += yAbsMaxTransl*marginP
        
        rotDiffs = diffs[:, 3:] #[d[3:] for d in self.diffs]
        yAbsMaxRot = max(np.max(rotDiffs), np.min(rotDiffs), key=abs)
        yAbsMaxRot += yAbsMaxRot*marginP

        # plot translation
        for i, (c, title) in enumerate(zip(colors, titles)):
            plt.subplot(subPlotRows, subPlotCols, 2*i+1)
            plt.cla()
            plt.gca().set_ylabel(title)

            plt.plot([p[i] for p in poses1], color=c)
            plt.plot([p[i] for p in poses2], "-o", markersize=2, color=c)

            plt.subplot(subPlotRows, subPlotCols, 2*i+2)
            plt.cla()
            if i == 0:
                plt.title("Average error, {} samples".format(self.movingErrAverage))
            
            
            # center error around 0
            plt.gca().set_ylim(ymin=-yAbsMaxTransl, ymax=yAbsMaxTransl)

            # zeros reference line
            plt.plot(np.zeros(len(diffs)), "--", color="gray")

            # average error
            avgDiff = np.average(diffs[:, i])
            plt.plot(np.ones(len(diffs))*avgDiff, "--", color=c)

            plt.gca().yaxis.set_label_position("right")
            plt.gca().yaxis.tick_right()
            plt.gca().set_ylabel("Avg: {}".format(round(avgDiff, 2)))

            plt.plot(diffs[:, i], color=c)

        # plot orientation
        titles = ["Roll", "Pitch", "Yaw"]
        colors = ["r", "g", "b"]
        for i, (c, title) in enumerate(zip(colors, titles)):
            plt.subplot(subPlotRows, subPlotCols, 2*i+1 + subPlotRows)
            plt.cla()
            plt.gca().set_ylabel(title)

            # plot in degrees
            plt.plot([p[3+i] for p in poses1], color=c)
            plt.plot([p[3+i] for p in poses2], "-o", markersize=2, color=c)

            plt.subplot(subPlotRows, subPlotCols, 2*i+2 + subPlotRows)
            plt.cla()

            # center error around 0
            plt.gca().set_ylim(ymin=-yAbsMaxRot, ymax=yAbsMaxRot)

            # zeros reference line
            plt.plot(np.zeros(len(diffs)), "--", color="gray")

            # average error
            avgDiff = np.average(diffs[:, 3+i])
            plt.plot(np.ones(len(diffs))*avgDiff, "--", color=c)

            plt.gca().yaxis.set_label_position("right")
            plt.gca().yaxis.tick_right()
            plt.gca().set_ylabel("Avg: {}".format(round(avgDiff, 2)))

            plt.plot(diffs[:, 3+i], color=c)

    def plot3D(self):
        if not self.poses1 or not self.poses2:
            return

        poses1 = np.array(self.poses1)
        poses2 = np.array(self.poses2)

        maxX = max(np.max(poses1[:, 0]), np.max(poses2[:, 0]))
        minX = min(np.min(poses1[:, 0]), np.min(poses2[:, 0]))
        absMaxX = max(abs(maxX), abs(minX))

        maxY = max(np.max(poses1[:, 1]), np.max(poses2[:, 1]))
        minY = min(np.min(poses1[:, 1]), np.min(poses2[:, 1]))
        absMaxY = max(abs(maxY), abs(minY))

        absMaxXY = max(absMaxX, absMaxY)

        maxZ = max(np.max(poses1[:, 2]), np.max(poses2[:, 2]))
        minZ = min(np.min(poses1[:, 2]), np.min(poses2[:, 2]))

        absMaxXYZ = max(absMaxXY, maxZ/2)
        self.ax.cla()

        self.ax.set_xlim(-absMaxXYZ*2, 0)
        self.ax.set_ylim(-absMaxXYZ, absMaxXYZ)
        self.ax.set_zlim(-absMaxXYZ, absMaxXYZ)
        
        self.ax.plot(-poses2[:, 2], poses2[:, 0], -poses2[:, 1], "o", markersize=1.6, color="r")
        self.ax.plot(-poses1[:, 2], poses1[:, 0], -poses1[:, 1], linewidth=1.6, color="g")

        # plot reference line so that it is possible to determine what measurement is
        # associated with what ground truth
        for p1, p2 in zip(poses1, poses2):
            self.ax.plot([-p1[2], -p2[2]], [p1[0], p2[0]], [-p1[1], -p2[1]], "-", linewidth=0.3, color="lightcoral")

        csCamera = CoordinateSystemArtist(scale=.4)
        csCamera.cs.rotation = R.from_euler("XYZ", (-np.pi/2, -np.pi/2, 0)).as_dcm()
        csCamera.draw(self.ax)

        csLastPose = CoordinateSystemArtist(scale=.4)
        lastTransl = self.poses2[-1][:3]
        csLastPose.cs.translation = np.matmul(csCamera.cs.rotation, lastTransl) #np.array([-lastTransl[2], lastTransl[0], -lastTransl[1]])
        az, ax, ay = self.poses2[-1][3:]
        csLastPose.cs.rotation = R.from_euler("YXZ", (ay, ax, az)).as_dcm()
        csLastPose.cs.rotation = np.matmul(csCamera.cs.rotation, csLastPose.cs.rotation)
        csLastPose.draw(self.ax)

        """
        self.ax.plot(poses2[:, 0], poses2[:, 1], poses2[:, 2], "-o", markersize=1.6, color="r")
        self.ax.plot(poses1[:, 0], poses1[:, 1], poses1[:, 2], linewidth=1, color="g")

        csCamera = CoordinateSystemArtist()
        csCamera.draw(self.ax)

        csLastPose = CoordinateSystemArtist(scale=.4)
        csLastPose.cs.translation = self.poses2[-1][:3]
        ax, ay, az = self.poses2[-1][3:]
        csLastPose.cs.rotation = R.from_euler("YXZ", (ay, ax, az)).as_dcm()
        csLastPose.draw(self.ax)
        """

    def transToPose(self, transl, rot):
        ay, ax, az = R.from_quat(rot).as_euler("YXZ")
        pose = list(transl) + [az, ax, ay] # [x, y, z, roll, pitch yaw]
        return np.array(pose)

    def run(self):
        hz = 10.0
        rate = rospy.Rate(hz)
        #rospy.Timer(rospy.Duration(1.0/2.0), lambda event: self.plot3D())
        self.fig = plt.figure()
        self.fig3D = plt.figure()
        self.fig3D.add_subplot(projection="3d")
        self.ax = Axes3D(self.fig3D)

        plotRate = 10.0
        plotIdx = int(hz / plotRate)

        i = 0
        lastTime = 0
        while not rospy.is_shutdown():
            i += 1
            timeStamp = rospy.Time(0)
            pose1 = None
            pose2 = None
            try:
                t = self.listener.getLatestCommonTime(self.referenceFrame, self.trans1Name)
            except:
                rospy.loginfo_throttle(2, "Waiting for first transformation")
                continue

            if t != lastTime:
                lastTime = t
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

                    self.poses1 = self.poses1[-self.bufferLength-self.movingErrAverage+1:]
                    self.poses2 = self.poses2[-self.bufferLength-self.movingErrAverage+1:]
                    self.diffs = self.diffs[-self.bufferLength-self.movingErrAverage+1:]
            else:
                rospy.loginfo_throttle(2, "Found the same transformation 1, probably done here?")

            if i % plotIdx == 0:
                self.plotPose2()
                self.plot3D()
            plt.pause(0.000001)
            
            
            rate.sleep()

        

if __name__ == '__main__':
    rospy.init_node("inverse_transform_node")
    invTransNode = InverseTransformNode("docking_station/feature_model_link", 
                                        "docking_station/feature_model_estimated_link", 
                                        referenceFrame="lolo_camera_link", 
                                        bufferLength=500,
                                        movingErrAverage=1)

    invTransNode.run()
