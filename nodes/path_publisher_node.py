#!/usr/bin/env python

import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped

class PathPublisher:
    def __init__(self, poseTopic="/pose_topic", pathTopic="/pose/path"):
        print("POSEPSOEPSOPOSEPOSEPO")
        rospy.Subscriber(poseTopic, PoseWithCovarianceStamped, self._poseCallback)
        self.path = Path()
        self.pub = rospy.Publisher(pathTopic, Path, queue_size=10)

    def _poseCallback(self, msg):
        p = PoseStamped()
        p.header = msg.header
        p.pose = msg.pose.pose
        self.path.header = msg.header
        self.path.poses.append(p)
        print("ASLSALASLLASLHSALSH")

    def publish(self):
        self.pub.publish(self.path)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if len(self.path.poses) > 0:
                self.publish()
            rate.sleep()

if __name__ == "__main__":
    rospy.init_node("path_publisher_node")
    pathPub = PathPublisher()
    pathPub.run()