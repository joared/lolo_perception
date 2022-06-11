#!/usr/bin/env python

import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
class PathPublisher:
    def __init__(self, poseTopic="/pose_topic", pathTopic="/pose/path", markerTopic="/pose/ellipses"):
        rospy.Subscriber(poseTopic, PoseWithCovarianceStamped, self._poseCallback)
        self.path = Path()
        self.markerArray = MarkerArray()
        self.pathPub = rospy.Publisher(pathTopic, Path, queue_size=10)
        self.markerPub = rospy.Publisher(markerTopic, MarkerArray, queue_size=10)

    def _poseCallback(self, msg):
        p = PoseStamped()
        p.header = msg.header
        p.pose = msg.pose.pose
        self.path.header = msg.header
        marker = Marker()
        marker.header = msg.header
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = .5
        marker.color.g = 0.0
        marker.color.b = .5
        marker.color.a = .1
        marker.pose = msg.pose.pose 
        self.markerArray.markers.append(marker)
        self.path.poses.append(p)
        
        id = 0
        for m in self.markerArray.markers:
            m.id = id
            id += 1

    def publish(self):
        self.pathPub.publish(self.path)
        self.markerPub.publish(self.markerArray)

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