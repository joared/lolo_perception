#!/usr/bin/env python

import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

class StaticPathPublisher:
    def __init__(self, pathTopic, frameID, positions):
        self.pathPub = rospy.Publisher(pathTopic, Path, queue_size=10)
        self.markerPub = rospy.Publisher("/some_markers", MarkerArray, queue_size=10)

        self.path = Path()
        self.path.header.frame_id = frameID
        self.path.header.stamp = rospy.Time.now()
        self.markerArray = MarkerArray()
        
        for id, pos in enumerate(positions):
            p = PoseStamped()
            p.header = self.path.header
            p.pose.position.x = pos[0]
            p.pose.position.y = pos[1]
            p.pose.position.z = pos[2]
            p.pose.orientation.w = 1
            self.path.poses.append(p)

            marker = Marker()
            marker.header = self.path.header
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = .5
            marker.color.g = 0.0
            marker.color.b = .5
            marker.color.a = .8
            marker.pose = p.pose
            marker.id = id
            self.markerArray.markers.append(marker)

    def run(self):
        rate = rospy.Rate(1)

        while not rospy.is_shutdown():
            self.pathPub.publish(self.path)
            self.markerPub.publish(self.markerArray)
            rate.sleep()



if __name__ == "__main__":
    rospy.init_node("static_path_publisher_node")

    positions = [(0,0,-8), (0,0,-6) ,(0,0,-4), (0,0,-2)]
    pathPub = StaticPathPublisher("/some_path", "docking_station_link", positions)
    pathPub.run()