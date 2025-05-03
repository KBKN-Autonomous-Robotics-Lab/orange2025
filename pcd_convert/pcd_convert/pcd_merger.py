import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from message_filters import Subscriber, ApproximateTimeSynchronizer


class PointCloudMerger(Node):
    def __init__(self):
        super().__init__('pointcloud_merger')

        qos_profile_sub = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        qos_profile_pub = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # message_filtersによるSubscriberを使用
        self.sub1 = Subscriber(self, PointCloud2, '/pcd_segment_obs', qos_profile=qos_profile_sub)
        self.sub2 = Subscriber(self, PointCloud2, '/pothole_points', qos_profile=qos_profile_sub)

        self.ts = ApproximateTimeSynchronizer([self.sub1, self.sub2], queue_size=1, slop=0.1)
        self.ts.registerCallback(self.callback)

        self.pub = self.create_publisher(PointCloud2, '/merged_cloud', qos_profile_pub)

    def callback(self, cloud1, cloud2):
        points1 = list(point_cloud2.read_points(cloud1, field_names=("x", "y", "z"), skip_nans=True))
        points2 = list(point_cloud2.read_points(cloud2, field_names=("x", "y", "z"), skip_nans=True))

        merged_points = points1 + points2

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = cloud1.header.frame_id  # または共通のframe_idを指定

        merged_cloud_msg = point_cloud2.create_cloud_xyz32(header, merged_points)
        self.pub.publish(merged_cloud_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudMerger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

