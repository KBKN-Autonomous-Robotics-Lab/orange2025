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
            depth=10  # QoS設定を少し深くする
        )

        qos_profile_pub = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10  # パブリッシュ側も同様に深く設定
        )

        # message_filtersによるSubscriberを使用
        self.sub1 = Subscriber(self, PointCloud2, '/pcd_segment_obs', qos_profile=qos_profile_sub)
        self.sub2 = Subscriber(self, PointCloud2, '/pothole_points', qos_profile=qos_profile_sub)

        self.ts = ApproximateTimeSynchronizer([self.sub1, self.sub2], queue_size=1, slop=0.1)
        self.ts.registerCallback(self.callback)

        self.pub = self.create_publisher(PointCloud2, '/merged_cloud', qos_profile_pub)

    def callback(self, cloud1, cloud2):
        # cloud1 (通常のセグメント化された点群)
        points1 = list(point_cloud2.read_points(cloud1, field_names=("x", "y", "z"), skip_nans=True))
        # cloud2 (ポットホールの点群)
        points2 = list(point_cloud2.read_points(cloud2, field_names=("x", "y", "z"), skip_nans=True))

        if not points2:
            # points2が空の場合はpoints1の点群だけを出力
            self.get_logger().info("No pothole points received, publishing only cloud1.")
            merged_points = points1
        else:
            # points2が存在する場合はpoints1とpoints2をマージ
            self.get_logger().info("Merging cloud1 and cloud2.")
            merged_points = points1 + points2

        # 点群が空でないか確認
        if not merged_points:
            self.get_logger().warn("Merged points are empty. Skipping publication.")
            return  # マージ後の点群が空の場合はパブリッシュしない

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = cloud1.header.frame_id  # ここでフレームIDを設定

        merged_cloud_msg = point_cloud2.create_cloud_xyz32(header, merged_points)
        self.pub.publish(merged_cloud_msg)
        self.get_logger().info("Published merged point cloud.")


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudMerger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

