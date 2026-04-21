#!/usr/bin/env python3
"""
intensity_filter_node.py
反射強度範囲フィルタノード
/pcd_segment_ground をサブスクライブし、
intensity が [intensity_min, intensity_max] の範囲内の点のみを
/intensity_filtered としてパブリッシュする。
rqt_reconfigure でリアルタイムに閾値を変更可能。
Usage:
    ros2 run try_navigation intensity_filter_node
    rqt  # → Plugins → Configuration → Dynamic Reconfigure
"""
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import sensor_msgs.msg as sensor_msgs
from std_msgs.msg import Header


# ============================================================
#  ユーティリティ（white_line_detection.py から流用）
# ============================================================
def pointcloud2_to_array(cloud_msg):
    """sensor_msgs/PointCloud2 → numpy配列 [4, N] (x, y, z, intensity)"""
    points = np.frombuffer(cloud_msg.data, dtype=np.uint8).reshape(
        -1, cloud_msg.point_step
    )
    x         = np.frombuffer(points[:, 0:4 ].tobytes(), dtype=np.float32)
    y         = np.frombuffer(points[:, 4:8 ].tobytes(), dtype=np.float32)
    z         = np.frombuffer(points[:, 8:12].tobytes(), dtype=np.float32)
    intensity = np.frombuffer(points[:, 12:16].tobytes(), dtype=np.float32)
    return np.vstack((x, y, z, intensity))   # shape: [4, N]


def point_cloud_intensity_msg(points, t_stamp, parent_frame):
    """numpy配列 [N, 4] → sensor_msgs/PointCloud2"""
    ros_dtype = sensor_msgs.PointField.FLOAT32
    dtype     = np.float32
    itemsize  = np.dtype(dtype).itemsize

    fields = [
        sensor_msgs.PointField(name='x',         offset=0,  datatype=ros_dtype, count=1),
        sensor_msgs.PointField(name='y',         offset=4,  datatype=ros_dtype, count=1),
        sensor_msgs.PointField(name='z',         offset=8,  datatype=ros_dtype, count=1),
        sensor_msgs.PointField(name='intensity', offset=12, datatype=ros_dtype, count=1),
    ]

    header = Header(frame_id=parent_frame, stamp=t_stamp)
    return sensor_msgs.PointCloud2(
        header       = header,
        height       = 1,
        width        = points.shape[0],
        is_dense     = True,
        is_bigendian = False,
        fields       = fields,
        point_step   = itemsize * 4,
        row_step     = itemsize * 4 * points.shape[0],
        data         = points.astype(dtype).tobytes(),
    )


# ============================================================
#  メインノード
# ============================================================
class IntensityFilterNode(Node):
    def __init__(self):
        super().__init__('intensity_filter')

        # ---------- パラメータ宣言 ----------
        # rqt_reconfigure / ros2 param set でリアルタイム変更可能
        self.declare_parameter('intensity_min', 0.0)    # 下限（0〜255）
        self.declare_parameter('intensity_max', 255.0)  # 上限（0〜255）

        # ---------- Subscriber ----------
        self.ground_sub = self.create_subscription(
            sensor_msgs.PointCloud2,
            '/pcd_segment_ground',
            self.ground_callback,
            qos_profile_sensor_data,
        )

        # ---------- Publisher ----------
        self.filtered_pub = self.create_publisher(
            sensor_msgs.PointCloud2,
            '/intensity_filtered',
            10,
        )

        self.get_logger().info('intensity_filter node started')
        self.get_logger().info('  subscribe: /pcd_segment_ground')
        self.get_logger().info('  publish  : /intensity_filtered')
        self.get_logger().info('  params   : intensity_min, intensity_max')
        self.get_logger().info('  GUI      : rqt → Plugins → Configuration → Dynamic Reconfigure')

    # ----------------------------------------------------------
    #  コールバック
    # ----------------------------------------------------------
    def ground_callback(self, msg):
        # パラメータを毎コールバックで取得（rqt_reconfigure の変更が即反映される）
        i_min = self.get_parameter('intensity_min').get_parameter_value().double_value
        i_max = self.get_parameter('intensity_max').get_parameter_value().double_value

        # PointCloud2 → numpy [4, N]
        pcd = pointcloud2_to_array(msg)     # [4, N]
        if pcd.shape[1] == 0:
            return

        # 強度フィルタ
        intensity = pcd[3, :]               # [N]
        mask = (intensity >= i_min) & (intensity <= i_max)
        filtered = pcd[:, mask]             # [4, M]

        # パブリッシュ（点がゼロでも空メッセージとして送る）
        out_msg = point_cloud_intensity_msg(
            filtered.T,                     # [M, 4]
            msg.header.stamp,
            msg.header.frame_id,
        )
        self.filtered_pub.publish(out_msg)

        self.get_logger().debug(
            f'intensity [{i_min:.0f}, {i_max:.0f}]  '
            f'total={pcd.shape[1]}  filtered={filtered.shape[1]}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = IntensityFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()