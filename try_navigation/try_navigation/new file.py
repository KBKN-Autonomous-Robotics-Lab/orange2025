import rclpy
from rclpy.node import Node
# rclpy (ROS 2のpythonクライアント)の機能を使えるようにします。
import rclpy
# rclpy (ROS 2のpythonクライアント)の機能のうちNodeを簡単に使えるようにします。こう書いていない場合、Nodeではなくrclpy.node.Nodeと書く必要があります。
from rclpy.node import Node
# ROS 2の文字列型を使えるようにimport
import std_msgs.msg as std_msgs
import sensor_msgs.msg as sensor_msgs
import nav_msgs.msg as nav_msgs
from livox_ros_driver2.msg import CustomMsg
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import pandas as pd
# import open3d as o3d
from std_msgs.msg import Int8MultiArray
from nav_msgs.msg import OccupancyGrid
import cv2
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
import yaml
import os
import time
import matplotlib.pyplot
import struct
import geometry_msgs.msg as geometry_msgs
from collections import OrderedDict

from scipy import interpolate
from std_msgs.msg import Float32MultiArray
import subprocess

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header


class PcdLineDetector(Node):
    def __init__(self):
        super().__init__('pcd_line_detector')

        qos_profile = rclpy.qos.QoSProfile(
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self.subscription = self.create_subscription(
            PointCloud2,
            '/pcd_segment_ground',
            self.pcd_callback,
            qos_profile
        )

    def pcd_callback(self, msg):
        points = self.pointcloud2_to_array(msg)
        xy_points = points[:2, :].T  # shape = (N, 2)

        lines = self.hough_transform(xy_points, angle_res=1, rho_res=0.1, threshold=50)
        self.get_logger().info(f"検出された直線数: {len(lines)}")
        for i, (rho, theta) in enumerate(lines[:5]):
            self.get_logger().info(f"Line {i+1}: rho={rho:.2f}, theta={theta:.2f}°")

    def pointcloud2_to_array(self, cloud_msg):
        points = np.frombuffer(cloud_msg.data, dtype=np.uint8).reshape(-1, cloud_msg.point_step)
        x = np.frombuffer(points[:, 0:4].tobytes(), dtype=np.float32)
        y = np.frombuffer(points[:, 4:8].tobytes(), dtype=np.float32)
        z = np.frombuffer(points[:, 8:12].tobytes(), dtype=np.float32)
        intensity = np.frombuffer(points[:, 12:16].tobytes(), dtype=np.float32)
        return np.vstack((x, y, z, intensity))

    def hough_transform(self, points, angle_res=1.0, rho_res=0.1, threshold=50):
        # θ を degree 単位で 0°〜180°までスキャン
        thetas = np.deg2rad(np.arange(0, 180, angle_res))
        accumulator = defaultdict(int)
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)

        for x, y in points:
            for i, (c, s) in enumerate(zip(cos_t, sin_t)):
                rho = x * c + y * s
                rho_quant = round(rho / rho_res) * rho_res
                accumulator[(rho_quant, i)] += 1

        # 閾値以上の票が入った (ρ, θ) を抽出
        lines = []
        for (rho, theta_idx), count in accumulator.items():
            if count >= threshold:
                theta_deg = theta_idx * angle_res
                lines.append((rho, theta_deg))

        return lines

def main(args=None):
    rclpy.init(args=args)
    node = PcdLineDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

