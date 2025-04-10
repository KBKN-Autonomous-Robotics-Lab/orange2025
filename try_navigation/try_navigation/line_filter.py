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


class linefilter(Node):
    def __init__(self):
        super().__init__('line_filter_node')

        # 入力PGMファイルのパス（絶対パス推奨）
        self.input_path = '/home/ubuntu/ros2_ws/src/kbkn_maps/maps/tsukuba/whiteline/whiteline.pgm'
        self.output_dir = '/home/ubuntu/ros2_ws/src/kbkn_maps/maps/tsukuba/whiteline'

        os.makedirs(self.output_dir, exist_ok=True)

        self.get_logger().info("Line Filter Node started.")
        #self.check_pgm_load()
        self.process_pgm()
        
    def process_pgm(self):
        try:
            image = cv2.imread(self.input_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                self.get_logger().error(f"Failed to load PGM file: {self.input_path}")
                return
            self.log_image_size(image)
            
            pgm_filename = self.save_image(image, 'output.pgm')
            binary_image = self.binarize_image(image)
            binary_filename = self.save_image(binary_image, 'binary_output.pgm')
            edge_image = self.detect_edges(binary_image)
            edge_filename = self.save_image(edge_image, 'edge_output.pgm')
            hough_image = self.detect_lines(edge_image)
            hough_filename = self.save_image(hough_image, 'hough_output.pgm')
            self.log_image_size(hough_image)

            self.get_logger().info(f"PGM処理完了: {pgm_filename}, {binary_filename}, {edge_filename}, {hough_filename}")
        except Exception as e:
            self.get_logger().error(f"画像処理中にエラーが発生しました: {e}")    

    def save_image(self, image, filename):
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, image)
        return filepath
   
    def binarize_image(self, image):
        _, binary_image = cv2.threshold(image, 90,255, cv2.THRESH_BINARY)
        return binary_image

    def detect_edges(self, image):
        return cv2.Canny(image, 50, 150)

    def detect_lines(self, image):
        lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=60, minLineLength=30, maxLineGap=20)
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), 255, 1)
        return line_image
        '''
パラメータ	 数学的意味	                           画像処理的な意味
rho=1	         ρの分解能（ピクセル単位）        距離の精度
theta=np.pi/180	 θの分解能（1度刻み）	          角度の精度
threshold=100	 投票数の閾値	                           これ以上の点が並んでいると直線と判定
minLineLength    最小線分長（ピクセル）	          これより短い線分は無視
maxLineGap	 許容する最大ギャップ                    線分間の切れ目を補完する許容範囲
'''
        
    def log_image_size(self, image):
    	height, width = image.shape
    	self.get_logger().info(f"PGM画像サイズ: 幅={width} 高さ={height}")



def main(args=None):
    rclpy.init(args=args)
    node = linefilter()
    rclpy.spin_once(node, timeout_sec=2.0)  # 一度だけ処理を実行して終了する
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
