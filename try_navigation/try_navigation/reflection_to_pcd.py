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
#import open3d as o3d
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
import cv2
import subprocess
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


#map save
#ros2 run nav2_map_server map_saver_cli -t /reflect_map_global -f ~/ros2_ws/src/map/test_map --ros-args -p map_subscribe_transient_local:=true -r __ns:=/namespace
#ros2 run nav2_map_server map_saver_cli -t /reflect_map_global --occ 0.10 --free 0.05 -f ~/ros2_ws/src/map/test_map2 --ros-args -p map_subscribe_transient_local:=true -r __ns:=/namespace
#--occ:  occupied_thresh  この閾値よりも大きい占有確率を持つピクセルは、完全に占有されていると見なされます。
#--free: free_thresh	  占有確率がこの閾値未満のピクセルは、完全に占有されていないと見なされます。

# C++と同じく、Node型を継承します。
class ReflectionIntensityMap(Node):
    # コンストラクタです、クラスのインスタンスを作成する際に呼び出されます。
    def __init__(self):
        # 継承元のクラスを初期化します。（https://www.python-izm.com/advanced/class_extend/）今回の場合継承するクラスはNodeになります。
        super().__init__('reflection_intensity_map_node')
        
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 1
        )
        
        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 1
        )
        
        map_qos_profile_sub = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth = 1
        )
        # Subscriptionを作成。CustomMsg型,'/livox/lidar'という名前のtopicをsubscribe。
        self.subscription = self.create_subscription(sensor_msgs.PointCloud2, '/pcd_segment_ground', self.reflect_map, qos_profile)
        self.subscription = self.create_subscription(nav_msgs.Odometry,'/odom', self.get_odom, qos_profile_sub)
        self.subscription = self.create_subscription(nav_msgs.Odometry,'/odom', self.get_ekf_odom, qos_profile_sub)
        #self.subscription = self.create_subscription(nav_msgs.Odometry,'/odom_fast', self.get_odom, qos_profile_sub)
        self.subscription  # 警告を回避するために設置されているだけです。削除しても挙動はかわりません。
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # Publisherを作成
        self.pcd_ground_global_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_ground_global', qos_profile) 
        self.reflect_map_local_publisher = self.create_publisher(OccupancyGrid, 'reflect_map_local', map_qos_profile_sub)
        self.reflect_map_global_publisher = self.create_publisher(OccupancyGrid, 'reflect_map_global', map_qos_profile_sub)
        self.dotted_pub = self.create_publisher(PointCloud2, 'dotted_lines', 10)
        self.solid_pub = self.create_publisher(PointCloud2, 'solid_lines', 10)
        self.solid_pub_buff = self.create_publisher(PointCloud2, 'solid_lines_buff', 10)
        self.publisher_map = self.create_publisher(Image, 'image_map', 10)
        self.publisher_binary = self.create_publisher(Image, 'image_binary', 10)
        self.publisher_oc = self.create_publisher(Image, 'image_oc', 10)
        self.publisher_edge = self.create_publisher(Image, 'image_edge', 10)
        #self.publisher_hough = self.create_publisher(Image, 'image_hough', 10)
        self.bridge = CvBridge()


        self.image_saved = False  # 画像保存フラグ（初回のみ保存する）

        #パラメータ
        #odom positon init
        self.position_x = 0.0 #[m]
        self.position_y = 0.0 #[m]
        self.position_z = 0.0 #[m]
        self.theta_x = 0.0 #[deg]
        self.theta_y = 0.0 #[deg]
        self.theta_z = 0.0 #[deg]
        #ekf_odom positon init
        self.ekf_position_x = 0.0 #[m]
        self.ekf_position_y = 0.0 #[m]
        self.ekf_position_z = 0.0 #[m]
        self.ekf_theta_x = 0.0 #[deg]
        self.ekf_theta_y = 0.0 #[deg]
        self.ekf_theta_z = 0.0 #[deg]
        
        #mid360 buff
        self.pcd_ground_buff = np.array([[],[],[],[]]);
        self.solid_array_buff = np.array([[],[],[]]);
        
        #ground 
        self.ground_pixel = 1000/50#障害物のグリッドサイズ設定
        self.MAP_RANGE = 15.0 #[m]
        
        self.MAP_RANGE_GL = 20.0 #[m]
        self.MAP_LIM_X_MIN = -25.0 #[m]
        self.MAP_LIM_X_MAX =  25.0 #[m]
        self.MAP_LIM_Y_MIN = -25.0 #[m]
        self.MAP_LIM_Y_MAX =  25.0 #[m]
        
        #map position
        self.map_position_x_buff = 0.0 #[m]
        self.map_position_y_buff = 0.0 #[m]
        self.map_position_z_buff = 0.0 #[m]
        self.map_theta_z_buff = 0.0 #[deg]
        self.map_number = 0 # int
        
        self.map_data = 0
        self.map_data_flag = 0
        self.map_data_gl = 0
        self.map_data_gl_flag = 0
        self.MAKE_GL_MAP_FLAG = 1
        self.save_dir = os.path.expanduser('~/ros2_ws/src/kbkn_maps/maps/tsukuba/whiteline')
        yaml.add_representer(OrderedDict, ordered_dict_representer, Dumper=MyDumper)
        yaml.add_representer(list, list_representer, Dumper=MyDumper)
        
    def timer_callback(self):
        if self.map_data_flag > 0:
            self.reflect_map_local_publisher.publish(self.map_data)     
        #gl map
        if self.map_data_gl_flag > 0:
            self.reflect_map_global_publisher.publish(self.map_data_gl) 
        
    def get_odom(self, msg):
        self.position_x = msg.pose.pose.position.x
        self.position_y = msg.pose.pose.position.y
        self.position_z = msg.pose.pose.position.z
        
        flio_q_x = msg.pose.pose.orientation.x
        flio_q_y = msg.pose.pose.orientation.y
        flio_q_z = msg.pose.pose.orientation.z
        flio_q_w = msg.pose.pose.orientation.w
        
        roll, pitch, yaw = quaternion_to_euler(flio_q_x, flio_q_y, flio_q_z, flio_q_w)
        
        self.theta_x = 0 #roll /math.pi*180
        self.theta_y = 0 #pitch /math.pi*180
        self.theta_z = yaw /math.pi*180
        
    def get_ekf_odom(self, msg):
        self.ekf_position_x = msg.pose.pose.position.x
        self.ekf_position_y = msg.pose.pose.position.y
        self.ekf_position_z = msg.pose.pose.position.z
        
        flio_q_x = msg.pose.pose.orientation.x
        flio_q_y = msg.pose.pose.orientation.y
        flio_q_z = msg.pose.pose.orientation.z
        flio_q_w = msg.pose.pose.orientation.w
        
        roll, pitch, yaw = quaternion_to_euler(flio_q_x, flio_q_y, flio_q_z, flio_q_w)
        
        self.ekf_theta_x = 0 #roll /math.pi*180
        self.ekf_theta_y = 0 #pitch /math.pi*180
        self.ekf_theta_z = yaw /math.pi*180
        
	
    def pointcloud2_to_array(self, cloud_msg):
        # Extract point cloud data
        points = np.frombuffer(cloud_msg.data, dtype=np.uint8).reshape(-1, cloud_msg.point_step)
        x = np.frombuffer(points[:, 0:4].tobytes(), dtype=np.float32)
        y = np.frombuffer(points[:, 4:8].tobytes(), dtype=np.float32)
        z = np.frombuffer(points[:, 8:12].tobytes(), dtype=np.float32)
        intensity = np.frombuffer(points[:, 12:16].tobytes(), dtype=np.float32)

        # Combine into a 4xN matrix
        point_cloud_matrix = np.vstack((x, y, z, intensity))
        
        return point_cloud_matrix
        
    def reflect_map(self, msg):
        
        #print stamp message
        t_stamp = msg.header.stamp
        #print(f"t_stamp ={t_stamp}")
        
        #get pcd data
        points = self.pointcloud2_to_array(msg)
        #print(f"points ={points.shape}")
        
        #position set
        position_x=self.position_x; position_y=self.position_y; position_z=self.position_z;
        position = np.array([position_x, position_y, position_z])
        theta_x=self.theta_x; theta_y=self.theta_y; theta_z=self.theta_z;
        
        ekf_position_x=self.ekf_position_x; ekf_position_y=self.ekf_position_y; ekf_position_z=self.ekf_position_z;
        ekf_position = np.array([ekf_position_x, ekf_position_y, ekf_position_z])
        ekf_theta_x=self.ekf_theta_x; ekf_theta_y=self.ekf_theta_y; ekf_theta_z=self.ekf_theta_z;
        
        #ground global
        ground_rot, ground_rot_matrix = rotation_xyz(points[[0,1,2],:], theta_x, theta_y, theta_z)
        ground_x_grobal = ground_rot[0,:] + position_x
        ground_y_grobal = ground_rot[1,:] + position_y
        ground_global = np.vstack((ground_x_grobal, ground_y_grobal, ground_rot[2,:], points[3,:]) , dtype=np.float32)
        
        #map lim set
        map_lim_x_min = position_x + self.MAP_LIM_X_MIN;
        map_lim_x_max = position_x + self.MAP_LIM_X_MAX;
        map_lim_y_min = position_y + self.MAP_LIM_Y_MIN;
        map_lim_y_max = position_y + self.MAP_LIM_Y_MAX;
        map_lim_ind = self.pcd_serch(self.pcd_ground_buff, map_lim_x_min, map_lim_x_max, map_lim_y_min, map_lim_y_max)
        self.pcd_ground_buff = self.pcd_ground_buff[:,map_lim_ind]
        
        #obs round&duplicated  :grid_size before:28239 after100:24592 after50:8894 after10:3879
        pcd_ground_buff = np.insert(self.pcd_ground_buff, len(self.pcd_ground_buff[0,:]), ground_global.T, axis=1)
        points_round = np.round(pcd_ground_buff * self.ground_pixel) / self.ground_pixel
        self.pcd_ground_buff =points_round[:,~pd.DataFrame({"x":points_round[0,:], "y":points_round[1,:], "z":points_round[2,:]}).duplicated()]
        
        #local reflect map
        ground_reflect_conv = self.pcd_ground_buff[3,:]/255*100.0
        map_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        map_data_set = grid_map_set(self.pcd_ground_buff[1,:], self.pcd_ground_buff[0,:], ground_reflect_conv, position, self.ground_pixel, self.MAP_RANGE)
        
        ##ekf pos local reflect map
        ekf_ground_buff_x = self.pcd_ground_buff[0,:] - position[0]
        ekf_ground_buff_y = self.pcd_ground_buff[1,:] - position[1]
        ekf_ground_buff_z = self.pcd_ground_buff[2,:] - position[2]
        ekf_ground_buff = np.vstack((ekf_ground_buff_x, ekf_ground_buff_y, ekf_ground_buff_z))
        ekf_ground_rot, ekf_ground_rot_matrix = rotation_xyz(ekf_ground_buff, ekf_theta_x-theta_x, ekf_theta_y-theta_y, ekf_theta_z-theta_z)
        ekf_ground_set_x = ekf_ground_rot[0,:] + ekf_position[0]
        ekf_ground_set_y = ekf_ground_rot[1,:] + ekf_position[1]
        ekf_ground_set_z = ekf_ground_rot[2,:] + ekf_position[2]
        ekf_ground_set = np.vstack((ekf_ground_set_x, ekf_ground_set_y, ekf_ground_set_z))
        map_data_set_4save = grid_map_set(ekf_ground_set[1,:], ekf_ground_set[0,:], ground_reflect_conv, ekf_position, self.ground_pixel, self.MAP_RANGE)
        print(f"map_data_set ={map_data_set.shape}")
	
        #GL reflect map
        #map_data_gl_set = grid_map_set(self.pcd_ground_buff[1,:], self.pcd_ground_buff[0,:], ground_reflect_conv, position, self.ground_pixel, self.MAP_RANGE_GL)
        map_data_gl_set = grid_map_set(ekf_ground_set[1,:], ekf_ground_set[0,:], ground_reflect_conv, ekf_position, self.ground_pixel, self.MAP_RANGE_GL)
        print(f"map_data_set ={map_data_set.shape}")
	
        
        #publish for rviz2 
        #global ground
        ground_global_msg = point_cloud_intensity_msg(self.pcd_ground_buff.T, t_stamp, 'odom')
        self.pcd_ground_global_publisher.publish(ground_global_msg) 
        #local map
        self.map_data = make_map_msg(map_data_set, self.ground_pixel, position, map_orientation, t_stamp, self.MAP_RANGE, "odom")
        #self.map_data = make_map_msg(map_data_set_4save, self.ground_pixel, ekf_position, map_orientation, t_stamp, self.MAP_RANGE, "odom")
        self.map_data_flag = 1
        #self.reflect_map_local_publisher.publish(self.map_data)     
        #gl map
        self.map_data_gl = make_map_msg(map_data_gl_set, self.ground_pixel, ekf_position, map_orientation, t_stamp, self.MAP_RANGE_GL, "odom")
        #self.reflect_map_global_publisher.publish(self.map_data_gl) 
        self.map_data_gl_flag = 1
        
                 
        
        #####  takamori line filter  ######
        self.process_pgm(map_data_gl_set, position_x, position_y)
        #self.process_pgm(map_data_set, position_x, position_y)
        
         
  
        if self.MAKE_GL_MAP_FLAG == 1:
            #self.make_ref_map(position_x, position_y, theta_z)
            #self.make_ref_map(ekf_position_x, ekf_position_y, ekf_theta_z)
            self.make_ref_map(map_data_gl_set, ekf_position_x, ekf_position_y, ekf_theta_z)
            
    def process_pgm(self, map_data_set, position_x, position_y):
        try:
            #self.get_logger().info("=== [11]開始 ===")
            image, image_data = self.ref_to_image(map_data_set)
            if image is None:  
               self.get_logger().error(f"Failed to load 'map_data_set' ")
               return
           # self.get_logger().info("=== [12]開始 ===")
            #self.log_image_size(image)
            
            #self.get_logger().info("=== [13]開始 ===")
            binary_image = self.binarize_image(image)
            #self.get_logger().info("=== [14]開始 ===")
            
            # カーネル定義（必要に応じて調整）
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

            # Open → Close
            opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
            open_close = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

            # ↑この結果に対して Close → Open
            closed_then = cv2.morphologyEx(open_close, cv2.MORPH_CLOSE, kernel)
            oc_image = cv2.morphologyEx(closed_then, cv2.MORPH_OPEN, kernel)     
                   
            edge_image = self.detect_edges(oc_image)
          
           # self.get_logger().info("=== [15]開始 ===")
            dotted_cloud, solid_cloud = self.classify_lines_to_pointcloud(edge_image, position_x, position_y, step=1.0)
            
        
            
            self.publish_pointclouds(solid_cloud, dotted_cloud)
            #self.publish_pointclouds(self.solid_array_buff, dotted_cloud)
            #self.get_logger().info("=== [16]開始 ===")
            # 元行列

            map_data_set_image_msg = self.bridge.cv2_to_imgmsg(map_data_set, encoding='mono8')
            self.publisher_map.publish(map_data_set_image_msg)
            # 二値化行列
            binary_image_msg = self.bridge.cv2_to_imgmsg(binary_image, encoding='mono8')
            self.publisher_binary.publish(binary_image_msg)
            # OC行列
            oc_image_msg = self.bridge.cv2_to_imgmsg(oc_image, encoding='mono8')
            self.publisher_oc.publish(oc_image_msg)
            # エッジ行列   
            edge_image_msg = self.bridge.cv2_to_imgmsg(edge_image, encoding='mono8')
            self.publisher_edge.publish(edge_image_msg)
            
            self.get_logger().info("=== [5]開始 ===")
            # 画像を1回だけ保存（デバッグ用）
            if not self.image_saved:
               save_dir = "/home/ubuntu/ros2_ws/src/kbkn_maps/maps/tsukuba/whiteline"
               os.makedirs(save_dir, exist_ok=True)
               
               #self.publisher_hough.publish(dotted_pc)
               cv2.imwrite(os.path.join(save_dir, "00_map_data_set.png"), map_data_set)
               cv2.imwrite(os.path.join(save_dir, "01_occupancy_grid_image.png"), image)
               cv2.imwrite(os.path.join(save_dir, "02_binary_image.png"), binary_image)
               cv2.imwrite(os.path.join(save_dir, "03_edge_image.png"), edge_image)
               self.image_saved = True
               self.get_logger().info("中間画像を保存しました")


        except Exception as e:
            self.get_logger().error(f"画像処理中にエラーが発生しました: {e}")    

    def ref_to_image(self, map_data_set):
        """
        map_data_set: 入力となる反射強度や確率などの2次元行列（NumPy配列）
    
        出力:
            - occupancy_grid_image: OpenCV等で画像化可能な行列（uint8）
            - occupancy_grid_data: 数値処理や分析向けの行列（float or int）
        """
        # 閾値の定義
        occ_threshold_param = 0.15  # 占有空間のしきい値
        free_threshold_param = 0.05 # 自由空間のしきい値

        # 0〜1スケールと仮定し100倍する
        occ_threshold = occ_threshold_param * 100  # = 
        free_threshold = free_threshold_param * 100 # = 13

        # 出力配列の初期化（画像出力形式）
        occupancy_grid_image = np.zeros_like(map_data_set, dtype=np.uint8)
    
        # 出力配列の初期化（元の値を保持したデータ形式）
        occupancy_grid_data = np.zeros_like(map_data_set, dtype=np.float32)

        # 処理1: 占有空間（黒: 0）
        occupancy_grid_image[map_data_set >= occ_threshold] = 0
        occupancy_grid_data[map_data_set >= occ_threshold] = 0.0

        # 処理2: 自由空間（白: 255）
        occupancy_grid_image[map_data_set <= free_threshold] = 255
        occupancy_grid_data[map_data_set <= free_threshold] = 1.0

        # 処理3: 未確定領域（グレー階調）
        uncertain_mask = (map_data_set > free_threshold) & (map_data_set < occ_threshold)
        scaled_values = 255 - (map_data_set[uncertain_mask] / occ_threshold * 100)
        occupancy_grid_image[uncertain_mask] = scaled_values.astype(np.uint8)

        # データ用（0〜1のスケーリングを保持）
        occupancy_grid_data[uncertain_mask] = map_data_set[uncertain_mask] / occ_threshold

        return occupancy_grid_image, occupancy_grid_data

           
    def binarize_image(self, image):
        _, binary_image = cv2.threshold(image, 90,255, cv2.THRESH_BINARY)# 90 255
        return binary_image

    def detect_edges(self, image):
        return cv2.Canny(image, 50, 150)#50 150
    
    def classify_lines_to_pointcloud(self, image, position_x, position_y,step=1.0):
    # Hough変換で直線を検出する
        #self.get_logger().info("=== [1]開始 ===")
        #lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=50, minLineLength=40, maxLineGap=20)
        #lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=30) #No1 2025.5.8
        lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=30, minLineLength=10, maxLineGap=30) #

        #self.get_logger().info("=== [2開始 ===")
        dotted_points = []  # 点線として分類される点のリスト
        solid_points = []   # 実線として分類される点のリスト
        
        #self.get_logger().info("=== [3]開始 ===")
        if lines is not None:
           for line in lines:
               x1, y1, x2, y2 = line[0]

            # 線分の長さを計算
               length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # 線分を一定間隔に分割する点の数を計算
               num_points = max(int(length / step), 2)

            # 線分を等間隔に分割した点の座標を生成
               xs = np.linspace(x1, x2, num_points)
               ys = np.linspace(y1, y2, num_points)
              # self.get_logger().info("=== [4]開始 ===")
            # 各点を地図座標（m単位）に変換して分類
            #('resolution', 1/self.ground_pixel), 
            #('origin', [round(position_x - self.MAP_RANGE_GL, 1), round(position_y - self.MAP_RANGE_GL, 1), round(0, 1)]) 
               resolution = 1/self.ground_pixel
               origin_x = round(position_x - self.MAP_RANGE_GL-0.3, 1)
               origin_y = round(position_y - self.MAP_RANGE_GL+0.15, 1)
              # self.get_logger().info("=== [5]開始 ===")
               for x, y in zip(xs, ys):
                   map_x = x * resolution + origin_x
                   map_y = (image.shape[0] - y) * resolution + origin_y
                   map_z = 0.0

                   if length < 80:
                      dotted_points.append([map_x, map_y, map_z])
                   else:
                      solid_points.append([map_x, map_y, map_z])
        #self.get_logger().info("=== [6]開始 ===")
        # NumPyのfloat32型配列に変換
        dotted_array = np.array(dotted_points, dtype=np.float32)
        solid_array = np.array(solid_points, dtype=np.float32)
        
        #solid_array_buff
        #if len(solid_array[0,:]) > 3:
        #    solid_array_buff = np.insert(self.solid_array_buff, len(self.solid_array_buff[0,:]), solid_array.T, axis=1)
        #else:
        #    solid_array_buff = self.solid_array_buff
        #points_round = np.round(solid_array_buff * self.ground_pixel) / self.ground_pixel
        #self.solid_array_buff =points_round[:,~pd.DataFrame({"x":points_round[0,:], "y":points_round[1,:], "z":points_round[2,:]}).duplicated()]
        
        
       #self.get_logger().info("=== [7]開始 ===")
        # ログ出力で確認
        self.get_logger().info(f"点線ポイント数: {len(dotted_array)}, 直線ポイント数: {len(solid_array)}")
        self.get_logger().info(f"点線データの一部: {dotted_array[:5]}")
        self.get_logger().info(f"直線データの一部: {solid_array[:5]}")

        return dotted_array, solid_array
        
    def publish_pointclouds(self, solid_array, dotted_array):
        self.get_logger().info("=== [1]開始 ===")
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        self.get_logger().info("=== [2]開始 ===")
        stamp = self.get_clock().now().to_msg()
        
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "odom"
        
        self.get_logger().info("=== [3]開始 ===")
        solid_pc = pc2.create_cloud(header, fields, solid_array.tolist())
        dotted_pc = pc2.create_cloud(header, fields, dotted_array.tolist())
        self.get_logger().info("=== [4]開始 ===")
        self.solid_pub.publish(solid_pc)
        self.dotted_pub.publish(dotted_pc)
        self.get_logger().info("直線・点線の点群をパブリッシュしました")
    
        
    def make_ref_map(self, image, position_x, position_y, theta_z):
        map_pos_diff = math.sqrt((position_x - self.map_position_x_buff)**2 + (position_y - self.map_position_y_buff)**2)
        map_theta_diff = abs(theta_z -  self.map_theta_z_buff)
        if ( (map_pos_diff > 10) or ((map_pos_diff > 2) and (map_theta_diff > 40)) ):
            map_number_str = str(self.map_number).zfill(3)
            # 保存ディレクトリの絶対パスを取得
            #save_path = os.path.join(self.save_dir, f'waypoint_map_{map_number_str}')
            pgm_filename = os.path.join(self.save_dir, f'waypoint_map_{map_number_str}' + ".pgm")
            pgm_filename_meta = os.path.join(f'waypoint_map_{map_number_str}' + ".pgm")
            yaml_filename = os.path.join(self.save_dir, f'waypoint_map_{map_number_str}' + ".yaml")
            # ディレクトリが存在するか確認、存在しない場合は作成
            os.makedirs(self.save_dir, exist_ok=True)
            '''
            subprocess.run([
                'ros2', 'run', 'nav2_map_server', 'map_saver_cli',
                '-t', '/reflect_map_global',
                '--occ', '0.13',
                '--free', '0.05',
                '-f', save_path,
                '--ros-args', '-p', 'map_subscribe_transient_local:=true', '-r', '__ns:=/namespace'
            ])
            self.get_logger().info(f'External node executed with argument --arg1 {map_number_str}')
            '''
            # 閾値の設定 
            occ_threshold_param = 0.13 # 占有のしきい値  for save
            occ_threshold = occ_threshold_param * 100 # 占有のしきい値 
            free_threshold_param = 0.05 # 自由空間のしきい値  for save 
            free_threshold = free_threshold_param * 100 # 自由空間のしきい値 
            #image = self.map_data_gl
            #image = np.array(self.map_data_gl.data).reshape((self.map_data_gl.info.height, self.map_data_gl.info.width))
            #print(f"image ={image}")
            # マスクを初期化 
            occupancy_grid = np.zeros_like(image) 
            # 占有空間、自由空間、未確定領域を設定 
            occupancy_grid[image >= occ_threshold] = 255 - 255
            # 占有空間 
            occupancy_grid[image <= free_threshold] = 255 - 0
            # 自由空間 
            occupancy_grid[(image > free_threshold) & (image < occ_threshold)] = 255 - (image[(image > free_threshold) & (image < occ_threshold)])/occ_threshold*100 # 未確定領域は元の値を保持 
            # マップの保存 
            cv2.imwrite(pgm_filename, occupancy_grid)
            
            # メタデータを定義 
            metadata = OrderedDict([ 
                ('image', pgm_filename_meta), 
                ('mode', 'trinary'), 
                ('resolution', 1/self.ground_pixel), 
                ('origin', [round(position_x - self.MAP_RANGE_GL, 1), round(position_y - self.MAP_RANGE_GL, 1), round(0, 1)]), 
                ('negate', 0), ('occupied_thresh', occ_threshold_param), 
                ('free_thresh', free_threshold_param) 
            ])
            
            # YAMLファイルとしてメタデータを保存 
            with open(yaml_filename, 'w') as yaml_file: 
                yaml.dump(metadata, yaml_file, Dumper=MyDumper, default_flow_style=False)
            
            
            self.map_position_x_buff = position_x #[m]
            self.map_position_y_buff = position_y #[m]
            self.map_theta_z_buff = theta_z #[deg]
            self.map_number += 1
        
        
    def pcd_serch(self, pointcloud, x_min, x_max, y_min, y_max):
        pcd_ind = (( (x_min <= pointcloud[0,:]) * (pointcloud[0,:] <= x_max)) * ((y_min <= pointcloud[1,:]) * (pointcloud[1,:]) <= y_max ) )
        return pcd_ind
	

# カスタムDumperの設定を追加 
class MyDumper(yaml.Dumper): 
    def increase_indent(self, flow=False, indentless=False): 
        return super(MyDumper, self).increase_indent(flow=flow, indentless=indentless)
def ordered_dict_representer(dumper, data): 
    return dumper.represent_dict(data.items()) 
def list_representer(dumper, data): 
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True) 
    



def rotation_xyz(pointcloud, theta_x, theta_y, theta_z):
    theta_x = math.radians(theta_x)
    theta_y = math.radians(theta_y)
    theta_z = math.radians(theta_z)
    rot_x = np.array([[ 1,                 0,                  0],
                      [ 0, math.cos(theta_x), -math.sin(theta_x)],
                      [ 0, math.sin(theta_x),  math.cos(theta_x)]])
    
    rot_y = np.array([[ math.cos(theta_y), 0,  math.sin(theta_y)],
                      [                 0, 1,                  0],
                      [-math.sin(theta_y), 0, math.cos(theta_y)]])
    
    rot_z = np.array([[ math.cos(theta_z), -math.sin(theta_z), 0],
                      [ math.sin(theta_z),  math.cos(theta_z), 0],
                      [                 0,                  0, 1]])
    rot_matrix = rot_z.dot(rot_y.dot(rot_x))
    #print(f"rot_matrix ={rot_matrix}")
    #print(f"pointcloud ={pointcloud.shape}")
    rot_pointcloud = rot_matrix.dot(pointcloud)
    return rot_pointcloud, rot_matrix

def quaternion_to_euler(x, y, z, w):
    # クォータニオンから回転行列を計算
    rot_matrix = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x**2 + z**2), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x**2 + y**2)]
    ])

    # 回転行列からオイラー角を抽出
    roll = np.arctan2(rot_matrix[2, 1], rot_matrix[2, 2])
    pitch = np.arctan2(-rot_matrix[2, 0], np.sqrt(rot_matrix[2, 1]**2 + rot_matrix[2, 2]**2))
    yaw = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
    return roll, pitch, yaw
    

def point_cloud_intensity_msg(points, t_stamp, parent_frame):
    # In a PointCloud2 message, the point cloud is stored as an byte 
    # array. In order to unpack it, we also include some parameters 
    # which desribes the size of each individual point.
    ros_dtype = sensor_msgs.PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize # A 32-bit float takes 4 bytes.
    data = points.astype(dtype).tobytes() 

    # The fields specify what the bytes represents. The first 4 bytes 
    # represents the x-coordinate, the next 4 the y-coordinate, etc.
    fields = [
            sensor_msgs.PointField(name='x', offset=0, datatype=ros_dtype, count=1),
            sensor_msgs.PointField(name='y', offset=4, datatype=ros_dtype, count=1),
            sensor_msgs.PointField(name='z', offset=8, datatype=ros_dtype, count=1),
            sensor_msgs.PointField(name='intensity', offset=12, datatype=ros_dtype, count=1),
        ]

    # The PointCloud2 message also has a header which specifies which 
    # coordinate frame it is represented in. 
    header = std_msgs.Header(frame_id=parent_frame, stamp=t_stamp)
    

    return sensor_msgs.PointCloud2(
        header=header,
        height=1, 
        width=points.shape[0],
        is_dense=True,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 4), # Every point consists of three float32s.
        row_step=(itemsize * 4 * points.shape[0]), 
        data=data
    )


def make_map_msg(map_data_set, resolution, position, orientation, header_stamp, map_range, frame_id):
    map_data = OccupancyGrid()
    map_data.header.stamp =  header_stamp
    map_data.info.map_load_time = header_stamp
    map_data.header.frame_id = frame_id
    map_data.info.width = map_data_set.shape[0]
    map_data.info.height = map_data_set.shape[1]
    map_data.info.resolution = 1/resolution #50/1000#resolution
    pos_round = np.round(position * resolution) / resolution
    map_data.info.origin.position.x = float(pos_round[0] -map_range) #位置オフセット
    map_data.info.origin.position.y = float(pos_round[1] -map_range)
    map_data.info.origin.position.z = float(0.0) #position[2]
    map_data.info.origin.orientation.w = float(orientation[0])#
    map_data.info.origin.orientation.x = float(orientation[1])
    map_data.info.origin.orientation.y = float(orientation[2])
    map_data.info.origin.orientation.z = float(orientation[3])
    map_data_cv = cv2.flip(map_data_set, 0, dst = None)
    map_data_int8array = [i for row in  map_data_cv.tolist() for i in row]
    map_data.data = Int8MultiArray(data=map_data_int8array).data
    return map_data

'''
フィールド名	内容
image	占有データを含む画像ファイルへのパス。 絶対パス、またはYAMLファイルの場所からの相対パスを設定可能。
resolution	地図の解像度（単位はm/pixel）。
origin	（x、y、yaw）のような地図の左下のピクセルからの2D姿勢で、yawは反時計回りに回転します（yaw = 0は回転しないことを意味します）。現在、システムの多くの部分ではyawを無視しています。
occupied_thresh	この閾値よりも大きい占有確率を持つピクセルは、完全に占有されていると見なされます。
free_thresh	占有確率がこの閾値未満のピクセルは、完全に占有されていないと見なされます。
negate	白/黒について、空き/占有の意味を逆にする必要があるかどうか（閾値の解釈は影響を受けません）
'''

def grid_map_set(map_x, map_y, data, position, map_pixel, map_range):
    map_min_x = (-map_range + position[1] ) * map_pixel
    map_max_x = ( map_range + position[1] ) * map_pixel
    map_min_y = (-map_range + position[0] ) * map_pixel
    map_max_y = ( map_range + position[0] ) * map_pixel
    map_ind_px = np.round(map_x * map_pixel )# index
    map_ind_py = np.round(map_y * map_pixel )
    map_px = np.round(map_x * map_pixel -position[1]*map_pixel )#障害物をグリッドサイズで間引き
    map_py = np.round(map_y * map_pixel -position[0]*map_pixel )
    map_ind = (map_min_x +map_pixel < map_ind_px) * (map_ind_px < map_max_x - (1)) * (map_min_y+map_pixel < map_ind_py) * (map_ind_py < map_max_y - (1))#
    
    #0/1 judge
    #map_xy =  np.zeros([int(map_max_x - map_min_x),int(map_max_y - map_min_y)], np.uint8)
    map_xy =  np.zeros([int(2* map_range * map_pixel),int(2* map_range * map_pixel)], np.uint8)
    map_data = map_xy #reflect to map#np.zeros([int(map_max_x - map_min_x),int(map_max_y - map_min_y),1], np.uint8)
    
    print(f"map_xy ={map_xy.shape}")
    print(f"data ={data.shape}")
    print(f"data(map_ind) ={data[map_ind].shape}")
    
    map_data = map_data.reshape(1,len(map_xy[0,:])*len(map_xy[:,0]))
    map_data[:,:] = 0.0
    map_data_x = (map_px[map_ind] - map_range*map_pixel  ) * len(map_xy[0,:])
    map_data_y =  map_py[map_ind] - map_range*map_pixel
    map_data_xy =  list(map(int, map_data_x + map_data_y ) )
    print(f"map_data ={map_data.shape}")
    print(f"map_data_xy ={len(map_data_xy)}")
    print(f"data[map_ind] ={len(data[map_ind])}")
    
    data_max = np.max(data[map_ind])
    print(f"data_max ={data_max}")
    map_data_xy_max = np.max(map_data_xy)
    print(f"map_data_xy_max ={map_data_xy_max}")
    
    
    map_data[0,map_data_xy] = data[map_ind]
    map_data_set = map_data.reshape(len(map_xy[:,0]),len(map_xy[0,:]))
    
    print(f"map_data_set ={map_data_set.shape}")
    
    #map flipud
    #map_xy = np.flipud(map_xy)
    map_xy = np.flipud(map_data_set)
    
    map_xy_max_ind = np.unravel_index(np.argmax(map_xy), map_xy.shape)
    print(f"map_xy_max_ind ={map_xy_max_ind}")
    print(f"map_xy_max ={map_xy[map_xy_max_ind]}")
    
    return map_xy

# mainという名前の関数です。C++のmain関数とは異なり、これは処理の開始地点ではありません。
def main(args=None):
    # rclpyの初期化処理です。ノードを立ち上げる前に実装する必要があります。
    rclpy.init(args=args)
    # クラスのインスタンスを作成
    reflection_intensity_map = ReflectionIntensityMap()
    # spin処理を実行、spinをしていないとROS 2のノードはデータを入出力することが出来ません。
    rclpy.spin(reflection_intensity_map)
    # 明示的にノードの終了処理を行います。
    reflection_intensity_map.destroy_node()
    # rclpyの終了処理、これがないと適切にノードが破棄されないため様々な不具合が起こります。
    rclpy.shutdown()

# 本スクリプト(publish.py)の処理の開始地点です。
if __name__ == '__main__':
    # 関数`main`を実行する。
    main()
