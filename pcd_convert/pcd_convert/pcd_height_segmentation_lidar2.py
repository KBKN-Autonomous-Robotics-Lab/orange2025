# rclpy (ROS 2のpythonクライアント)の機能を使えるようにします。
import rclpy
# rclpy (ROS 2のpythonクライアント)の機能のうちNodeを簡単に使えるようにします。こう書いていない場合、Nodeではなくrclpy.node.Nodeと書く必要があります。
from rclpy.node import Node
import std_msgs.msg as std_msgs
import sensor_msgs.msg as sensor_msgs
import numpy as np
import math
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
import time


# C++と同じく、Node型を継承します。
class PcdHeightSegmentation(Node):
    # コンストラクタです、PcdHeightSegmentationクラスのインスタンスを作成する際に呼び出されます。
    def __init__(self):
        # 継承元のクラスを初期化します。
        super().__init__('pcd_heigth_segmentation_node')
        
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
        
        # Subscriptionを作成。
        self.subscription = self.create_subscription(sensor_msgs.PointCloud2, '/pcd_rotation2', self.pcd_heigth_segmentation, qos_profile) #set subscribe pcd topic name
        self.subscription  # 警告を回避するために設置されているだけです。削除しても挙動はかわりません。
        
        # Publisherを作成
        self.pcd_segment_sky_publisher = self.create_publisher(sensor_msgs.PointCloud2, 'pcd_segment_sky', qos_profile) #set publish pcd topic name
       
        #パラメータ
        #set sky range
        self.SKY_HIGHT_MIN =   200/1000; #hight range[m]
        self.SKY_HIGHT_MAX =  4000/1000; #hight range[m]
        self.SKY_MASK_X_MIN = -550/1000; #x mask range[m]
        self.SKY_MASK_X_MAX =  400/1000; #x mask range[m]
        self.SKY_MASK_Y_MIN = -350/1000; #y mask range[m]
        self.SKY_MASK_Y_MAX =  350/1000; #y mask range[m]

        
    def pointcloud2_to_array(self, cloud_msg):
        # Extract point cloud data
        points = np.frombuffer(cloud_msg.data, dtype=np.uint8).reshape(-1, cloud_msg.point_step)
        x = np.frombuffer(points[:, 0:4].tobytes(), dtype=np.float32)
        y = np.frombuffer(points[:, 4:8].tobytes(), dtype=np.float32)
        z = np.frombuffer(points[:, 8:12].tobytes(), dtype=np.float32)
        intensity = np.frombuffer(points[:, 12:16].tobytes(), dtype=np.float32)

        # Combine into a 4xN matrix
        point_cloud_matrix = np.vstack((x, y, z, intensity))
        print(point_cloud_matrix)
        print(f"point_cloud_matrix ={point_cloud_matrix.shape}")
        print(f"x ={x.dtype, x.shape}")
        
        return point_cloud_matrix
        
    def pcd_heigth_segmentation(self, msg):
        
        #print stamp message
        t_stamp = msg.header.stamp
        print(f"t_stamp ={t_stamp}")
        
        #get pcd data
        points = self.pointcloud2_to_array(msg)
        print(f"points ={points.shape}")
        
        #sky segment
        pcd_sky_height = self.height_segment(points, self.SKY_HIGHT_MIN, self.SKY_HIGHT_MAX)
        pcd_sky = self.pcd_mask(pcd_sky_height, self.SKY_MASK_X_MIN, self.SKY_MASK_X_MAX, self.SKY_MASK_Y_MIN, self.SKY_MASK_Y_MAX)
        print(f"pcd_sky_height ={pcd_sky_height.shape}")
        print(f"pcd_sky ={pcd_sky.shape}")
        
        #publish for rviz2
        self.pcd_segment_sky = point_cloud_intensity_msg(pcd_sky.T, t_stamp, 'odom')
        self.pcd_segment_sky_publisher.publish(self.pcd_segment_sky ) 
        
    def height_segment(self, pointcloud, height_min, height_max):
        pcd_ind = ((height_min <= pointcloud[2,:]) * (pointcloud[2,:] <= height_max ))
        pcd_segment = pointcloud[:, pcd_ind]
        return pcd_segment
        
    def pcd_mask(self, pointcloud, x_min, x_max, y_min, y_max):
        pcd_ind = (( (x_min <= pointcloud[0,:]) * (pointcloud[0,:] <= x_max)) * ((y_min <= pointcloud[1,:]) * (pointcloud[1,:]) <= y_max ) )
        pcd_mask = pointcloud[:, ~pcd_ind]
        return pcd_mask
        
    def pcd_serch(self, pointcloud, x_min, x_max, y_min, y_max):
        pcd_ind = (( (x_min <= pointcloud[0,:]) * (pointcloud[0,:] <= x_max)) * ((y_min <= pointcloud[1,:]) * (pointcloud[1,:]) <= y_max ) )
        pcd_mask = pointcloud[:, pcd_ind]
        return pcd_mask
        

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

# mainという名前の関数です。C++のmain関数とは異なり、これは処理の開始地点ではありません。
def main(args=None):
    # rclpyの初期化処理です。ノードを立ち上げる前に実装する必要があります。
    rclpy.init(args=args)
    # クラスのインスタンスを作成
    pcd_height_segmentation = PcdHeightSegmentation()
    # spin処理を実行、spinをしていないとROS 2のノードはデータを入出力することが出来ません。
    rclpy.spin(pcd_height_segmentation)
    # 明示的にノードの終了処理を行います。
    pcd_height_segmentation.destroy_node()
    # rclpyの終了処理、これがないと適切にノードが破棄されないため様々な不具合が起こります。
    rclpy.shutdown()

# 本スクリプト(publish.py)の処理の開始地点です。
if __name__ == '__main__':
    # 関数`main`を実行する。
    main()
