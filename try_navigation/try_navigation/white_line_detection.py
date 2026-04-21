import rclpy
import numpy as np
import math
import pandas as pd
from rclpy.node import Node
from nav_msgs.msg import Odometry
import sensor_msgs.msg as sensor_msgs
from std_msgs.msg import Header
from rclpy.qos import qos_profile_sensor_data
from sklearn.cluster import DBSCAN

# C++と同じく、Node型を継承します。
class whitelineDetection(Node):
    # コンストラクタです、クラスのインスタンスを作成する際に呼び出されます。
    def __init__(self):
        # コンストラクタです、クラスのインスタンスを作成する際に呼び出されます。
        super().__init__('white_line_detection')

        #make Subscription
        self.subscription = self.create_subscription(
            sensor_msgs.PointCloud2, 
            '/pcd_segment_ground', 
            self.ground_callback,
            qos_profile_sensor_data
            )
        self.subscription = self.create_subscription(
            Odometry,
            '/fusion/odom', 
            self.get_odom,
            1
            )
        self.subscription = self.create_subscription(
            Odometry,
            '/fusion/odom', 
            self.get_ekf_odom,
            1
            )

        #Make Publisher
        self.white_pub = self.create_publisher(
            sensor_msgs.PointCloud2, '/white_candidates', 1
            )
        self.boundary_pub = self.create_publisher(
            sensor_msgs.PointCloud2, '/white_line_boundary', 1
            )
        self.dashed_pub = self.create_publisher(
            sensor_msgs.PointCloud2, '/white_line_dashed', 1
            )

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
        self.csv_saved_local = False
        self.frame_buffer = []
        self.max_frames = 15
        
        #ground 
        self.ground_pixel = 1000/50 #obstacle grid set
        self.map_lim = 20.0  #[m]auto nav 7m  selfdrive:12
    
        self.intensity_threshold = 25 # intensity set

        #dbscan parameter
        self.dbscan_eps = 0.3 #[m]
        self.dbscan_min_samples = 10 #min points for cluster
        self.boundary_min_points = 100 #
        
        self.save_csv = True 
        self.csv_frame_count = 0
        
    def get_odom(self, msg):
        self.position_x = msg.pose.pose.position.x
        self.position_y = msg.pose.pose.position.y
        self.position_z = msg.pose.pose.position.z

        q_x = msg.pose.pose.orientation.x
        q_y = msg.pose.pose.orientation.y
        q_z = msg.pose.pose.orientation.z
        q_w = msg.pose.pose.orientation.w

        roll, pitch, yaw = quaternion_to_euler(q_x, q_y, q_z, q_w)
        
        self.theta_x = 0 #roll /math.pi*180
        self.theta_y = 0 #pitch /math.pi*180
        self.theta_z = yaw #/math.pi*180

    def get_ekf_odom(self, msg):
        self.ekf_position_x = msg.pose.pose.position.x
        self.ekf_position_y = msg.pose.pose.position.y
        self.ekf_position_z = msg.pose.pose.position.z

        q_x = msg.pose.pose.orientation.x
        q_y = msg.pose.pose.orientation.y
        q_z = msg.pose.pose.orientation.z
        q_w = msg.pose.pose.orientation.w

        roll, pitch, yaw = quaternion_to_euler(q_x, q_y, q_z, q_w)
        
        self.ekf_theta_x = 0 #roll /math.pi*180
        self.ekf_theta_y = 0 #pitch /math.pi*180
        self.ekf_theta_z = yaw #/math.pi*180

    def pointcloud2_to_array(self, cloud_msg):
        #Extract point cloud data
        points = np.frombuffer(cloud_msg.data, dtype = np.uint8).reshape(-1, cloud_msg.point_step)
        x = np.frombuffer(points[:, 0:4].tobytes(), dtype = np.float32)
        y = np.frombuffer(points[:, 4:8].tobytes(), dtype = np.float32)
        z = np.frombuffer(points[:, 8:12].tobytes(), dtype = np.float32)
        intensity = np.frombuffer(points[:, 12:16].tobytes(), dtype = np.float32)

        #combine into a*N matrix
        point_cloud_matrix = np.vstack((x, y, z, intensity))

        return point_cloud_matrix

    def ground_callback(self, msg):
        
        #point stamp message
        t_stamp = msg.header.stamp
        
        #get pcd data
        points = self.pointcloud2_to_array(msg)

        # -------- CSV-A保存（1フレームだけ・ローカル座標） --------
        if not self.csv_saved_local:
            np.savetxt(
                '/home/ubuntu/ros2_ws/csv/local_frame.csv',
                points.T,
                delimiter=',',
                header='x,y,z,intensity',
                comments='',
                fmt='%.6f'
            )
            self.csv_saved_local = True
            self.get_logger().info(
                f'CSV-A保存完了: /home/ubuntu/ros2_ws/csv/local_frame.csv ({points.shape[1]} points)'
            )
        # ---------------------------------------------------------

        #position set
        position_x = self.position_x; position_y = self.position_y; position_z = self.position_z;
        position = np.array([position_x, position_y, position_z])
        theta_x=self.theta_x; theta_y=self.theta_y; theta_z=self.theta_z;
        
        ekf_position_x = self.ekf_position_x; ekf_position_y = self.ekf_position_y; ekf_position_z = self.ekf_position_z;
        ekf_position = np.array([ekf_position_x, ekf_position_y, ekf_position_z])
        ekf_theta_x = self.ekf_theta_x; ekf_theta_y = self.ekf_theta_y; ekf_theta_z = self.ekf_theta_z;

        # local to global
        ground_rot, gruond_rot_matrix = rotation_xyz(points[[0,1,2],:], theta_x, theta_y, theta_z)
        ground_x_global = ground_rot[0,:]  + position_x
        ground_y_global = ground_rot[1,:]  + position_y
        ground_z_global = ground_rot[2,:]  + position_z
        ground_global = np.vstack((
            ground_x_global,
            ground_y_global,
            ground_z_global,
            points[3,:]
        )).astype(np.float32)

        # pcd buffer 
        #self.pcd_ground_buff = np.hstack((self.pcd_ground_buff, ground_global))
        self.frame_buffer.append(ground_global)
        if len(self.frame_buffer) > self.max_frames:
            self.frame_buffer.pop(0)
        self.pcd_ground_buff = np.hstack(self.frame_buffer)

        #delete far pcd data
        dist = np.sqrt(
            (self.pcd_ground_buff[0,:] - position_x)**2 + 
            (self.pcd_ground_buff[1,:] - position_y)**2
        )
        self.pcd_ground_buff = self.pcd_ground_buff[:, dist < self.map_lim]

        #delete duplicate
        points_round = np.round(self.pcd_ground_buff * self.ground_pixel) / self.ground_pixel
        mask = ~pd.DataFrame({
            "x": points_round[0,:],
            "y": points_round[1,:],
            "z": points_round[2,:],
        }).duplicated()
        self.pcd_ground_buff = self.pcd_ground_buff[:, mask]

        #intensity filter
        mask = self.pcd_ground_buff[3,:] >= self.intensity_threshold
        white_candidates = self.pcd_ground_buff[:, mask]

        #white pub
        pub_msg = point_cloud_intensity_msg(white_candidates.T, t_stamp, 'odom')
        self.white_pub.publish(pub_msg)

        ##DBSCAN
        boundary_pcd, dashed_pcd = run_dbscan(
            white_candidates,
            self.dbscan_eps,
            self.dbscan_min_samples,
            self.boundary_min_points
        )
        #pub after dbscan
        if boundary_pcd.shape[1] > 0:
            self.boundary_pub.publish(
                point_cloud_intensity_msg(boundary_pcd.T, t_stamp, 'odom'))
        if dashed_pcd.shape[1] > 0:
            self.dashed_pub.publish(
                point_cloud_intensity_msg(dashed_pcd.T, t_stamp, 'odom'))

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

def rotation_xyz(pointcloud, theta_x, theta_y, theta_z):
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

def point_cloud_intensity_msg(points, t_stamp, parent_frame):
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

def run_dbscan(white_candidates, eps, min_samples, boundary_min_points):
    from sklearn.cluster import DBSCAN

    # 点が0のとき早期リターン
    if white_candidates.shape[1] == 0:
        return np.empty((4,0), dtype=np.float32), np.empty((4,0), dtype=np.float32)

    # DBSCANはXYだけ使う
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(white_candidates[:2, :].T)

    # ノイズ（label=-1）を除去
    noise_mask = labels != -1
    white_candidates = white_candidates[:, noise_mask]
    labels = labels[noise_mask]

    # 分類
    boundary_points = []
    dashed_points = []

    for label in np.unique(labels):
        cluster_mask = labels == label
        cluster = white_candidates[:, cluster_mask]

        if cluster.shape[1] >= boundary_min_points:
            boundary_points.append(cluster)
        else:
            dashed_points.append(cluster)

    # 結合
    boundary_pcd = np.hstack(boundary_points) if boundary_points else np.empty((4,0), dtype=np.float32)
    dashed_pcd   = np.hstack(dashed_points)   if dashed_points   else np.empty((4,0), dtype=np.float32)

    return boundary_pcd, dashed_pcd

def main(args=None):
    rclpy.init(args=args)
    node = whitelineDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()