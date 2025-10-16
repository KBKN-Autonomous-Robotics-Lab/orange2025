#!/usr/bin/env python3
import rclpy
import math
import numpy as np
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Header
import threading
import time
from my_msgs.srv import Avglatlon

class Odom_Combination(Node):
    def __init__(self):
        super().__init__('odom_combination')

        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 1
        )
        
        # subscription
        self.odom_sub = self.create_subscription(Odometry, '/odom/wheel_spimu', self.get_odom, qos_profile)
        self.gps_odom_sub = self.create_subscription(Odometry, '/odom/UM982', self.get_gps_odom, qos_profile)
        
        # publisher
        self.odom_pub = self.create_publisher(Odometry, '/odom/combine', qos_profile)
        
        # service
        #self.avg_gps_service = self.create_service(Avglatlon, 'send_avg_gps', self.receive_avg_gps_callback)

        self.position_x = 0.0
        self.position_y = 0.0
        self.theta_z = 0.0
        self.theta = 0.0
        self.initial_xy = None #(0.0, 0.0)
        #self.Position_magnification = 1.0  # 必要なら調整

        self.timer = self.create_timer(0.1, self.combine)
    
    # /odom callback
    def get_odom(self, msg):
        '''
        self.position_x = msg.pose.pose.position.x
        self.position_y = msg.pose.pose.position.y
        self.position_z = msg.pose.pose.position.z
        x, y, z, w = msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
        roll, pitch, yaw = quaternion_to_euler(x, y, z, w)
        self.theta_x, self.theta_y, self.theta_z = 0, 0, yaw * 180 / math.pi
        '''
        self.position_x = msg.pose.pose.position.x
        self.position_y = msg.pose.pose.position.y
        x, y, z, w = msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
        roll, pitch, yaw = quaternion_to_euler(x, y, z, w)
        self.theta_z = yaw  # 単位はラジアンのままでOK
    
    # /odom/UM982 callback
    def get_gps_odom(self, msg):
        if self.initial_xy is None:
            init_x = msg.pose.pose.position.x
            init_y = msg.pose.pose.position.y
            self.initial_xy = (init_x, init_y)
            self.get_logger().info(f"Initial /odom/UM982 position set to: x={init_x:.3f}, y={init_y:.3f}")   
    
    '''
    def conversion(self, coordinate, origin, theta):
        ido = coordinate[0]
        keido = coordinate[1]
        ido0 = origin[0]
        keido0 = origin[1]

        # self.get_logger().info(f"theta: {theta}")

        a = 6378137
        f = 35/10439
        e1 = 734/8971
        e2 = 127/1547
        n = 35/20843
        a0 = 1
        a2 = 102/40495
        a4 = 1/378280
        a6 = 1/289634371
        a8 = 1/204422462123
        pi180 = 71/4068
        # %math.pi/180
        d_ido = ido - ido0
        d_keido = keido - keido0
        rd_ido = d_ido * pi180
        rd_keido = d_keido * pi180
        r_ido = ido * pi180
        r_keido = keido * pi180
        r_ido0 = ido0 * pi180
        W = math.sqrt(1-(e1**2)*(math.sin(r_ido)**2))
        N = a / W
        t = math.tan(r_ido)
        ai = e2*math.cos(r_ido)

       # %===Y===%
        S = a*(a0*r_ido - a2*math.sin(2*r_ido)+a4*math.sin(4*r_ido) -
               a6*math.sin(6*r_ido)+a8*math.sin(8*r_ido))/(1+n)
        S0 = a*(a0*r_ido0-a2*math.sin(2*r_ido0)+a4*math.sin(4*r_ido0) -
                a6*math.sin(6*r_ido0)+a8*math.sin(8*r_ido0))/(1+n)
        m0 = S/S0
        B = S-S0
        y1 = (rd_keido**2)*N*math.sin(r_ido)*math.cos(r_ido)/2
        y2 = (rd_keido**4)*N*math.sin(r_ido) * \
            (math.cos(r_ido)**3)*(5-(t**2)+9*(ai**2)+4*(ai**4))/24
        y3 = (rd_keido**6)*N*math.sin(r_ido)*(math.cos(r_ido)**5) * \
            (61-58*(t**2)+(t**4)+270*(ai**2)-330*(ai**2)*(t**2))/720
        gps_y = self.Position_magnification * m0 * (B + y1 + y2 + y3)

       # %===X===%
        x1 = rd_keido*N*math.cos(r_ido)
        x2 = (rd_keido**3)*N*(math.cos(r_ido)**3)*(1-(t**2)+(ai**2))/6
        x3 = (rd_keido**5)*N*(math.cos(r_ido)**5) * \
            (5-18*(t**2)+(t**4)+14*(ai**2)-58*(ai**2)*(t**2))/120
        gps_x = self.Position_magnification * m0 * (x1 + x2 + x3)

        # point = (gps_x, gps_y)Not match

        degree_to_radian = math.pi / 180
        r_theta = theta * degree_to_radian
        h_x = math.cos(r_theta) * gps_x - math.sin(r_theta) * gps_y
        h_y = math.sin(r_theta) * gps_x + math.cos(r_theta) * gps_y
        #point = (-h_y, h_x)
        point = (h_y, -h_x)

        return point    
    '''
    '''
    def receive_avg_gps_callback(self, request, response):
        avg_lat, avg_lon, current_lat, current_lon, theta = request.avg_lat, request.avg_lon, request.current_lat, request.current_lon, request.current_theta
        if theta == 0.0:
            self.get_logger().warn("GPSからのthetaがまだ取得されていません。")
            response.success = False
            return response
        
        self.theta = theta    
        current_coordinate = [current_lat, current_lon]
        initial_coordinate = [avg_lat, avg_lon]
        self.initial_xy = self.conversion(current_coordinate, initial_coordinate, theta)
        xy_np = np.array(self.initial_xy)
    '''
    def yaw_to_orientation(self, yaw):
        orientation_z = np.sin(yaw / 2.0)
        orientation_w = np.cos(yaw / 2.0)
        return orientation_z, orientation_w
    
    def combine(self):
        if self.initial_xy is None:
            self.get_logger().warn("Waiting for initial /odom/UM982 position...")
            return
            
        pos_x = self.position_x
        pos_y = self.position_y
        #degree_to_radian = math.pi / 180
        #pos_theta = self.theta * degree_to_radian # maybe -self.theta * degree_to_radian
        pos_theta = self.theta_z
        # conversionによるxy変換結果（初期座標）
        #delta_x, delta_y = self.initial_xy

        # theta_z（現在の向き）で回転
        #cos_theta = math.cos(pos_theta)
        #sin_theta = math.sin(pos_theta)

        #rotated_x = cos_theta * pos_x - sin_theta * pos_y
        #rotated_y = sin_theta * pos_x + cos_theta * pos_y

        # 加算して新しい位置
        #combined_x = self.initial_xy[0] + rotated_x
        #combined_y = self.initial_xy[1] + rotated_y
        combined_x = self.initial_xy[0] + pos_x
        combined_y = self.initial_xy[1] + pos_y
                
        # quartanion z,w
        odom_orientation = self.yaw_to_orientation(pos_theta)

        # Odometryメッセージ作成
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.pose.pose.position.x = combined_x
        odom_msg.pose.pose.position.y = combined_y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation.x = 0.0
        odom_msg.pose.pose.orientation.y = 0.0
        odom_msg.pose.pose.orientation.z = float(odom_orientation[0])
        odom_msg.pose.pose.orientation.w = float(odom_orientation[1])

        # publish
        self.odom_pub.publish(odom_msg)        

def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw
              
def main(args=None):
    rclpy.init(args=args)
    odom_combination = Odom_Combination()
    rclpy.spin(odom_combination)

if __name__ == '__main__':
    main()          
