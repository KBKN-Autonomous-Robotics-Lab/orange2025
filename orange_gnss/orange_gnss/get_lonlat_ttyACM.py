#!/usr/bin/env python3
import rclpy
import serial
import math
import tkinter as tk
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import Header
import threading
import time
from my_msgs.srv import Avglatlon


class GPSData(Node):
    def __init__(self):
        super().__init__('gps_data_acquisition')

        self.declare_parameter('port', '/dev/sensors/GNSSbase')
        self.declare_parameter('baud', 9600)
        self.declare_parameter('country_id', 0)
        self.declare_parameter('heading', 90.0)
        self.declare_parameter('start_lat', 35.425952230280004) # tsukuba start point 36.0497502, 140.0459234 /nakaniwa 35.4257898377487,139.313807281254 /35.425952230280004, 139.31380123427
        self.declare_parameter('start_lon', 139.31380123427)

        self.dev_name = self.get_parameter(
            'port').get_parameter_value().string_value
        self.serial_baud = self.get_parameter(
            'baud').get_parameter_value().integer_value
        self.country_id = self.get_parameter(
            'country_id').get_parameter_value().integer_value

        self.lonlat_pub = self.create_publisher(NavSatFix, "/fix", 1)
        self.lonlat_msg = NavSatFix()

        self.initial_coordinate = None
        self.start_lat = self.get_parameter('start_lat').get_parameter_value().double_value
        self.start_lon = self.get_parameter('start_lon').get_parameter_value().double_value
        self.start_GPS_coordinate = [self.start_lat, self.start_lon]
        self.fix_data = None

        self.timer = self.create_timer(1.0, self.publish_GPS_lonlat)

        self.get_logger().info("Start get_lonlat node")
        self.get_logger().info("-------------------------")
        
        self.tsukuba_theta=self.get_parameter('heading').get_parameter_value().double_value # tsukuba:90.0 nakaniwa 180.0
        self.theta = self.tsukuba_theta # navigation start theta
        
        self.initialized = None
        
        # service client
        self.client = self.create_client(Avglatlon, 'send_avg_gps')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available...")
        
        # tkinter GUI setup
        self.root = tk.Tk()
        self.root.title("GPS Data Acquisition")
        self.start_button = tk.Button(self.root, text="Start GPS Acquisition", command=self.start_gps_acquisition)
        self.start_button.pack()

        self.gps_acquisition_thread = None
        self.is_acquiring = False

    # service client
    def send_request(self):
        request = Avglatlon.Request()
        request.avg_lat = self.initial_coordinate[0]  # ← average lat
        request.avg_lon = self.initial_coordinate[1]  # ← average lon
        request.current_lat = self.current_coordinate[0]  # ← current lat
        request.current_lon = self.current_coordinate[1]  # ← current lon
        #request.theta = self.theta
        request.theta = self.tsukuba_theta # tsukuba start theta
        request.current_theta = self.theta # for tsukuba

        future = self.client.call_async(request)
        future.add_done_callback(self.response_callback)
        
    def response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('サービス送信成功')
            else:
                self.get_logger().warn('サービスは受け取られましたが、処理は失敗しました')
        except Exception as e:
            self.get_logger().error(f'サービス呼び出し失敗: {e}')
    
    # gps data collect
    def start_gps_acquisition(self):
        if not self.is_acquiring:
            self.is_acquiring = True
            self.gps_acquisition_thread = threading.Thread(target=self.acquire_gps_data)
            self.gps_acquisition_thread.start()
    
    def acquire_gps_data(self):
        lat_sum = 0.0
        lon_sum = 0.0
        heading_sum = 0.0
        count = 0
        #serial_port = serial.Serial(self.dev_name, self.serial_baud)
        #line = serial_port.readline()

        start_time = time.time()
        while time.time() - start_time < 10:  # 10 seconds
            GPS_data = self.get_gps(self.dev_name, self.country_id)
            #gps_data = line.split(b",")
            if GPS_data and GPS_data[1] != 0 and GPS_data[2] != 0:
                lat_sum += GPS_data[1]
                lon_sum += GPS_data[2]
                #heading_sum += float(GPS_data[5])
                count += 1
            time.sleep(0.1)  # Slight delay to avoid overwhelming the GPS device

        if count > 0:
            #self.initial_coordinate = [lat_sum / count, lon_sum / count] # calculate average
            self.initial_coordinate = self.start_GPS_coordinate
            self.current_coordinate = [lat_sum / count, lon_sum / count] # for tsukuba
            #self.theta = (heading_sum / count) - 90 # for UM982
            self.initialized = True
            self.get_logger().info(f"Initial coordinate set to: {self.initial_coordinate}")
            self.get_logger().info(f"current coordinate set to: {self.current_coordinate}")
            self.get_logger().info(f"Initial theta set to: {self.theta}")
            self.send_request()
        self.is_acquiring = False
    
    def get_gps(self, dev_name, country_id):
        try:
            serial_port = serial.Serial(dev_name, self.serial_baud)
        except serial.SerialException as serialerror:
            self.get_logger().error(f"Serial error: {serialerror}")
            return None

        initial_letters = None
        if country_id == 0:   # Japan
            initial_letters = "$GNGGA"
        elif country_id == 1:  # USA
            initial_letters = "$GPGGA"

        while True:
            line = serial_port.readline().decode('latin-1')
            gps_data = line.split(',')
            if gps_data[0] == initial_letters:
                break

        Fixtype_data = int(gps_data[6])
        if Fixtype_data != 0:
            satelitecount_data = float(gps_data[7])
            # ddmm.mmmmm to dd.ddddd
            latitude_data = float(gps_data[2]) / 100.0
            if gps_data[3] == 'S':  # south
                latitude_data *= -1
            # ddmm.mmmmm to dd.ddddd
            longitude_data = float(gps_data[4]) / 100.0
            if gps_data[5] == 'W':  # west
                longitude_data *= -1
            altitude_data = float(gps_data[9])
        else:
            latitude_data = 0
            longitude_data = 0
            altitude_data = 0
            satelitecount_data = 0

        serial_port.close()
        gnggadata = (Fixtype_data, latitude_data, longitude_data,altitude_data, satelitecount_data)

        return gnggadata

    def publish_GPS_lonlat(self):
        if not self.initialized:
            # 初期化が完了していないので何もしない
            return 
        lonlat = self.get_gps(self.dev_name, self.country_id)
        if lonlat:
            self.lonlat_msg.header = Header()
            self.lonlat_msg.header.frame_id = "gps"
            self.lonlat_msg.header.stamp = self.get_clock().now().to_msg()

            self.lonlat_msg.status.status = NavSatStatus.STATUS_FIX if lonlat[
                0] != 0 else NavSatStatus.STATUS_NO_FIX
            self.lonlat_msg.latitude = float(lonlat[1])
            self.lonlat_msg.longitude = float(lonlat[2])
            self.lonlat_msg.altitude = float(lonlat[3])

            self.lonlat_pub.publish(self.lonlat_msg)
            # self.get_logger().info(f"Published GPS data: {lonlat}")
        else:
            self.get_logger().error("!!!!-gps data error-!!!!")


def main(args=None):
    rclpy.init(args=args)
    gpslonlat = GPSData()
    ros_thread = threading.Thread(target=rclpy.spin, args=(gpslonlat,))
    ros_thread.start()
    gpslonlat.root.mainloop()  # tkinter GUI表示
    gpslonlat.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
