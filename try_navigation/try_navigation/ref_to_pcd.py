

# C++と同じく、Node型を継承します。
class whitelineDetection(Node):
    # コンストラクタです、クラスのインスタンスを作成する際に呼び出されます。
    def __init__(self):
        # コンストラクタです、クラスのインスタンスを作成する際に呼び出されます。
        super().__init__('white_line_detection')

        self.subscription = self.create_subscription(
            sensor_msgs.PointCloud2, 
            '/pcd_segment_ground', 
            self.ground_callback,
            qos_profile_sensor_data
            )
        self.subscription = self.create_subscription(
            Odometry,
            '/fusion/odom', 
            self.odom_callback,
            1
            )

