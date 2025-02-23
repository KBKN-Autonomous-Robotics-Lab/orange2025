#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

class LEDController(Node):
    def __init__(self):
        super().__init__("LED_controller")
        self.declare_parameter("node_flag", 1)
        self.navigation_pub = self.create_publisher(Bool, "/navBool", 10)
        self.timer = self.create_timer(0.5, self.publish_navBool)
        self.navBool_msg = Bool()
        self.navBool_msg.data = True

    def publish_navBool(self):
        node_flag = self.get_parameter("node_flag")
                    .get_parameter_value().integer_value
        if node_flag == 4:
            self.navBool_msg.data = False
        self.navigation_pub.publish(self.navBool_msg)

def main(args=None):
    rclpy.init(args=args)
    lc_node = LEDController()
    rclpy.spin(lc_node)
    lc_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

