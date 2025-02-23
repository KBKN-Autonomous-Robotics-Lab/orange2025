#define USE_USBCON
#define SERIAL_PORT Serial

#include <Adafruit_NeoPixel.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>

#define PIN 6
#define ANALOG_PIN A0
#define NUMPIN 30
Adafruit_NeoPixel strip(NUMPIN, PIN, NEO_GRB + NEO_KHZ800);

class LEDControlNode : public rclcpp::Node {
public:
    LEDControlNode() : Node("led_control_node") {
        nav_subscription_ = this->create_subscription<std_msgs::msg::Bool>(
            "/navBool", 10, std::bind(&LEDControlNode::navigation_callback, this, std::placeholders::_1));

        estop_subscription_ = this->create_subscription<std_msgs::msg::Bool>(
            "/estop", 10, std::bind(&LEDControlNode::estop_callback, this, std::placeholders::_1));

        navigation = false;
        stop = false;
    }

private:
    void navigation_callback(const std_msgs::msg::Bool::SharedPtr msg) {
        navigation = msg->data;
        time = millis();
    }

    void estop_callback(const std_msgs::msg::Bool::SharedPtr msg) {
        stop = msg->data;
    }

    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr nav_subscription_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr estop_subscription_;

    volatile bool navigation;
    volatile bool stop;
    unsigned long time;
};

void blueLight() {
    for (int i = 0; i < NUMPIN; i++) strip.setPixelColor(i, 0, 0, 50);
    strip.show();
}

void redLightflash() {
    int j = 0;
    while (j < 5) {
        for (int i = 0; i < NUMPIN; i++) strip.setPixelColor(i, 50, 0, 0);
        strip.show();
        delay(100);
        for (int i = 0; i < NUMPIN; i++) strip.setPixelColor(i, 0, 0, 0);
        strip.show();
        delay(100);
        j++;
        if (!navigation) break;
        if (stop) break;
    }
}

void controlLoopnav3() {
    if (stop && navigation) { blueLight(); }
    else if (stop && !navigation) { blueLight(); }
    else if (!stop && navigation) { redLightflash(); }
    else { blueLight(); }
}

void setup() {
    SERIAL_PORT.begin(115200);
    strip.begin();
    strip.show();
    delay(100);

    rclcpp::init(0, nullptr);
    auto node = std::make_shared<LEDControlNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
}

void loop() {
    if (millis() > time + 1000) { navigation = false; } // WDT (1000ms)
    controlLoopnav3();
    delay(10);
}

