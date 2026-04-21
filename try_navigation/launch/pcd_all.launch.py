from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='livox_to_pointcloud2',
            executable='livox_to_pointcloud2_node',
            name='livox_to_pointcloud2_node',
            output='screen',
        ),
        Node(
            package='pcd_convert',
            executable='pcd_rotation',
            name='pcd_rotation',
            output='screen',
        ),
        Node(
            package='pcd_convert',
            executable='pcd_height_segmentation',
            name='pcd_height_segmentation',
            output='screen',
        ),
        Node(
            package='try_navigation',
            executable='white_line_detection',
            name='white_line_detection',
            output='screen',
        ),
        ExecuteProcess(
            cmd=['rviz2'],
            output='screen',
        ),
    ])
