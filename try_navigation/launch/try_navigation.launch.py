import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    rviz_config_dir = os.path.join(
        get_package_share_directory('try_navigation'),
        'config', 'config.rviz')
    assert os.path.exists(rviz_config_dir)
    #get livox data    
    livox_to_pointcloud2_launch_file = os.path.join(
        get_package_share_directory('livox_to_pointcloud2'),
        'launch',
        'livox_to_pointcloud2.launch.py'
    )
    
    return LaunchDescription([
        #rviz2
        Node(package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_dir],
            output='screen'
        ),
        #get livox data
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(livox_to_pointcloud2_launch_file)
        ),
        
        #pcd rotation
        Node(package='pcd_convert',
            executable='pcd_rotation',
            name='pcd_rotation_node',
            output='screen',
            arguments=[]
        ),
   
        #gps ekf edit
        Node(package='try_navigation',
            executable='ekf_myself_gps',
            name='sensor_fusion',
            output='screen',
            arguments=[]
        ),
        
        #pcd segmentation
        Node(package='pcd_convert',
            executable='pcd_height_segmentation',
            name='pcd_heigth_segmentation_node',
            output='screen',
            arguments=[]
        ),
        #odom wheel
        #Node(package='try_navigation',
        #    executable='odom_wheel',
        #    name='odom_wheel_node',
        #    output='screen',
        #    arguments=[]
        #),
        
        #waypoint manager
        # waypoint gps command
        Node(package='navigation_control',
            executable='gps_waypoint',
            name='gps_waypoint',
            output='screen',
            arguments=[],
        ),
        # $ ros2 run navigation_control gps_waypoint
        # file path /ros2_ws/src/Use_action/navigation_control/navigation_control/gps_waypoint
        
        #reflection intensity map
        #Node(package='try_navigation',
        #    executable='reflection_intensity_map',
        #    name='reflection_intensity_map_node',
        #    output='screen',
        #    arguments=[],
        #),
        #path planning
        Node(package='try_navigation',
            executable='potential_astar',
            name='potential_astar_node',
            output='screen',
            arguments=[],
        ),
        #robot ctrl
        #Node(package='try_navigation',
        #    executable='path_follower',
        #    name='path_follower_node',
        #    output='screen',
        #    arguments=[],
        #),
        
        #navigation start
        Node(package='navigation_control',
            executable='button',
            name='button',
            output='screen',
            arguments=[],
        ),
        #takamori Autonav
        Node(package='try_navigation',
            executable='reflection_to_pcd',
            name='reflection_to_pcd',
            output='screen',
            arguments=[],
        ),
        #takamori Selfdrive
        #Node(package='try_navigation',
        #    executable='self_drive_line',
        #    name='self_drive_line',
        #    output='screen',
        #    arguments=[],
        #),
    ])
