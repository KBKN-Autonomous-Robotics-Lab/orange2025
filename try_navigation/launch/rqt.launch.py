from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        ExecuteProcess(
            cmd=['rqt'],
            name='rqt1',
            output='screen'
        ),
        ExecuteProcess(
            cmd=['rqt'],
            name='rqt2',
            output='screen'
        ),
        ExecuteProcess(
            cmd=['rqt'],
            name='rqt3',
            output='screen'
        ),
        ExecuteProcess(
            cmd=['rqt'],
            name='rqt4',
            output='screen'
        ),
    ])

