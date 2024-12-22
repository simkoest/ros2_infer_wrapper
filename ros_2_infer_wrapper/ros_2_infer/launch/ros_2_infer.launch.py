import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import ThisLaunchFileDir
from launch.substitutions import TextSubstitution
from launch_ros.actions import Node
from launch.launch_context import LaunchContext
import launch_ros.actions
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
import sys
import launch
import os


def generate_launch_description():
    path = os.path.abspath(os.getcwd())
    config = path + '/ros_2_infer_wrapper/ros_2_infer/ros_2_infer/config/depth_image_segmentation.yaml'

    # use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    # #node_name = LaunchConfiguration('node_name')
    # node_name = "LidarNode"
    # rviz = Node(
    #     package='rviz2',
    #     executable='rviz2',
    # )
    # lidar_node =  Node(
    #         package="ros_2_infer",
    #         executable="ros_2_infer",
    #         parameters=[config],
    #         arguments=[node_name]
    # )

    # return LaunchDescription([

    #    # IncludeLaunchDescription(
    #     #    PythonLaunchDescriptionSource(
    #     #        [get_package_share_directory('realsense2_camera'), '/launch/rs_launch.py']),
    #     #),
    #     lidar_node
    #     #rviz
    # ])
    return LaunchDescription([
        Node(
            package='ros_2_infer',
            namespace='ros_2_infer',
            executable='ros_2_infer',            
        )
    ])

    
