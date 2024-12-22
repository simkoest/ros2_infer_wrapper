import argparse

#from camera_node import CameraNode
#from lidar_node import LidarNode
#from inference_node import InferenceNode


import rclpy
import sys
import time
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.executors import SingleThreadedExecutor
from rclpy.logging import get_logger
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.qos_event import SubscriptionEventCallbacks
from rclpy.executors import MultiThreadedExecutor

from .infer_pipeline_node import InferPipelineNode

def main():
    rclpy.init()
    qos_profile = QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=10
    )

    node_name = "Inference_Node"
    node = rclpy.create_node(node_name)    
    inference_node = InferPipelineNode(node_name, qos_profile)
    executor = MultiThreadedExecutor()
    executor.add_node(inference_node)    
   
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    inference_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
