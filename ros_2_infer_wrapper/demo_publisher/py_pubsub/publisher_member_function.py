# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
import numpy as np
import rclpy.time
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2, PointField
import array
import sys
import time
import sensor_msgs.msg as sensor_msgs
import std_msgs.msg as std_msgs
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import cv2
from cv_bridge import CvBridge
import numpy as np
import os.path as path
import os

FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]

BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(PointCloud2, 'point_clouds', 10)
        self.publisher_rgb = self.create_publisher(Image, '/image/rgb', 10)
        self.publisher_depth = self.create_publisher(Image, '/image/depth', 10)
        self.publisher_demo = self.create_publisher(Image, '/image/demo', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):        
        bridge = CvBridge()
        dirname = os.path.dirname(__file__)
        share_path =  path.abspath(path.join(dirname,"../../../../share/py_pubsub/resources"))

        path.join(share_path, 'sample_depth.png')
        depth = cv2.imread(path.join(share_path, 'sample_depth.png'),cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(path.join(share_path, 'sample_rgb.png'),cv2.IMREAD_COLOR)
        dem = cv2.imread(path.join(share_path, 'demo.jpg'),cv2.IMREAD_COLOR)            

        depth = bridge.cv2_to_imgmsg(depth)
        rgb = bridge.cv2_to_imgmsg(rgb)
        dem = bridge.cv2_to_imgmsg(dem)
        pc = np.fromfile(path.join(share_path, '000000.bin'), dtype=np.float32).reshape(-1, 4)
        pc = pc[1:]
        pc = point_cloud(pc,"1")        

        self.publisher_.publish(pc)
        self.publisher_rgb.publish(rgb)
        self.publisher_depth.publish(depth)
        self.publisher_demo.publish(dem)        
        self.i += 1



def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)    

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()





def point_cloud(points, parent_frame):
    """ Creates a point cloud message.
    Args:
        points: Nx3 array of xyz positions.
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    Code source:
        https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0
    """
    ros_dtype = sensor_msgs.PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize  # A 32-bit float takes 4 bytes.

    data = points.astype(dtype).tobytes()
    fields = [sensor_msgs.PointField(
        name=n, offset=i * itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyz')]
    fields.append(sensor_msgs.PointField(name='intensity', offset= 3 * itemsize, datatype=ros_dtype, count=1))
    header = std_msgs.Header(frame_id=parent_frame)

    return sensor_msgs.PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 4),  # Every point consists of three float32s.
        row_step=(itemsize * 4 * points.shape[0]),
        data=data
    )


main()
