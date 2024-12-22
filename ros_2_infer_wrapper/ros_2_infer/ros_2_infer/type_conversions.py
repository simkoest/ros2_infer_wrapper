from vision_msgs.msg import BoundingBox2D, BoundingBox3D, Detection2DArray, Detection2D, Classification, Detection3D, Detection3DArray, ObjectHypothesis, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
import open3d as o3d
import numpy as np
from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2

def type_conv_image_to_numpy(data:Image):
    cvbridge = CvBridge()
    return cvbridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

def type_conv_numpy_to_image(data):
    cvbridge = CvBridge()
    return cvbridge.cv2_to_imgmsg(data, encoding="passthrough")

def type_conv_image_to_point_cloud(data:Image, width, height, fx=1.0, fy=1.0, cx=1.0, cy=1.0):
    o3d_depth = o3d.geometry.Image(data)

    o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(
        o3d_depth,
        o3d.camera.PinholeCameraIntrinsic(
        width= width,
        height= height,
        fx=fx, fy=fy,
        cx=cx, cy=cy
    ))

    # Set "header"
    header = Header()
    header.stamp = data.header.stamp
    header.frame_id = data.header.frame_id

    # Set "fields" and "cloud_data"
    points=np.asarray(o3d_cloud.points)
    fields= [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    cloud_data=points

    # create ros_cloud
    return pc2.create_cloud(header, fields, cloud_data)    


def type_conv_pointcloud_to_numpy_array(data: PointCloud2, fields = ('x','y','z','intensity')):
    return np.array(pc2.read_points_list(data, fields),dtype=np.float32)
    

def type_conv_inference_to_detection2darray(data:Image, boxes, labels, scores, score_threshold = 0.7):
    dectArray = Detection2DArray()
    dectArray.detections = []

    header = Header()
    header.frame_id = data.header.frame_id
    header.stamp = data.header.stamp


    for box, label, score in zip(boxes, labels, scores):
        if score > score_threshold:
            detection = Detection2D()
            detection.header = header            
            detection.results = []
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = str(label)
            hypothesis.score = float(score)
            
            detection.results.append(hypothesis)

            box_length_y = float(box[3] - box[1])
            box_length_x = float(box[2] - box[0])
            center_point = [float((box[3] + box[1]) /2), float((box[2] + box[0])/2)]
            box_center = Pose2D()
            box_center.x = center_point[0]
            box_center.y = center_point[1]
            bbox = BoundingBox2D()
            bbox.center = box_center
            bbox.size_x = box_length_x
            bbox.size_y = box_length_y

            detection.bbox = bbox
            detection.source_img = data

            dectArray.detections.append(
                detection
            )
    
    return dectArray

def type_conv_inference_to_classification(data, scores):
    header = Header()
    header.frame_id = data.header.frame_id
    header.stamp = data.header.stamp
    hypos = []

    classification =  Classification()
    classification.header = header

    class_id = 0
    for score in scores:
        hypothesis = ObjectHypothesis()
        hypothesis.class_id = str(class_id)
        hypothesis.score = float(score)
        hypos.append(hypothesis)
        class_id += 1
    
    classification.results = hypos
    return classification


def type_conv_inference_to_detection3darray(data: PointCloud2, boxes):
    a=1