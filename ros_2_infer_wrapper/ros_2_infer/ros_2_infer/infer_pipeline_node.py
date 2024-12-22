from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult, ParameterDescriptor
from rclpy.parameter import Parameter
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from sensor_msgs.msg import Image, PointCloud2
from vision_msgs.msg import Detection2DArray, Detection3DArray, Classification
from geometry_msgs.msg import Pose2D
from std_msgs.msg import String
import sensor_msgs_py
import message_filters


from cv_bridge import CvBridge 
import numpy as np
import concurrent.futures


from .image_preprocessing import *
from .image_preprocessing import *
from .point_cloud_preprocessing import *
from .point_cloud_postprocessing import *
from .image_postprocessing import *
from .np_array_preprocessing import *
from .torch_tensor_operations import *
from .type_conversions import *

from sensor_msgs.msg import LaserScan
import os
from .config_reader import ConfigReader
from .onnx_backend import ONNXBackend
import os.path as path
import time

class InferPipelineNode(Node):
    def __init__(self,node_name , qos_profile, defer_subscribe=True):
        super().__init__(node_name)
        self.get_logger().info("Inference Pipeline Node started!")
        
        self.qos_profile = qos_profile
        self.thread_executor = concurrent.futures.ThreadPoolExecutor(max_workers=60)
        self.callback_group = MutuallyExclusiveCallbackGroup()
        self.cv_bridge = CvBridge()        

        self.declare_parameter('config_name')
        self.declare_parameter('device', 'cpu')
        
        self.publisher = None
        
        # Wait for the parameter configuration to start running the node
        self.add_on_set_parameters_callback(self.parameter_change_callback)
        self. inference_backend = None


    def init_inference_backend(self, modelpath):
        try:            
            self.inference_backend = ONNXBackend(modelpath)
            return True
        except Exception as e:
            self.get_logger().error(
                f'Cant init Inference Backend: {e}',
                throttle_duration_sec=1)
            return False       
        
    def parameter_change_callback(self, params):
        result = SetParametersResult()        
        # Iterate over each parameter in this node
        for param in params:            
            if param.name == "config_name":                
                # Change the configuration of the node to the specified model
                self.get_logger().info(f'param changed to {param.value}')
                dirname = os.path.dirname(__file__)
                share_path =  path.abspath(path.join(dirname,"../../../../share/ros_2_infer/"))
                configuration = path.join('config/',param.value)
                config = path.join(share_path,configuration)
                self.config_reader = ConfigReader(config)
                
                self.config_reader.read_conf()
                modelname = path.join('models',self.config_reader.configs["model_path"])
                model_path = path.join(share_path,modelname)
                self.get_logger().info(model_path)
                
                if self.init_inference_backend(model_path):
                    self.setup_subscriptions()
                    self.setup_publisher()

                result.successful = True

    
        return result

    def setup_subscriptions(self):
        subscriptions = []
        for topic in self.config_reader.configs["sub_topics"]:            
            sub_type = topic["subscribed_type"]            
            if sub_type == "Image":
                image_sub = message_filters.Subscriber(self, Image, topic["topic"])
                subscriptions.append(image_sub)

            elif sub_type == "PointCloud2":
                subscriptions.append(message_filters.Subscriber(self,  PointCloud2, topic["topic"]))

            elif sub_type == "LaserScan":
                subscriptions.append(message_filters.Subscriber(self, LaserScan, topic["topic"]))

            else:
                self.get_logger().error("Subscribed Type %s not supported!" % (sub_type))            

        ts = message_filters.TimeSynchronizer(subscriptions, 10)
        ts.registerCallback(self.subscriptions_callback)

        self.get_logger().info('Subscriptions created')

    def subscriptions_callback(self, *args):
        try:
            self.get_logger().info('Received Request')            
            future = self.thread_executor.submit(self.process, *args)
            try:
                future.result()
            except Exception as e:
                self.get_logger().error(str(e))
        except Exception as e:
            a = e
            return   


    def process(self, *args):       
        i = 0
        original_inputs = []
        model_inputs= []
        for topic in self.config_reader.configs["sub_topics"]:
            original_inputs.append(args[i])            
            model_inputs.append(self.preprocess_pipeline(args[i],i))
            
            
            i = i+1

        output = self.inference_backend.infer(model_inputs, self.get_logger())


        post = self.postprocess_pipeline(output,original_inputs, model_inputs)
        self.publisher.publish(post)        
        
    
    def preprocess_pipeline(self, orig_input, topic_num):
        t_pre_start = time.time()

        #Use the subscribed input as the initial input for the first step
        kwargs = {"data": orig_input}  
        for step in self.config_reader.configs["sub_topics"][topic_num]["preprocessing_steps"]:
            if step["params"] != None:
                for param in step["params"]:
                    # Add the parameters as keyword arguments
                    kwargs[param["name"]] = param["value"]
            t_start = time.time()
            
            #execute the specified preprocess step name with the gathered arguments
            input = globals()[step["name"]](**kwargs)
            t_end = time.time()
            step_name = step["name"]
            self.get_logger().debug(f"Duration of preprocess step {step_name}: {t_end - t_start:.3f}")

            # set the output of the step as the input for the next step
            kwargs = {"data": input}

        t_pre_end = time.time()
        self.get_logger().debug(f"Total duration of preprocessing: {t_pre_end - t_pre_start:.3f}")
        return input
    
    def postprocess_pipeline(self, model_output, orig_input, model_input):
        t_post_start = time.time()
        input = model_output

        #use the model output as the input for the first postprocessing step
        kwargs = {"data": input}  
        for step in self.config_reader.configs["postprocessing_steps"]:                      
            if step["params"] != None:
                for param in step["params"]:
                    output = model_output
                    input = orig_input
                    mod_input = model_input
                    if "get_from_output_pos" in param:
                        # Allow using nested inputs from the model output e.g. if bboxes are at output postion [0][0] use [0,0] in the configuration
                        for pos in param["get_from_output_pos"]:
                            output = output[pos]
                        kwargs[param["name"]] = output                        
                    elif "get_from_model_input_pos" in param:
                        # Allow using nested inputs from the model input e.g. if the image data is at output postion [0][0] use [0,0] in the configuration
                        for pos in param["get_from_model_input_pos"]:
                            mod_input = mod_input[pos]
                        kwargs[param["name"]] = mod_input
                    elif "use_orig_input_pos" in param:
                        # Allow using the original subscribed data as a paramter input e.g. for drawing boxes to the original image                    
                        for pos in param["use_orig_input_pos"]:
                            input = input[pos]
                        kwargs[param["name"]] = input
                    elif "value" in param and  param["value"] != None:
                        kwargs[param["name"]] = param["value"]

            t_start = time.time()

            # execute the specified preprocess step name with the gathered arguments
            input = globals()[step["name"]](**kwargs)
            t_end = time.time()
            step_name = step["name"]
            self.get_logger().info(f"Duration of postprocessing step {step_name}: {t_end - t_start:.3f}")

            # set the output as the input for the next step
            kwargs = {"data": input}

        t_post_end = time.time()
        self.get_logger().info(f"Total duration of postprocessing: {t_post_end - t_post_start:.3f}")
        return input
    
    def setup_publisher(self):
        publish_type = self.config_reader.configs["publish_type"]        
        if publish_type == "Image":
            self.publisher = self.create_publisher(Image, self.config_reader.configs["pub_topic"], 10)
        elif publish_type == "PointCloud2":
            self.publisher = self.create_publisher(PointCloud2, self.config_reader.configs["pub_topic"], 10)
        elif publish_type == "Detection2DArray":
            self.publisher = self.create_publisher(Detection2DArray, self.config_reader.configs["pub_topic"], 10)
        elif publish_type == "Detection3DArray":
            self.publisher = self.create_publisher(Detection3DArray, self.config_reader.configs["pub_topic"], 10)
        elif publish_type == "Classification":
            self.publisher = self.create_publisher(Classification, self.config_reader.configs["pub_topic"], 10)
        elif publish_type == "String":
            self.publisher = self.create_publisher(String, self.config_reader.configs["pub_topic"], 10)
        else:
            self.get_logger().error(f"Publish Type: {publish_type} is not found!")
