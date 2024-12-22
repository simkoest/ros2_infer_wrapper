# Ros 2 Infer Wrapper

## Description
The ROS 2 Infer Wrapper is used to excecute ONNX-AI-Model Inference in a dedicated Node while allowing different type of sensor inputs. The wrapper can be configured with a yaml file which configures the subscribed and published types, topics and the dedicated pre- and postprocessing steps.

## Current features
- [x] Allowing different sensor inputs like rgb-Images, depth-Images and PointCloud2 inputs from sensors like LiDAR
- [x] Preprocessing pipeline, that can be configured in the yaml file and allows ordering implemented modular steps in any order before inputting the results into the onnx model
- [x] Model inference using onnxruntime executable with or without cuda support
- [x] Postprocessing pipeline, that can be configured in the yaml file which takes the inference output and executes the configured steps before publishing the configured published type
- [x] Simultaneous inputs of sensor data that allows synchronizing e.g. rgb and depth images of a camera as long as their time stamps match, while each input can be processed via a seperate preprocessing pipeline

## Current integrated models
- Simultaneous RGB and depth image segmentation with the EsaNet Model (configured **[here](./ros_2_infer/config/depth_image_segmentation.yaml)**)
- Image Object Detection based on the COCO Dataset with the Faster-RCNN Model (configured **[here](./ros_2_infer/config/image_bboxes_rcnn.yaml)**)
- Image Object Detection based on the Kitti Dataset with the Yolov4 Model (configured **[here](./ros_2_infer/config/image_bboxes.yaml)**)
- Image Classification with the Caffenet-12 Model (configured **[here](./ros_2_infer/config/image_classification.yaml)**)
- Image Segmentation based on the Kitti Dataset with the Bisenet Model (configured **[here](./ros_2_infer/config/image_segmentation_kitti.yaml)**)
- PointCloud Object Detection with the SFA3D Model transforming the PointCloud to a Bird-Eye-View (configured **[here](./ros_2_infer/config/pc_bev_maps.yaml)**)

## 1. Functionality
The way the wrapper is executing is based on the configuration yaml files. Following dummy configuration explains the way the yaml file is set up

```yaml
#Name of the Model in the ros_2_infer/models folder
model_path: Dummy.onnx

# Each subscribed topic dan be set up with individual preprocessing steps
sub_topics: 
    #name of the topic that should be subscribed
  - topic: rgb-camera
    subscribed_type: Image        
    preprocessing_steps:     
    #Each Step receives the result of the previous step and has configurable parameters
    - name: Step 1
      params:
        - name: Param 1
          value: 1
        - name: Param 2
          value: 2
    #Example for integrating 2 seperate subscription topics
  - topic: depth-camera    
    subscribed_type: Image
    preprocessing_steps:         
    - name: Step 1
      params:
        - name: Param 1
          value: 1
          
    - name: Step 2
      params:
        - name: Param 1
          value: 1        

# The postprocesing steps receive the model output as their initial input
postprocessing_steps:
  - name: Step 1
    params:
      - name: Bounding Box
      # In Addition to normal parameters the inputs can be obtained from a specific index of the model output using: For nested values like [0][0] use [0,0] in the configuration
        get_from_output_pos: [0]
        # alternatively use_orig_input_pos gets the initially subscribed message for example to draw bboxes or segmentation to an image

#Type of the result that will be published
publish_type: Detection2DArray
pub_topic: /BoundingBoxes

```

Each step in the pre- and post processing is executed using pythons globals() name lookup where the method is called using the name configured in the yaml file. The methods are implemented in the **[image_preprocesing](./ros_2_infer/ros_2_infer/image_preprocessing.py)**, **[image_postprocessing](./ros_2_infer/ros_2_infer/point_cloud_preprocessing.py)**, **[point_cloud_preprocessing](./ros_2_infer/ros_2_infer/point_cloud_preprocessing.py)** , **[point_cloud_postprocessing](./ros_2_infer/ros_2_infer/point_cloud_postprocessing.py)** , **[np_array_preprocessing](./ros_2_infer/ros_2_infer/np_array_preprocessing.py)**, and the **[type_conversions](./ros_2_infer/ros_2_infer/type_conversions.py)**

The pre and postprocess pipelines executes the steps in the following way:

```python
def preprocess_pipeline(self, orig_input, topic_num):
    t_pre_start = time.time()

    
    kwargs = {"data": orig_input}  
    for step in self.config_reader.configs["sub_topics"][topic_num]["preprocessing_steps"]:
        if step["params"] != None:
            # use the python keyword arugments to pass parameters
            for param in step["params"]:
                kwargs[param["name"]] = param["value"]
        t_start = time.time()
        
        # get the method to execute via pythons global() lookup
        input = globals()[step["name"]](**kwargs)
        t_end = time.time()
        step_name = step["name"]
        self.get_logger().debug(f"Duration of preprocess step {step_name}: {t_end - t_start:.3f}")
        kwargs = {"data": input}

    t_pre_end = time.time()
    self.get_logger().debug(f"Total duration of preprocessing: {t_pre_end - t_pre_start:.3f}")
    return input
```

The following graphic demonstrates the process the wrapper goes through.

![image](/ros_2_infer_wrapper/resources/pipeline_node_process.jpg)

## 2. Getting Started
### 2.1 Requirements
```shell script
git clone https://gitlab.hs-osnabrueck.de/agro-technicum/ag-intelligente-agrarsysteme/student-projects/wp_koesters_ros2_wrapper_ai_model_deployment.git
cd ros_2_infer_wrapper/ros_2_infer
pip install --ignore-installed -r requirements.txt

apt update && \
    apt install -y --no-install-recommends \
    ros-humble-ros-base \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    ros-humble-sensor-msgs \
    ros-humble-sensor-msgs-py
```

### 2.2 Running the node locally
```shell script
cd ros_2_infer_wrapper
colcon build
source install/setup.bash
ros2 run ros_2_infer ros_2_infer
```

once the node is started you can configure the wrapper via a ros paramter. Start a second shell and run

```shell script
ros2 param set /Inference_Node config_name 'depth_image_segmentation.yaml'
```

The name has to match the file name in the **[config](./ros_2_infer/config/)** folder. Make sure that the topics are published to the configured subribed topic in the yaml file using either a demo publisher or a camera like the intel realsense d435i

If values are published to the subscribed topics the wrapper will publish the inference results to the corresponding published topic

### 2.3 Running the node via docker
To run the node with docker check out the **[Dockerfile](./Dockerfile)**. By default the nvidia TensorRT Image is used to allow usage for nvida gpu devices. If your Hardware does not support this, you can replace the image with a standard ubuntu 22.04 one.

To build the container, execute the **[build script](./build-container.sh)** and let it run through. Once it it finished run the **[run script](./run-container.sh)** which will start the wrapper. The model that is being used can be configured via ros2 parameters

once the node is started you can configure the wrapper via a ros paramter. Start a second shell and run

```shell script
ros2 param set /Inference_Node config_name 'depth_image_segmentation.yaml'
```

The name has to match the file name in the **[config](./ros_2_infer/config/)** folder. Make sure that the topics are published to the configured subribed topic in the yaml file using either a demo publisher or a camera like the intel realsense d435i

If values are published to the subscribed topics the wrapper will publish the inference results to the corresponding published topic

## 3. Adding new Models and processing steps
### 3.1 Adding a new Model to the wrapper
- In order to integrate new AI Models into the wrapper, add the onnx model to the **[model folder](./ros_2_infer/models/)**. In addition to that create a new yaml configuration file in the **[config](./ros_2_infer/config/)** folder. 
- Make sure to match the specified format that is used for the other models.
- Edit the expected subscribed and published types to the configuration e.g. Image and Detection2DArray
- Check the existing pre and postprocesing steps to evaluate if you can already fit the model data to your model and add each steps in the configuration
- If there are any specific pre- or postprocessing steps needed that are not yet implemented, add it to the corresponding file
### 3.2 Adding a new process step
If your model needs specific pre-/postprocessing steps that are not yet implemented add a new method to the corresponding file e.g the **[image_preprocessing](./ros_2_infer/ros_2_infer/image_preprocessing.py)**

- Make sure that the first input parameter is named 'data' as this will represent the received data from the step before
- You can add as many additional parameters as you want and fill the method with the required logic. 
- Make sure to follow the naming pattern in order to avoid having duplicate methods
- Make sure to return the result of your method to allow the pipeline to pass the 
result along to the next step

Example for a simple processing method:

```python
def image_resize_by_ratio(data, ratio):
    image = data
    ratio = ratio / min(image.shape[1], image.shape[0])
    image = cv2.resize(image,(int(ratio * image.shape[1]), int(ratio * image.shape[0]))).astype(np.float32)
    return image
```
