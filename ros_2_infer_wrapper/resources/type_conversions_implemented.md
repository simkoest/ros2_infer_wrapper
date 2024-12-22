#  **[Type Conversions](../ros_2_infer/ros_2_infer/type_conversions.py)** 

## This Document holds a description of all implemented methods in the type conversions file to keep track of the current methods

# Methods: 


## type_conv_image_to_numpy
### Inputs:
- data (value from previous step of type ROS2 Image)

### Output
- Array of Image

### Functionality
Converts the ROS2 Image to an array using the CV Bridge

## type_conv_numpy_to_image
### Inputs:
- data (value from previous step of type Array)

### Output
- ROS2 Image

### Functionality
Converts an array of imagepoints to a ROS2 Image using the CV Bridge

## type_conv_image_to_point_cloud
### Inputs:
- data (value from previous step of type ROS2 Image)
- width (width of the image)
- width (height of the image)
- optional:
    Add Camera parameters for conversion matrix
    - fx (default 1)
    - fy (default 1)
    - cx (default 1)
    - cy (default 1)


### Output
- ROS2 PointCloud2

### Functionality
Converts a ROS2 Image to a PointCloud2 Message

## type_conv_pointcloud_to_numpy_array
### Inputs:
- data (value from previous step of type ROS2 PointCloud)
- fields (array of epected fieldsnames (default ['x', 'y', 'z', 'intensity']))

### Output
- Numpy Array of PointCloud Data

### Functionality
Converts a ROS2 PointCloud2 Message to an Array

## type_conv_inference_to_detection2darray
### Inputs:
- data (original Input Image)
- boxes (model output boxes)
- labels (model output labels)
- scores (model output scores)
- score_treshold (minimum score of the box to be added)

### Output
ROS2 Detection2DArray Message

### Functionality
 Converts the Inference Output of a model to a Detection2DArray Messa

 ## type_conv_inference_to_classification
### Inputs:
- data (original Input ROS Message (Pointcloud or Image))
- scores (model_output scores)

### Output
ROS2 Classification Message

### Functionality
 Converts the Inference Output of a model to a ROS2 Classificatoin Message

