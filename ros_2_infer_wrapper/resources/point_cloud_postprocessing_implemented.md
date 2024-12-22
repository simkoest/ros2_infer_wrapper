#  **[PointCloud Postprocessing](../ros_2_infer/ros_2_infer/point_cloud_postprocessing.py)** 

## This Document holds a description of all implemented methods in the Point Cloud Preprocessing file to keep track of the current methods

# Methods: 


## point_cloud_decode_SFA
Specific for the SFA3d Model
### Inputs:
- data (value from previous step (expects SFA3d Model output types))
- cen_offset (center offset output of the model)
- direction (direction output of the model)
- z_coor (z_coordinate output of the model)
- dim (dimension output of the model)

### Output
- Array of Detections

### Functionality
Decodes the Output of the SFA3D Model to filter the outputs into the corresponding detections

## point_cloud_post_process_SFA
Specific for the SFA3d Model
### Inputs:
- data (value from previous step (expects SFA3d Model output types))
- bound_size_x (boundary size of the x axis)
- bound_size_y (boundary size of the y axis)
- bev_height (height of the bird-eye-view-map)
- bev_width (width of the bird-eye-view-map)
- num_classes (number of detectable classes (default = 3))
- down_ratio (down scaling ratio (default = 4))
- peak_trhesh (threshold of the score to filter out bad predititions (default = 0.2))

### Output
- Array of Bounding Boxes

### Functionality
Filter out bounding boxes provided by the SFA 3D Model

## point_cloud_draw_predictions_SFA
Specific for the SFA3d Model
### Inputs:
- data (value from previous step (expects SFA3d Model output types))
- RGB_MAP (Expected RGB Map that was given to as the model input)
- colors (class colors)
- bev_height (height of the bird-eye-view-map)
- bev_width (width of the bird-eye-view-map)
- num_classes (number of detectable classes (default = 3))

### Output
- Bird-Eye-View Image

### Functionality
Draw the predicted bounding boxes of the SFA3D Model to the bird eye view map
