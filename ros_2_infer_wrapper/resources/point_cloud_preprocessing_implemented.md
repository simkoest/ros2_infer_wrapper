#  **[PointCloud Preprocessing](../ros_2_infer/ros_2_infer/point_cloud_preprocessing.py)** 

## This Document holds a description of all implemented methods in the Point Cloud Preprocessing file to keep track of the current methods

# Methods: 


## point_cloud_create_bev_map
### Inputs:
- data (value from previous step (expects an array of a pointcloud))
- boundaries (boundaries of the bird-eye-view)
- bev_size (tuple of the expected size of the bird eye view)

### Output
- Array of Image

### Functionality
Transforms an array of a pointcloud into a bird-eye-view array which can be transformed into a 2 dimensional image

## point_cloud_normalize
### Inputs:
- data (value from previous step (expects an array of a pointcloud))

### Output
- Array of PointCloud

### Functionality
Normalizes the data of a pointcloud

