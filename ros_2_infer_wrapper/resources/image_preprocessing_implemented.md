#  **[Image Preprocessing](../ros_2_infer/ros_2_infer/image_preprocessing.py)** 

## This Document holds a description of all implemented methods in the Image Preprocessing file to keep track of the current methods

# Methods: 

## image_normalize
### Inputs:
- data (value from previous step)

### Output
- Array of Image

### Functionality
Normalize the image by dividing the values by 255

## image_resize_image

### Inputs:
- data (value from previous step)
- target_size: tuple holding the width and length of the target size
- interpolation: Interpolation used by the cv2 library to resize (Possible values are INTER_LINEAR, INTER_AREA, INTER_CUBIC, INTER_NEAREST)

### Output
- Array of Image

### Functionality
Resize the image to the given size using cv2

## image_subtract_mean
### Inputs:
- data (value from previous step)
- mean (mean value of the image dataset)

### Output
- Array of Image

### Functionality
Subtract the configured mean value from the image

## image_divide_by_std
### Inputs:
- data (value from previous step)
- std (standard deviation value of the image dataset)

### Output
- Array of Image

### Functionality
Divide the image by configured standard deviation value

## image_resize_and_scale
### Inputs:
- data (value from previous step)
- target_size: tuple holding the width and length of the target size
- padding (add padding to the scaled image (default False))

### Output
- Array of Image

### Functionality
Resizes the image to the target size and scaling it by dividing the width and height values. Add a default padding by filling in blanks with the value 128 if it is configured


## image_resize_by_ratio
### Inputs:
- data (value from previous step)
- ratio (value to multiply the resized image by to scale it)

### Output
- Array of Image

### Functionality
Resizes the image and multiplies the value with the configured ratio to scale it accordingly

## image_pad_image
### Inputs:
- data (value from previous step)
- pad_value

### Output
- Array of Image

### Functionality
Adds a padding to the image based on the configured value

## image_subtract_mean_vec
### Inputs:
- data (value from previous step)
- mean

### Output
- Array of Image

### Functionality
Subtracts a mean vector to all 3 dimension of the array
