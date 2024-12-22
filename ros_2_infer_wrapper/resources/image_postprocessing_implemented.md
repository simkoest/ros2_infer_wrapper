#  **[Image Postprocessing](../ros_2_infer/ros_2_infer/image_postprocessing.py)** 

## This Document holds a description of all implemented methods in the Image Postprocessing file to keep track of the current methods

# Methods: 


## image_deresize
### Inputs:
- data (value from previous step)
- size (original size to resize back to)

### Output
- Array of Image

### Functionality
Resize the image back to the original shape and scale it back

## image_draw_bounding_boxes
### Inputs:
- data (value from previous step)
- boxes (resulting output bounding boxes of the model)
- labels (resulting output labels of the model)
- scores (resulting output scores of the model)
- scale_boxes_by_ratio (value to scale the boxes by (default 1))
- score_threshold (minimum threshhold of the score of the boxes to be displayed)

### Output
- Array of Image 

### Functionality
Draws bounding boxes based on the model output onto the image while only selecting boxes with a minimum score threshold

### Functionality
Resize the image back to the original shape and scale it back

## image_colorize_image_segmentation
### Inputs:
- data (value from previous step, filled with the model output segmentation)
- rgb_image (original input image)
- color_pallete (array of color values for each class)
- opacity

### Output
- Array of Image 

### Functionality
Colors images based on the model output segmentation and the configured colors using a defined opacity

## image_define_anchor_boxes
Specific to the usage in Yolov4
### Inputs:
- data (value from previous step, filled with the model output boxes)
- anchors (array of defined anchors)
- strides (array of strides)
- xyscale (scaling for x and y coordinates)

### Output
- Array of Anchor Boxes 

### Functionality
Filters out the specified anchor boxes

## remove_low_prob_boxes
Specific to the usage in Yolov4
### Inputs:
- data (value from previous step, filled with the model output boxes)
- org_img_shape (shape of the original image)
- input_size (tuple of the size of the input image)
- score_threshold (minimum score for boxes to be kept)

### Output
- Array of Bouding Boxes

### Functionality
Removes Boxes with a low detection probability and discard invalid boxes (Used for yolov4 to get rid of unnecessary boxes)


## image_non_maximum_suppression
Specific to the usage in Yolov4
### Inputs:
- data (value from previous step, filled with the model output boxes)
- org_img_shape (shape of the original image)
- iou_threshold (threshold of the intersection over union value)
- score_threshold (minimum score for boxes to be kept)

### Output
- Array of Bouding Boxes

### Functionality
Performs non maximum suppression based on the yolov4 model output

## image_bboxes_iou
Specific to the usage in Yolov4
### Inputs:
- data (value from previous step, filled with the model output boxes)
- org_img_shape (shape of the original image)
- iou_threshold (threshold of the intersection over union value)
- score_threshold (minimum score for boxes to be kept)

### Output
- Array of Bouding Boxes

### Functionality
Performs non maximum suppression based on the yolov4 model output


## image_color_label
### Inputs:
- data (value from previous step, filled with the model output segmentation)
- org_img_shape (array of colors per class)

### Output
- Array of Image

### Functionality
Color image segmentation based on the given colors

