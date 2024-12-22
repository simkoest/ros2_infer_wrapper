#  **[Numpy Array Preprocessing](../ros_2_infer/ros_2_infer/np_array_preprocessing.py.py)** 

## This Document holds a description of all implemented methods in the Numpy Array Preprocessing file to keep track of the current methods

# Methods: 


## array_transpose_to_shape
### Inputs:
- data (value from previous step)
- transpose_order (tuple of the order in which the values should be transposed)

### Output
- Array

### Functionality
Transposes the array based on the tuple (e.g. value (2,1,0) switches the first and 3 dimension)

## array_to_contigous_array
### Inputs:
- data (value from previous step)
- input_type (float by default)

### Output
- Array

### Functionality
Transforms the array to a contigious numpy array and adds a dimension

## array_choose_random_samples
### Inputs:
- data (value from previous step)
- length (tuple of the order in which the values should be transposed)
- sample_size (number of sample entries)

### Output
- Array

### Functionality
Takes a random sample of a an array e.g. for large datasets


## array_expand_dims
### Inputs:
- data (value from previous step)

### Output
- Array

### Functionality
Adds a dimension to the current array e.g. for fitting the model input