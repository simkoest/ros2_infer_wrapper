import numpy as np

def array_transpose_to_shape(data, transpose_order = (0,1,2)):        
    return data.transpose(transpose_order)

def array_to_contigous_array(data, input_type = np.float32):
    return np.ascontiguousarray(data[None], dtype=input_type)    

def array_choose_random_samples(data, length, sample_size):   
    choice = np.random.choice(length, sample_size, replace=True)
    return data[choice]

def array_expand_dims(data,type='float32'):
    return np.expand_dims(data, 0).astype(type)