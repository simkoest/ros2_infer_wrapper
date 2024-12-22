import cv2
import numpy as np
import math

def image_normalize(data):
    return data / 255

# def normalize_image(data):        
#     rgb_mean = np.array([0.485, 0.456, 0.406])
#     rgb_std  = np.array([0.229, 0.224, 0.225])
#     data = normalize(data)
#     data = subtract_mean(data,rgb_mean)
#     data = divide_by_std(data, rgb_std)
#     #image = image.astype(np.float32)
#     #image = image / 255.0
#     #image = image - rgb_mean
#     #image = image / rgb_std
#     return data

def image_resize_image(data, target_size, interpolation="INTER_LINEAR"):
    if interpolation == "INTER_LINEAR":
        return cv2.resize(data, target_size, interpolation=cv2.INTER_LINEAR)
    if interpolation == "INTER_AREA":
        return cv2.resize(data, target_size, interpolation=cv2.INTER_AREA)
    if interpolation == "INTER_CUBIC":
        return cv2.resize(data, target_size, interpolation=cv2.INTER_CUBIC)
    if interpolation == "INTER_NEAREST":
        return cv2.resize(data, target_size, interpolation=cv2.INTER_NEAREST)
    else:
        print("interpolation not configured")
        return
#def subtract_mean_cv2(data_) 


def image_subtract_mean(data, mean):
    mean = np.array(mean)
    data = data.astype(np.float32)
    return data - mean

def image_divide_by_std(data, std):
    std = np.array(std)
    data = data.astype(np.float32)
    return data / std

# def transpose_to_shape(data, target_shape, transpose_order = (0,1,2), input_type = np.float32):
#     addDimension = False
#     datashape = data.shape[:3]
#     if len(target_shape) == 4:
#         target_shape = target_shape[1:4]
#         addDimension = True
#     elif len(target_shape) == 3:
#         target_shape = target_shape[:3]    
    
#     #data = data.transpose(posx,posy,posc)
#     data = data.transpose(transpose_order)    
#     if addDimension:
#         return np.ascontiguousarray(data[None], dtype=input_type)
#     else:
#         return np.ascontiguousarray(data,dtype=input_type)    
    
def image_resize_and_scale(data, target_size, padding=False):
    target_height, target_width = target_size
    image_height, image_width = data.shape[:2]
    scale = min(target_width/image_width, target_height/image_height)
    scaled_width, scaled_height = int(scale * image_width), int(scale * image_height)

    if(padding):
        image_resized = cv2.resize(data, (scaled_width, scaled_height))
        image_padded = np.full(shape=[target_height, target_width, 3], fill_value=128.0)
        dw, dh = (target_width - scaled_width) // 2, (target_height-scaled_height) // 2
        image_padded[dh:scaled_height+dh, dw:scaled_width+dw, :] = image_resized
        image_padded = image_padded / 255.  
        return image_padded
    
    image_resized = np.zeros([target_height, target_width, 3])
    image_resized[0:scaled_height, 0:scaled_width] = cv2.resize(data, (scaled_width, scaled_height),
                                                interpolation=cv2.INTER_LINEAR)

    return image_resized

def image_resize_by_ratio(data, ratio):
    image = data
    ratio = ratio / min(image.shape[1], image.shape[0])
    image = cv2.resize(image,(int(ratio * image.shape[1]), int(ratio * image.shape[0]))).astype(np.float32)
    return image

def image_pad_image(data, pad_value):
    image = data
    padded_h = int(math.ceil(image.shape[1] / pad_value) * pad_value)
    padded_w = int(math.ceil(image.shape[2] / pad_value) * pad_value)

    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, :image.shape[1], :image.shape[2]] = image
    image = padded_image
    return image

def image_subtract_mean_vec(data, mean):
    mean = np.array(mean)
    image = data
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean[i]
    return image