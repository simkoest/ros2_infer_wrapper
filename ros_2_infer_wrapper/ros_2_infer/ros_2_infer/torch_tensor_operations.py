import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

def numpy_to_torch_tensor(data):
    data = np.array(data)
    return torch.from_numpy(data).float()

def torch_normalize(data, mean, std):
    return torchvision.transforms.Normalize(mean=mean, std=std)(data)

def torch_interpolate(data, size,mode='bilinear', align_corners=False):
    return torch.nn.functional.interpolate(data, size=size, mode=mode, align_corners=align_corners)

def torch_argmax(data):
    return torch.argmax(data, dim=1)

def torch_tensor_to_numpy(data, type=np.uint8):
    return data.cpu().numpy().squeeze().astype(type)