import matplotlib.pyplot as plot
import numpy as np
import torch as torch
import torch.optim as optim
from PIL import Image as Image
from PIL import ImageDraw as imDraw
from PIL import ImageFont as imFont
import os
import random
# import torchvision
from math import log10, sqrt
# from skimage.measure import compare_psnr

def get_mask(image, mask_type, max_ratio=0.50):
        return get_mask_with_noise(image, max_ratio)

def get_mask_with_noise(image, max_ratio):
    (h, w) = image.size

    text_font = imFont.load_default();

    img_mask_np = (np.random.random_sample(size=image_to_ndarray(image).shape) > max_ratio).astype(int)

    return ndarray_to_image(img_mask_np)

def image_to_ndarray(image):
    arr = np.array(image)
    if len(arr.shape) == 3:
        arr = arr.transpose(2, 0, 1)
    else:
        arr = arr[None, ...]

    return arr.astype(np.float32) / 255.

def ndarray_to_image(ndarray):
    array = np.clip(ndarray * 255, 0, 255).astype(np.uint8)

    if ndarray.shape[0] == 1:
        array = array[0]
    else:
        array = array.transpose(1, 2, 0)

    return Image.fromarray(array)

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.
    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.
    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]

def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1. / 10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                           np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        assert False

    return net_input

def norm1_loss(x):
    vec = torch.abs(x)
    return torch.sum(vec)

def norm2_loss(x):
    vec = torch.pow(x, 2)
    return torch.sum(vec)