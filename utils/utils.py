import cv2
import os
import random
import json
import yaml
import numpy as np
import SimpleITK as sitk
import torch
import matplotlib.pyplot as plt
import albumentations as A

from skimage import io
from typing import Callable, Tuple, List, Union


class Converter(object):
    """Convert images from numpy ndarray to pytorch tensors and vice versa.
    Args:
        image_dtype (type): The desired output type of the image.
        contour_dtype (type): The desired output type of the image contour.
        requires_grad (boolean): Set the requires grad for the image and
            contour or not.
    """
    def __init__(self,
                 image_dtype: type,
                 contour_dtype: type,
                 requires_grad: bool=False):
        assert isinstance(image_dtype,
                          (torch.dtype, type)), "image_dtype must be a callable type."
        self.image_dtype = image_dtype
        assert isinstance(contour_dtype,
                          (torch.dtype, type)), "contour_dtype must be a callable type."
        self.contour_dtype = contour_dtype
        self.requires_grad = requires_grad

    def to_tensor(self, image, contour):
        """Convert the image and its contour image into a torch tensor image.
        This conversion could be done only between numpy arrays and torch
        tensors. It changes the a 3 dimensional channel last image into a
        channel first image.
        """
        # Change image and contour types.
        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                image = image.transpose(2, 0, 1)
            image = torch.tensor(image)
        if isinstance(contour, np.ndarray):
            if contour.ndim == 3:
                contour = contour.transpose(2, 0, 1)
            elif contour.ndim == 2:
                contour = contour[np.newaxis, :, :]
            contour = torch.tensor(contour)
        # Change dtypes and requirements of gradients.
        image = image.type(self.image_dtype).requires_grad_(self.requires_grad)
        contour = contour.type(self.contour_dtype).requires_grad_(self.requires_grad)
        return image, contour

    def to_numpy(self, image, contour):
        """Convert the image and its contour image into a numpy array image.
        This conversion could be done only between numpy arrays and torch
        tensors. It changes the a 3 dimensional channel first image into a
        channel last image.
        """
        # Change image and contour types.
        if isinstance(image, torch.Tensor):
            image = image.numpy()
            if image.ndim == 3:
                image = image.transpose(1, 2, 0)
        if isinstance(contour, torch.Tensor):
            contour = contour.numpy()
            if contour.ndim == 3:
                contour = contour.transpose(2, 0, 1)
            elif contour.ndim == 2: 
                contour = contour[np.newaxis, :, :]
        # Change image and contour dtypes.
        image = image.astype(self.image_dtype)
        contour = contour.astype(self.contour_dtype)
        return image, contour


class ColorBalancer(object):
    """Balance the color of the foreground objects based on the color of the
        background. The goal is to reduce the color distance between a
        foreground object and the background on which object  is going to be overlayed.
    Args:
        None
    Returns:
        None
    """
    def __init__(self):
        pass

    def __call__(self,
                 background: np.ndarray,
                 foreground: np.ndarray,
                 mask: np.ndarray):
        """
        Args:
            background (np.ndarray): The background image.
            foreground (np.ndarray): Wheat head object.
            mask (np.ndarray): The mask image related to the wheat head object.
        Returns:
            The balanced wheat head image.
        """
        # Extract statistical information from both background and foreground
            # images.
        back_mean = background.mean(axis=(0, 1), keepdims=True)
        fore_max = foreground.max(axis=(0, 1), keepdims=True)
        fore_max[fore_max == 0] = 1
        fore_std = foreground.std(axis=(0, 1), keepdims=True)
        # Balance the foreground object.
        foreground = (foreground / fore_max)
        foreground = foreground * back_mean
        if foreground.max() <= 1:
            foreground = foreground.astype(np.float32)
        else:
            foreground = foreground.astype(np.uint8)
        foreground = foreground.astype(np.uint8)
        foreground = cv2.multiply(foreground, np.stack((mask,) * 3, axis=-1))
        return foreground


class ColorSwapper(object):
    """ Swap a specific colormap in an image with a new colormap.
    Args:
        source_color (dict): A list with two elements in which the first one is
            the lower bound value and the second one is upper bound value for Hue.
            The default value is set to the original `green` HSV color range when
            this parameter is `None`.
        destin_color (dict): A list with two elements in which the first one is
            the lower bound value and the second one is the upper bound value for Hue.
            The default value is set to the original `yellow` HSV color range
            when this parameter is `None`.
    """
    def __init__(self,
                 source_color: Tuple=None,
                 destin_color: Tuple=None):
        if source_color is None:
            # Source color is set to the default green color range.
            self.source_color = {
                    'lower': np.array([41, 0, 0]),
                    'upper': np.array([80, 255, 255])
            }
        else:
            self.source_color = {}
            self.source_color['lower'] = np.array([source_color[0], 0, 0])
            self.source_color['upper'] = np.array([source_color[1], 255, 255])

        if destin_color is None:
            # Destination color is set to the default yellow color range.
            self.destin_color = {
                    'lower': np.array([21, 0, 0]),
                    'upper': np.array([40, 255, 255])
            }
        else:
            self.destin_color = {}
            self.destin_color['lower'] = np.array([destin_color[0], 0, 0])
            self.destin_color['upper'] = np.array([destin_color[1], 255, 255])

    def __call__(self,
                 image: np.ndarray):
        # Convert image from RGB to HSV.
        img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # Find where the image pixels intensities are in the range of source
            # color lower and upper bound.
        msk = self.create_mask(img, self.source_color)
        # Calculate the difference between source and destination color.
        diff = self.calc_colors_diff(self.source_color, self.destin_color)
        img[msk > 0] += np.array([diff, 0, 0], dtype=np.uint8)
        # Convert the image back to a RGB image from HSV.
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img

    @staticmethod
    def create_mask(image: np.ndarray,
                    source_color: Tuple):
        """Create a mask image by finding the pixels in the range of source
            colors.
        """
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        mask += cv2.inRange(image, source_color['lower'], source_color['upper'])
        return mask

    @staticmethod
    def calc_colors_diff(source_color: Tuple,
                         dest_color: Tuple):
        """Calculate the difference between source and destination color
            ranges."""
        s_mid = (source_color['lower'][0] + source_color['upper'][0]) // 2
        d_mid = (dest_color['lower'][0] + dest_color['upper'][0]) // 2
        return d_mid - s_mid


class ObjectScaler(object):
    """Scale an image and its mask if the size passes the lower bound
        criteria.
    Args:
        lower_bound (int): If the precomputed size of the image is lower
            than this number, this transformation will be applied.
        lower_bound_std (int): If the precomputed output size of the image is
            lower than the lower_bound, the new output size will be
            calculated using the lower bound as the center and this number as
            the standard deviation.
        reduction_ratios (sequence): A list of positive float numbers which
            are less than 1, and the image size will be reduced with one random
            ratio from this sequence.
        p (float): The probability of applying this transformation on the
            input image and mask.
    """
    def __init__(self,
                 lower_bound: int=20,
                 lower_bound_std: int=3,
                 reduction_ratios: List=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                 p: float=0.5):
        self.lower_bound = lower_bound
        self.lower_bound_std = lower_bound_std
        self.reduction_ratios = reduction_ratios
        self.p = p

    def __call__(self,
                 image,
                 mask):
        if random.random() < self.p:
            image = image.copy()
            mask = mask.copy()
            height, width = image.shape[:2]
            if height < self.lower_bound or width < self.lower_bound:
                return image, mask
            red_ratio = random.choice(self.reduction_ratios)
            out_size = min(height, width) * red_ratio
            if out_size < self.lower_bound:
                if random.random() > 0.5:
                    out_size = (self.lower_bound + self.lower_bound_std * random.random())
                else:
                    out_size = (self.lower_bound - self.lower_bound_std * random.random())
                red_ratio =  out_size / min(height, width)
            height = int(height * red_ratio)
            width = int(width * red_ratio)
            scl_tr = A.Resize(height=height, width=width,
                              interpolation=1, always_apply=False, p=1.0)
            augmented = scl_tr(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        return image, mask


class Visualizer(object):
    def __init__(self,
                 image_paths: List,
                 masks_paths: List):
        self.images = image_paths
        self.masks = masks_paths

    def __call__(self):
        for img_pth, msk_pth in zip(self.images, self.masks):
            image = io.imread(img_pth)
            mask = io.imread(msk_pth)
            plt.imshow(image)
            plt.imshow(mask, alpha=0.3, cmap='spring')
            plt.show()


def iouCalculator(mask: np.ndarray,
                  patch: np.ndarray):
    """Calculate the Intersection between an object mask and a patch of the
        original image contour.
    """
    # Check if all the pixels are 0 or 1.
    mask = mask.copy()
    mask[mask > 1] = 1
    patch = patch.copy()
    patch[patch > 1] = 1
    assert patch.shape[0] == mask.shape[0]
    assert patch.shape[1] == mask.shape[1]
    intersect = np.logical_and(patch, mask)
    union = np.logical_or(patch, mask)
    intersect = intersect.sum()
    union = union.sum()
    assert union != 0, 'IoU denominator can not be `zero`.'
    return intersect / union

def get_color_range(image_paths: List,
                    saturation_bound: int=15):
    """Calculate the range of dominant non-black colors in a list of related images.
        The range is calculated in HSV format.
    """
    assert isinstance(saturation_bound, int)

    images_mean = []
    for pth in image_paths:
        img = cv2.imread(pth, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        images_mean.append(np.sum(img, axis=(0, 1)) /
                           np.count_nonzero(img, axis=(0, 1)))
    Hue = np.mean(list(zip(*images_mean))[0])
    lower_bound = np.array([max(0, Hue - saturation_bound), 0, 0]).astype(np.int)
    upper_bound = np.array([min(180, Hue + saturation_bound), 255, 255]).astype(np.int)
    return lower_bound, upper_bound

def crop(image: np.ndarray,
         mask: np.ndarray,
         tol: int=0):
    """Crop the regions that contain no pixes of a object."""
    new_mask = image > tol
    # Coordinates of non-black pixels.
    coords = np.argwhere(new_mask[:, :, 0])
    if 0 in coords.shape:  # If there is no object.
        return image, mask, False
    # Bounding box of non-black pixels.
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1  # slices are exclusive at the top
    # Get the contents of the bounding box.
    image = image[r0:r1, c0:c1, :]
    mask = mask[r0:r1, c0:c1]
    return image, mask, True

def configLoader(config_path: str):
    assert (config_path.endswith('.yaml') or
            config_path.endswith('.yml') or
            config_path.endswith('.json'))

    with open(config_path, 'r') as fin:
        "Reading config file which can be either `Json` or `Yaml` file."
        if config_path.endswith('.json'):
            configs = json.load(fin)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            configs = yaml.load(fin, Loader=yaml.FullLoader)
        else:
            raise ValueError('Only `Json` or `Yaml` configs are acceptable.')
    return configs

def imageReader(path: str,
                reader: Union[Callable, None]=None,
                reader_params: Tuple=tuple()):
    if reader is None:
        if path.endswith('.nrrd'):
            image = sitk.GetArrayFromImage(sitk.ReadImage(path))
            image = np.squeeze(image)
        else:
            image = io.imread(path)
    else:
        image = reader(path, *reader_params)
    return image
