"""
This file contains the used augmentation methods.
"""
import cv2
import random
import albumentations as A
from typing import Callable, Tuple, List, Union


class SpatialTRFMS(object):
    """Returning a predefined transformation using its index.

    Args:
        image_size (int): An integer number indicating the output size of
            image in squire.
        angle (integer): An integer number indicating the angle of
            applying rotation. For other spatial transformations it will do nothing.
            The default value is `30` degrees.
        p (float): A floating point number to indicate the probability of
            applying a transformation or not. For the absolute transformations
            this parameter will be ignored. The default value is `0`.

    -- note: The followings are the indexes to each of the individual
        transformations:
            0 --> None
            1 --> Center Crop
            2 --> Random Crop
            3 --> Resize
            4 --> Horisontal Flip
            5 --> Vertical Flip
            6 --> Flip
            7 --> Rotate
            8 --> Elastic
            9 --> Composition (With Elastic)
    Returns:
        A single Albumentations' transformation.
    """
    def __init__(self, 
                 image_size: Tuple=(1024, 1024), 
                 angle: int=30,
                 interpolation: int=1, 
                 border_mode: Callable=cv2.BORDER_CONSTANT,
                 value: int = 0, 
                 alpha: float=0.1, 
                 sigma: float=0.5,
                 alpha_affine: float=1.0, 
                 p: float=0.5, 
                 elast_p: float=0.25):
        self.image_size = image_size
        self.angle = angle
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine
        self.p = p
        self.elast_p = elast_p

        # Define transformations. 
        self.spatial_trfms = {
            0: None,
            1: A.CenterCrop(height=self.image_size[0], width=self.image_size[1],
                            always_apply=True, p=1.0),
            2: A.RandomCrop(height=image_size[0], width=image_size[1], p=1.0),
            3: A.Resize(height=image_size[0], width=image_size[1],
                        interpolation=self.interpolation,
                        p=1.0),
            4: A.HorizontalFlip(always_apply=False, p=self.p),
            5: A.VerticalFlip(always_apply=False, p=self.p),
            6: A.Flip(p=self.p),
            7: A.Rotate(limit=self.angle, interpolation=self.interpolation,
                        border_mode=self.border_mode,
                        value=self.value, always_apply=False, p=self.p),
            8: A.ElasticTransform(alpha=self.alpha, sigma=self.sigma,
                                  alpha_affine=self.alpha_affine,
                                  interpolation=self.interpolation,
                                  border_mode=self.border_mode,
                                  value=None, mask_value=None,
                                  always_apply=False,
                                  approximate=False,
                                  p=self.elast_p),
            9: A.Compose([
                A.HorizontalFlip(always_apply=False, p=self.p),
                A.VerticalFlip(always_apply=False, p=self.p),
                A.Rotate(limit=self.angle,
                         interpolation=self.interpolation,
                         border_mode=self.border_mode,
                         value=self.value, always_apply=False, p=self.p),
                A.ElasticTransform(alpha=self.alpha, sigma=self.sigma,
                                   alpha_affine=self.alpha_affine,
                                   interpolation=self.interpolation,
                                   border_mode=self.border_mode,
                                   value=None, mask_value=None,
                                   always_apply=False,
                                   approximate=False,
                                   p=self.elast_p)
            ])
        }

    def __getitem__(self, item, *args, **kwargs):
        assert item in self.spatial_trfms.keys(), f'The item must be in' \
                                            '{self.spatial_trfms.keys()}.'
        return self.spatial_trfms[item]


class Transformation(object):
    """Returning a complete combination of transformations that would be
        useful for applying into the simages and masks together.
    Args:
        image_size (int): The output size of the image.
        mean (sequence): The mean valuse per channel. 
        std (sequence): The standard deviation per channel. 

    -- note: The following are the list of avaliable indexes for getting the
        appropriate transformations:
        0 --> Just normalize.
        1 --> A strong version of transformations.
    Returns:
        The requested composition of the defined transformations of type
         Albumentations.core.composition.Compose using its item.
    """
    def __init__(self, 
                 image_size: Tuple=(1024, 1024),
                 mean: Union[Tuple, List]=(0.5, 0.5, 0.5), 
                 std: Union[Tuple, List]=(0.5, 0.5, 0.5)):
        self.image_size = image_size
        self.mean = mean 
        self.std = std

        # Define transformations.
        NormTRFM = A.Normalize(
                    mean=self.mean,
                    std=self.std,
                    always_apply=True,
                    p=1.0
        )
        self.trfms = {
            0: NormTRFM,
            1: A.Compose(transforms=[
                  A.ColorJitter(brightness=0.3, 
                                contrast=0.5, 
                                saturation=0.5,
                                hue=0.2, 
                                always_apply=False, p=0.5
                  ),
                  A.OneOf([
                      A.ChannelShuffle(p=1.0),
                      A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50,
                                 p=1.0),
                      A.ChannelDropout(channel_drop_range=(1, 2),
                                       fill_value=0,
                                       p=1.0),
                      A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30,
                                           val_shift_limit=20, p=1.0),
                      A.Emboss(alpha=(0.2, 0.5),
                               strength=(0.2, 0.7),
                               always_apply=False,
                               p=1.0)
                  ], p=0.3),
                  A.OneOf([
                      A.Solarize(threshold=128, p=1.0),
                      A.InvertImg(p=1.0),
                      A.ToGray(p=1.0),
                      A.ToSepia(p=1.0),
                      A.FancyPCA(alpha=0.2, p=1.0), 
                      A.Posterize(num_bits=4, 
                                  always_apply=True, p=1.0), 
                      A.Sharpen(alpha=(0.2, 0.5), 
                                lightness=(0.5, 1.0), 
                                always_apply=False, p=1.0)
                  ], p=0.5),
                  A.OneOf([
                      A.RandomGamma(gamma_limit=(80, 120), eps=None,
                                    always_apply=False,
                                    p=1.0),
                      A.Equalize(mode='cv', by_channels=True, mask=None,
                                 mask_params=(),
                                 p=1.0),
                      A.RandomBrightnessContrast(brightness_limit=0.2,
                                                 contrast_limit=0.1,
                                                 brightness_by_max=True, p=1.0),
                      A.CLAHE(clip_limit=(1.0, 5.0), tile_grid_size=(8, 8), p=1.0)
                  ], p=0.2),
                  A.OneOf([
                      A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                      A.Blur(blur_limit=(3, 7), p=1.0),
                      A.GlassBlur(sigma=0.7, max_delta=4, iterations=2, mode='fast',
                                  p=1.0),
                      A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                      A.MotionBlur(blur_limit=(3, 7), p=1.0),
                      A.MedianBlur(blur_limit=(3, 7), always_apply=False, p=1.0)
                  ], p=0.3),
                  A.OneOf([
                  A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=10,
                               drop_width=1, drop_color=(200, 200, 200),
                               blur_value=5, brightness_coefficient=0.5,
                               rain_type=None, p=1.0),
                  A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.2,
                              alpha_coef=0.08, p=1.0),
                  A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.2,
                               brightness_coeff=1.0, p=1.0),
                  A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0,
                                   angle_upper=1, num_flare_circles_lower=3,
                                   num_flare_circles_upper=5, src_radius=20,
                                   src_color=(255, 255, 255), p=1.0)
                  ], p=0.2),
                  A.OneOf([
                      A.GaussNoise(var_limit=(10.0, 50.0), mean=50, p=1.0),
                      A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False,
                                            elementwise=False, p=1.0),
                      A.ISONoise(color_shift=(0.01, 0.09), intensity=(0.1, 0.5),
                                 p=1.0),
                  ], p=0.2),
              ], p=1.0)
        }

    def __getitem__(self, item):
        return self.trfms[item]
