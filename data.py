"""Wheat Head Segmentation dataset loader.
It uses generator to create and return a new simulated image and its contour.
"""
import cv2
import numpy as np
import pandas as pd
from typing import Callable, Tuple, List, Union

import utils

# Define Constants. 
MASKMINPIXELS = 400


class BackgroundDataset:
    """Load extracted background paths, load, augment, and return images.
    Args:
        metadata_path (str): The `.csv` path of the metadata for loading
            background images.
        transformer (Callable): A list of transformations to be applied to
        the background images. Default is `None`.
    """
    def __init__(self,
                 metadata_path: str,
                 transformer: Union[Callable, None]=None):
        self.metadata_path = metadata_path
        assert self.metadata_path.endswith('.csv'), 'The metadata path must ' \
                                                    'be a csv file.'
        self.background_dataframe = pd.read_csv(self.metadata_path)
        self.transformer = transformer

    def __len__(self):
        return self.background_dataframe.shape[0]

    def __call__(self,
                 item: int):
        path = self.background_dataframe.loc[item, 'Image']
        image = utils.imageReader(path)
        if self.transformer is not None:
            image = self.transformer(image=image)['image']
        return image


class RealObjectDataset:
    """Load extracted foreground objects pathes, and return the transformed
        objects.
    Args:
        metadata_path (str): The `.csv` path of the metadata for loading
            wheat head objects.
        scale_trfms (Callable): The image scaler transformer. Default is `None`.
        transformer (Callable): A list of transformations to be applied to
            the real wheat head objects. Default is `None`.
    """
    def __init__(self,
                 metadata_path: str,
                 transformer: Union[Callable, None]=None):
        self.metadata_path = metadata_path
        assert self.metadata_path.endswith('.csv'), 'The metadata path must ' \
                                                    'be a csv file.'
        self.real_dataframe = pd.read_csv(self.metadata_path)
        self.transformer = transformer

    def __len__(self):
        return self.real_dataframe.shape[0]

    def __call__(self,
                 size: int):
        batch = []
        items = np.random.randint(0, self.__len__(), size)
        for i in items:
            batch.append(self.loadSingleObject(i))
        return batch

    def loadSingleObject(self,
                         item: int):
        path = self.real_dataframe.loc[item, 'Image']
        transparent_object = utils.imageReader(path, reader=cv2.imread,
                                        reader_params=(cv2.IMREAD_UNCHANGED,))
        image = transparent_object[:, :, :3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = transparent_object[:, :, 3] // 255
        if self.transformer is not None:
            augmented = self.transformer(image=image, mask=mask)
            aug_image, aug_mask, flag = utils.crop(image=augmented['image'],
                                                   mask=augmented['mask'],
                                                   tol=0)
            if flag is True and np.sum(aug_mask) > MASKMINPIXELS:
                image, mask = aug_image, aug_mask
        return image, mask


class FakePatchDataset:
    """Load extracted fake objects, and return the transformed
        objects.
    Args:
        metadata_path (str): The `.csv` path of the metadata for loading
            fake head objects.
        transformer (Callable): A list of transformations to be applied to
            the real wheat head objects. Default is `None`.
    """
    def __init__(self,
                 metadata_path: str,
                 transformer: Callable=None):
        self.metadata_path = metadata_path
        assert self.metadata_path.endswith('.csv'), 'The metadata path must ' \
                                                    'be a csv file.'
        self.fake_dataframe = pd.read_csv(self.metadata_path)
        self.transformer = transformer

    def __len__(self):
        return self.fake_dataframe.shape[0]

    def __call__(self, size):
        batch = []
        items = np.random.randint(0, self.__len__(), size)
        for i in items:
            batch.append(self.loadSinglePatch(i))
        return batch

    def loadSinglePatch(self,
                        item: int):
        path = self.fake_dataframe.loc[item, 'Image']
        # Load the transparent image object and split up the image and its mask.
        transparent_patch = utils.imageReader(path, reader=cv2.imread,
                                              reader_params=(cv2.IMREAD_UNCHANGED,))
        image = transparent_patch[:, :, :3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = transparent_patch[:, :, 3] // 255
        # Apply transformations.
        if self.transformer is not None:
            augmented = self.transformer(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
            if np.sum(aug_mask) > MASKMINPIXELS:
                image, mask = aug_image, aug_mask
        return image, mask
