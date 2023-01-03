"""
Extract video frames from list of videos by a user-defined interval.
Extract all the objects are exists in a list of segmented images.
"""
import os
import cv2
import argparse
import numpy as np
import pandas as pd
import random
import SimpleITK as sitk
from skimage import io
from PIL import Image
from typing import Callable, List, Tuple, Union

import utils

SEED = 123
np.random.seed(SEED)
random.seed(SEED)


class RealObjectExtractor:
    """Extract real and fake wheat head objects inside wheat field frames.
    Args:
        fore_image_ext (str):
        min_object_size (str):
        swappers (Callable):
        transformer (Callable):
    """
    def __init__(self,
                 segment_extension: str='.png',
                 min_object_size: int=400,
                 swapper: Union[Callable, None]=None,
                 transformer: Union[Callable, None]=None):
        self.seg_ext = segment_extension
        self.min_obj_size = min_object_size
        self.swapper = swapper
        self.transformer = transformer

    def __call__(self,
                 image_path: str,
                 mask_path: str,
                 out_dir: str):
        print('Processing:', image_path)
        image = utils.imageReader(image_path)
        mask = utils.imageReader(mask_path)
        image_ext_length = len(os.path.basename(image_path).split('.')[-1]) + 1
        base_name = os.path.basename(image_path)[:-image_ext_length]
        out_dir = os.path.join(out_dir, 'real')
        os.makedirs(out_dir, exist_ok=True)
        real_collection = self.objectExtractor(image, mask, out_dir, base_name)
        print(f'{len(real_collection)} objects have been extracted.')
        return real_collection

    def objectExtractor(self,
                        image: np.ndarray,
                        mask: np.ndarray,
                        base_dir: str,
                        base_name: str):
        collection = []
        for k, obj in enumerate(np.unique(mask)):
            if obj == 0:
                continue
            object_patch = (mask == obj)
            rows, columns = np.where(object_patch)
            high_lower, hight_upper = np.min(rows), np.max(rows)
            width_lower, width_upper = np.min(columns), np.max(columns)
            # Define the height and width of the found object.
            height, width = hight_upper - high_lower, width_upper - width_lower
            if np.sum(object_patch) < self.min_obj_size:
                continue
            # Extract the current object from the image and mask.
            object_mask = mask[high_lower:hight_upper,
                               width_lower:width_upper].copy()
            object_image = image[high_lower:hight_upper,
                                 width_lower:width_upper].copy()
            # Clean the extracted patch and just keep the current object.
            object_mask[object_mask != obj] = 0
            object_mask[object_mask == obj] = 255
            object_image[object_mask == 0] = 0
            # Apply color swapper.
            if self.swapper is not None:
                object_image = self.swapper(object_image)
            if self.transformer is not None:
                augmented = self.transformer(image=object_image,
                                             mask=object_mask)
                object_image = augmented['image']
                object_mask = augmented['mask']
            segment = np.concatenate(
                    (object_image.astype(np.uint8),
                    object_mask[..., np.newaxis].astype(np.uint8)), axis=-1)
            segment = Image.fromarray(segment)
            segment_path = os.path.join(base_dir,
                                        f'{base_name}_rlobj{k:0>4}{self.seg_ext}')
            segment.save(segment_path)
            collection.append(segment_path)
        return collection


class FakeObjectExtractor:
    def __init__(self,
                 segment_extension: str='.png',
                 num_fakes_per_real: int=10,
                 swapper: Union[Callable, None]=None,
                 transformer: Union[Callable, None]=None):
        self.seg_ext = segment_extension
        self.num_fakes_per_real = num_fakes_per_real
        self.swapper = swapper
        self.transformer = transformer

    def __call__(self,
                 image_path: str,
                 mask_path: str,
                 templates_paths: List[str],
                 out_dir: str):
        out_dir = os.path.join(out_dir, 'fake')
        os.makedirs(out_dir, exist_ok=True)
        fakes_collection = []
        image = utils.imageReader(image_path)
        mask = utils.imageReader(mask_path)
        for temp_pth in templates_paths:
            template = utils.imageReader(temp_pth,
                                         reader=cv2.imread,
                                         reader_params=(cv2.IMREAD_UNCHANGED,))
            temp_ext_len = len(os.path.basename(temp_pth).split('.')[-1]) + 1
            base_name = os.path.basename(temp_pth)[:-temp_ext_len]
            fakes_collection.extend(self.patchExtractor(image, mask, template,
                                                       out_dir, base_name))
        print(f'{len(fakes_collection)} patches have been extracted.')
        return fakes_collection

    def patchExtractor(self,
                       image: np.ndarray,
                       mask: np.ndarray,
                       template: np.ndarray,
                       out_dir: str,
                       base_name: str):
        collection = []
        k = 0
        while k < self.num_fakes_per_real:
            patch = FakeObjectExtractor.getNewPatch(image.copy(),
                                                    mask.copy(),
                                                    template.copy())
            if patch is None:
                continue
            patch_img, patch_msk = patch
            if self.swapper is not None:
                patch_img = self.swapper(patch_img)
            if self.transformer is not None:
                augmented = self.transformer(image=patch_img, mask=patch_msk)
                patch_img, patch_msk = augmented['image'], augmented['mask']
            transparent_patch = np.concatenate(
                                    (patch_img.astype(np.uint8),
                                     patch_msk[..., np.newaxis].astype(np.uint8)),
                                    axis=-1)
            transparent_patch = Image.fromarray(transparent_patch)
            out_path = os.path.join(out_dir,
                                    f'{base_name}_fkobj{k:0>4}{self.seg_ext}')
            transparent_patch.save(out_path)
            collection.append(out_path)
            k += 1
        return collection

    @staticmethod
    def getNewPatch(image: np.ndarray,
                    mask: np.ndarray,
                    template: np.ndarray):
        mask[mask > 0] = 1
        assert template.shape[-1] == 4, 'Template must be a 4 channels object.'
        template = template[:, :, 3]
        template[template > 0] = 1

        shape = image.shape
        rows, columns = np.where(template == 1)
        row_lower, row_upper = np.min(rows), np.max(rows)
        col_lower, col_upper = np.min(columns), np.max(columns)
        # Relocate the object.
        height = row_upper - row_lower
        width = col_upper - col_lower
        counter = 0
        loc_found = False
        while counter < 200:
            rnd_row = random.randint(0, shape[0] - height)
            rnd_col = random.randint(0, shape[1] - width)
            iou = utils.iouCalculator(
                mask=mask[rnd_row:rnd_row + height, rnd_col:rnd_col + width],
                patch=template[row_lower:row_upper, col_lower:col_upper])
            if iou < 0.1:
                loc_found = True
                break
            counter += 1
        if loc_found == False:
            return None
        # Select and return the new object and its mask.
        patch_msk = template[row_lower:row_upper, col_lower:col_upper]
        patch_img = image[rnd_row:rnd_row + height, rnd_col:rnd_col + width]
        patch_msk[patch_msk > 0] = 255
        patch_img[patch_msk != 255] = 0
        return patch_img, patch_msk


if __name__ == '__main__':
    # Define input arguments.
    parser = argparse.ArgumentParser(description='Segment Extractor Params.')
    parser.add_argument('-c', '--config', dest='config_path', type=str,
                        help='The path of the `Json` or `Yaml` config file.')
    args = parser.parse_args()
    configs = utils.configLoader(args.config_path)

    # Define Color Swappers.
    if configs['swap_color'] is not None:
        swapper = utils.ColorSwapper(
            source_color=configs['swap_color']['source'],
            destin_color=configs['swap_color']['destin'])
    else:
        swapper = None

    # Define transformations.
    transformations = utils.SpatialTRFMS(
        image_size=(configs['transformer']['height'],
                    configs['transformer']['width']),
        angle=configs['transformer']['rotation_angle'],
        p=configs['transformer']['rotation_p'],
        elast_p=configs['transformer']['elast_p'])
    transformer = transformations[configs['transformer']['trfm_index']]

    # Define Extractors.
    real_extractor = RealObjectExtractor(segment_extension=configs['segment_extension'],
                                         min_object_size=configs['min_object_size_in_pixel'],
                                         swapper=swapper,
                                         transformer=transformer)
    fake_extractor = FakeObjectExtractor(segment_extension=configs['segment_extension'],
                                         num_fakes_per_real=configs['num_fakes_per_real'],
                                         swapper=swapper,
                                         transformer=transformer)
    metadata = pd.read_csv(configs['segmented_images_meta_csv_path'])
    real_collection = []
    fake_collection = []
    for i, row in metadata.iterrows():
        collection = real_extractor(image_path=row['Image'],
                                    mask_path=row['Mask'],
                                    out_dir=row['OutDir'])
        real_collection.extend(collection)
        collection = fake_extractor(image_path=row['Image'],
                                    mask_path=row['Mask'],
                                    templates_paths=collection,
                                    out_dir=row['OutDir'])
        fake_collection.extend(collection)
    # Save metadata.
    real_dataset = pd.DataFrame(real_collection, columns=['Image'])
    fake_dataset = pd.DataFrame(fake_collection, columns=['Image'])
    real_dataset.to_csv(configs['real_metadata_path'], header=True, index=False)
    fake_dataset.to_csv(configs['fake_metadata_path'], header=True, index=False)
    print('Extracted real objects metadata has been saved into {}.'.format(
        configs['real_metadata_path']))
    print('Extracted fake objects metadata has been saved into {}.'.format(
        configs['fake_metadata_path']))
