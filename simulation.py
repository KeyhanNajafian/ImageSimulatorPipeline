"""Simulate the Weak dataset by overlaying wheat head objects on the
background images.
"""
import os
import cv2
import random
import argparse
import numpy as np
import pandas as pd
from skimage import io 
from typing import Callable, Tuple, List, Union

import utils
import data

random.seed(246)
np.random.seed(246)


class Simulation:
    """Simulate image/mask pair samples.
    Args:
        back_loader (Callable): Background image loader.
        fore_loader (Callable): Real object loader.
        fake_loader (Callable): Fake object loader.
        num_objs_per_image (Tuple): An integer range from which randomly
            choose the number of real objects to be overlaid on the background.
        num_ptchs_per_image (Tuple): An integer range from which randomly
            choose the number of fake patches to be overlaid on the background.
        num_loc_finder_try (int): Number of times to try to find a location
            with the best overlap with other existence objects. The default value
            is `10`.
        iou_thred (float): Intersection over Union threshold for finding
            overlaps. Default value is `0.5`.
        size (int): Dataset size. Default value is `1000`.
        color_balancer (Callable): Balance wheat head objects with corresponding
            background image using this transformer. Default value is `None`.
        transformer (Callable): A combination of transformations. 
    """
    def __init__(self,
                 back_loader: Callable,
                 fore_loader: Callable,
                 fake_loader: Callable,
                 num_objs_per_image: Union[Tuple, List],
                 num_ptchs_per_image: Union[Tuple, List],
                 num_loc_finder_try: 10,
                 iou_thred: float=0.5,
                 dataset_size: int=1000,
                 color_balancer: Union[Callable, None]=None, 
                 transformer: Union[Callable, None]=None):
        # Define class attributes.
        self.back_loader = back_loader
        self.fore_loader = fore_loader
        self.fake_loader = fake_loader
        assert len(num_objs_per_image) == 2, 'Just lower bound and upper ' \
                                             'bound are needed.'
        self.num_objs_per_image = num_objs_per_image
        assert len(num_ptchs_per_image) == 2, 'Just lower bound and upper ' \
                                              'bound are needed.'
        self.num_ptchs_per_image = num_ptchs_per_image
        self.num_loc_finder_try = num_loc_finder_try
        self.iou_thred = iou_thred
        self.dataset_size = dataset_size
        self.color_balancer = color_balancer
        self.transformer = transformer
        self.random_indexes = np.random.uniform(low=0,
                                                high=len(self.back_loader),
                                                size=self.dataset_size).astype(np.int64)

    def __call__(self, item):
        rnd_idx = self.random_indexes[item]
        background = self.back_loader(rnd_idx)
        batch_size = random.randint(*self.num_objs_per_image)
        real_batches = self.fore_loader(size=batch_size)
        patch_size = random.randint(*self.num_ptchs_per_image)
        fake_batches = self.fake_loader(size=patch_size)

        # Apply fake foreground patches to the background image.
        for ptch_img, obj_msk in fake_batches:
            x, y = Simulation.randLocationGenerator(background.shape,
                                                    obj_msk.shape)
            background, _ = Simulation.apply(x, y, ptch_img, obj_msk, background)

        # Apply real foreground patches to the background image.
        background_mask = np.zeros(background.shape[:2], dtype=np.uint8)
        bboxes = []
        for obj_img, obj_msk in real_batches:
            # Balance the foreground color using background image.
            if self.color_balancer is not None:
                obj_img = self.color_balancer(background, obj_img, obj_msk)
            # Find the location for overlaying the new foreground image.
            x, y = Simulation.randLocationFinder(background_mask, obj_msk,
                                                 num_try=self.num_loc_finder_try,
                                                 iou_thred=self.iou_thred)
            background, background_mask = Simulation.apply(x, y,
                                                           obj_img, obj_msk,
                                                           background,
                                                           background_mask)
            bboxes.append([x, y, obj_msk.shape[1], obj_msk.shape[0]])
        background = self.transformer(image=background)['image']
        return background, background_mask, bboxes

    @staticmethod
    def randLocationGenerator(background_shape: Tuple,
                              mask_shape: Tuple):
        """Randomly find a location for a new patch to be overlaid on the
            image.
        Args:
            background_shape (Tuple): A series of two  integer numbers showing
                the background image height and width.
            mask_shape (Tuple): A series of two integer numbers showing the
                shape of the patch.
        Returns:
            x, y: the coordinate of new patch on the background image.
        """
        x = random.randint(0, abs(background_shape[1] - mask_shape[1] - 1))
        y = random.randint(0, abs(background_shape[0] - mask_shape[0] - 1))
        return x, y

    @staticmethod
    def randLocationFinder(back_contour: np.ndarray,
                           mask: np.ndarray,
                           num_try: int=10,
                           iou_thred: float=0.5):
        """Randomly find a location for a new object to be overlaid on the
            image in a way that has the overlap of less than iou threshold
            with the previously applied wheat head objects.
        Args:
            back_contour (np.ndarray): The contour image realted to the
                overlaid background.
            mask (np.ndarray): The wheat head object mask.
            num_try (int): Number of tries to find a better location.
            iou_thred (int): The intersection over Union threshold.
        Returns:
            x, y: the coordinate for the new wheat head object on the
                background image.
        """
        back_h, back_w = back_contour.shape
        mask_h, mask_w = mask.shape
        # Start Searching for an appropriate location with less than overlap
            # with other objects.
        x = random.randint(0, abs(back_w - mask_w - 1))
        y = random.randint(0, abs(back_h - mask_h - 1))
        repeat = 0
        while repeat <= num_try:
            # See if the overlap of the found location with other currently
                # applied objects is less than the threshold.
            if utils.iouCalculator(
                    mask=mask,
                    patch=back_contour[y: y + mask_h, x: x + mask_w]) >= iou_thred:
                break
            else:
                x = random.randint(0, abs(back_w - mask_w - 1))
                y = random.randint(0, abs(back_h - mask_h - 1))
                repeat += 1
        return x, y

    @staticmethod
    def apply(x: int,
              y: int,
              patch: np.ndarray,
              mask: np.ndarray,
              background: np.ndarray,
              back_contour: np.ndarray=None):
        """Apply patch and its mask to the background image and its contour at
            (y, x) location of the background image.
        Args:
            x (int): The location of the new object on the x-axis on the image.
            y (int): The location of the new object on the y-axis on the image.
            patch (np.ndarray): A fake or reak wheat head object.
            mask (np.ndarray): The mask related to the patch.
            background (np.ndarray): The background image.
            back_contour (np.ndarray): The mask contour related to the
                background image.
        Returns:
            background (np.ndarray): The updated background with the new patch.
            back_contour (np.ndarray): The updated contour with the new real
                object.
        """
        # Get shapes.
        mask_h, mask_w = mask.shape
        # Multiply the background with ( 1 - alpha)
        background[y:y + mask_h, x:x + mask_w] = cv2.multiply(
                                            np.stack((1 - mask,)*3, axis=2),
                                            background[y:y + mask_h, x:x + mask_w]
        )
        # Add the masked foreground and background.
        background[y:y + mask_h, x:x + mask_w] = cv2.add(
                                            patch,
                                            background[y:y + mask_h, x:x + mask_w])
        # if the provided patch is not a fake head object.
        if back_contour is not None:
            # Multiply the back_contour with ( 1 - alpha )
            back_contour[y:y + mask_h, x:x + mask_w] = cv2.multiply(
                                            1 - mask,
                                            back_contour[y:y + mask_h, x:x + mask_w])
            # Add the mask patch and contour.
            back_contour[y:y + mask_h, x:x + mask_w] = cv2.add(
                                            mask,
                                            back_contour[y:y + mask_h, x:x + mask_w])
        return background, back_contour


def formatBoxes(bboxes, shape):
    """Format boxes as YOLOV5 input format."""
    formatd_boxes = []
    image_height, image_width, _ = shape
    for box in bboxes:
        box[0] = (box[0] + box[2] / 2.0) / image_width
        box[1] = (box[1] + box[3] / 2.0) / image_height
        box[2] = box[2] / image_width
        box[3] = box[3] / image_height
        box.insert(0, 1)
        formatd_boxes.append(box)
    return formatd_boxes

if __name__ == '__main__':
    TRANSFORMATION_INDEX = 9
    parser = argparse.ArgumentParser(description='Generator Params.')
    parser.add_argument('-c', '--config', dest='config_path', type=str,
                        help='The path to a CSV file.')

    args = parser.parse_args()
    configs = utils.configLoader(args.config_path)
    spatial_transformation = utils.SpatialTRFMS(
        image_size=(configs['image_heigth'],
                    configs['image_width']),
        angle=10, interpolation=1,
        border_mode=cv2.BORDER_REFLECT_101,
        value=0, alpha=0.1, sigma=0.5,
        alpha_affine=1.0, p=0.5,
        elast_p=0.25
    )
    color_transformation = utils.Transformation(image_size=(configs['image_heigth'], 
                                                            configs['image_width'])
                                               )[configs['img_trfm_idx']]
    # Define Loaders.
    back_loader = data.BackgroundDataset(
        metadata_path=configs['background_meta'],
        transformer=None)
    fore_loader = data.RealObjectDataset(
        metadata_path=configs['real_object_meta'],
        transformer=spatial_transformation[configs['real_trfm_idx']])
    fake_loader = data.FakePatchDataset(
        metadata_path=configs['fake_patch_meta'],
        transformer=spatial_transformation[configs['fake_trfm_idx']])

    # Define Generator.
    generator = Simulation(
        back_loader=back_loader,
        fore_loader=fore_loader,
        fake_loader=fake_loader,
        num_objs_per_image=(configs['min_num_objects'],
                            configs['max_num_objects']),
        num_ptchs_per_image=(configs['min_num_patches'],
                             configs['max_num_patches']),
        num_loc_finder_try=configs['num_loc_finder_try'],
        iou_thred=configs['iou_threshold'],
        dataset_size=configs['dataset_size'],
        color_balancer=None,
        transformer=color_transformation
    )
    os.makedirs(os.path.join(configs['out_dir'], 'images', configs['phase'],),
                exist_ok=True)
    os.makedirs(os.path.join(configs['out_dir'], 'masks', configs['phase'],),
                exist_ok=True)
    os.makedirs(os.path.join(configs['out_dir'], 'labels', configs['phase'],),
                exist_ok=True)
    # Check the plugin.
    local_plugin = 'tifffile' if configs['image_extension'] == '.tiff' else None

    # Run generator.
    collection = {
        'Image': [],
        'Mask': [],
        'BBoxes': []
    }
    for img_counter in range(configs['dataset_size']):
        # Simulate a new sample and save it.
        print(f'Process: Create sample {img_counter}')
        image, mask, bboxes = generator(item=img_counter)
        bboxes = formatBoxes(bboxes=bboxes, shape=image.shape)
        img_pth = os.path.join(configs['out_dir'], 'images', configs['phase'],
                               f'{img_counter:0>6}{configs["image_extension"]}')
        msk_pth = os.path.join(configs['out_dir'], 'masks', configs['phase'],
                               f'{img_counter:0>6}.png')
        bbx_pth = os.path.join(configs['out_dir'], 'labels', configs['phase'],
                               f'{img_counter:0>6}.txt')
        io.imsave(img_pth, image, plugin=local_plugin, check_contrast=False)
        io.imsave(msk_pth, mask, check_contrast=False)
        with open(bbx_pth, 'w') as writer:
            for box in bboxes:
                writer.write(' '.join([str(b) for b in box]))
                writer.write('\n')
        collection['Image'].append(img_pth)
        collection['Mask'].append(msk_pth)
        collection['BBoxes'].append(bbx_pth)

    df = pd.DataFrame(collection)
    df.to_csv(configs['out_dataset_path'],
              header=True,
              index=False)
