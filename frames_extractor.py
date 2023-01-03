"""
Extract video frames from list of videos by a user-defined interval.
Extract all the objects are exists in a list of segmented images.
"""
import os
import cv2
import argparse
import random
import numpy as np
import pandas as pd
import SimpleITK as sitk
from typing import Callable, List, Tuple, Union
from skimage import io

import utils

# Define Global Variables. 
MINDIM = 1024  # The minimum possible size for an extracted frame. 
SEED = 123
np.random.seed(SEED)
random.seed(SEED)


class FramesExtractor(object):
    """Extract frames from a video.
    Args:
        frame_extension (str): The image format of the extracted frames.
            Default value is '.tiff'.
        swapper (Callable): A callable Function used to change the color of the
            extracted frame. Default value is `None`,
        transformers (Callable): A composed list of augmentation transformations.
            Default to `None`.
    """
    def __init__(self,
                 frame_extension: str= '.tiff',
                 swapper: Union[Callable, None]=None,
                 transformer: Union[Callable, None]=None
    ) -> None:
        # Define attributes.
        self.frame_extension = frame_extension
        self.swapper = swapper
        self.transformer = transformer
        self.message = '{0}: {1} out of {2} have been created.'

    def __call__(self,
                 video_path: str,
                 frames_dir: str,
                 interval: int) -> List:
        """Process videos one by one to extract their frames."""
        print(f'Processing: {video_path}.')
        prefix = os.path.basename(video_path)[:-4]
        # Create a folder for the video frames if it has not been created yet.
        os.makedirs(frames_dir, exist_ok=True)
        collection, total_frames = self.extract(video_path, frames_dir,
                                                prefix, interval)
        print(self.message.format(prefix, len(collection), total_frames))
        return collection

    def extract(self,
                video_path: str,
                back_dir: str,
                prefix: str,
                interval: int
    ) -> Tuple[List, int]:
        """ Get a video clip and extract and save its frames with the predefined
            interval.
        Args:
            video_path (str): The path of a video to be processed.
            back_dir (str): A directory address to save extracted frames.
            prefix (str): A string indicates the first part of the names for the
                extracted frames.
            interval (int): An integer number indicated the extraction step length
                from the video.
        Returns:
            collection (int): The list of paths of generated and saved frames.
            total_num_frames (int): The number of total frames within the video.
        """
        cam = cv2.VideoCapture(video_path)
        total_num_frames = 0
        saved_frame_count = 0
        collection = []
        while True:
            ret, frame = cam.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if total_num_frames % interval == 0:
                    name = os.path.join(
                        back_dir,
                        f'{prefix}_{saved_frame_count:0>6}' + self.frame_extension
                    )
                    if self.swapper is not None:
                        frame = self.swapper(frame)
                    if frame.shape[0] < MINDIM or frame.shape[1] < MINDIM:
                        dims = (max(MINDIM, frame.shape[0]), 
                                max(MINDIM, frame.shape[1]))
                        frame = cv2.resize(src=frame, 
                                           dsize=dims, 
                                           interpolation=cv2.INTER_CUBIC)
                    if self.transformer is not None:
                        frame = self.transformer(image=frame)['image']
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(name, frame)
                    # Add generated frame path to the output paths.
                    collection = FramesExtractor.collector(collection, name)
                    saved_frame_count += 1
                total_num_frames += 1
            else:
                break
        cam.release()
        cv2.destroyAllWindows()
        return collection, total_num_frames

    @staticmethod
    def collector(collection: List,
                  item: str
    ) -> List:
        """Append a new extracted frame to the collection.
        Args:
            collection (Sequence): A list of previously collected frames for all
                the previously processed videos.
            item (str): The path of the new extracted frame.
        Returns:
            collection (Sequence): The updated sequence.
        """
        collection.append(item)
        return collection


if __name__ == '__main__':
    # Define input arguments.
    parser = argparse.ArgumentParser(description='Frames Extractor Params.')
    parser.add_argument('-c', '--config', dest='config_path', type=str,
                        help='The path of the `Json` or `Yaml` config file.')
    args = parser.parse_args()
    configs = utils.configLoader(args.config_path)

    # Define Color Swappers.
    if configs['swap_color'] is not None:
        swapper = utils.ColorSwapper(source_color=configs['swap_color']['source'],
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

    # Define Extractor.
    extractor = FramesExtractor(frame_extension=configs['frame_extension'],
                                swapper=swapper,
                                transformer=transformer)
    metadata = pd.read_csv(configs['videos_meta_csv_path'])
    dataset_collection = []
    for i, row in metadata.iterrows():
        collection = extractor(video_path=row['Video'],
                               frames_dir=row['FramesDir'],
                               interval=row['Interval'])
        dataset_collection.extend(collection)
    dataset = pd.DataFrame(dataset_collection, columns=['Image'])
    dataset.to_csv(configs['output_metadata_path'], header=True, index=False)
    print('Extracted frames metadata has been saved into {}.'.format(
                                            configs['output_metadata_path']))
