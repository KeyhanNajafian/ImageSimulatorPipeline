"""Visualize pairs of images/masks by hard coding the directory addresses."""
import os
import glob
from utils import Visualizer

images = glob.glob('data/simulation/weak/yellow/images/valid/*.jpg')
masks = glob.glob('data/simulation/weak/yellow/masks/valid/*.png')

images = sorted(images)
masks = sorted(masks)
print(len(images), len(masks))

visualize = Visualizer(images, masks)
visualize()
