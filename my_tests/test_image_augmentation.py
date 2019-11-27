import unittest
from deepfigures.data_generation.arxiv_pipeline import augment_images
from deepfigures.extraction.datamodels import Figure, BoxClass

import numpy as np
import imageio


class TestAugmentation(unittest.TestCase):
    def test_basigAugmentation(self):
        path = 'resources/test.png'
        original_image = imageio.imread(path)

        figure = Figure()
        figure.figure_boundary = BoxClass(x1=0, x2=0, y1=0, y2=0)
        augment_images(path, [figure])

        augmented_image = imageio.imread(path)

        if np.array_equal(original_image, augmented_image):
            print('Failed augmentation test.')
        else:
            print("Success!")
