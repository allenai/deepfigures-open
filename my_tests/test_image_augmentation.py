import unittest
from deepfigures.data_generation.arxiv_pipeline import augment_images
from deepfigures.extraction.datamodels import Figure, BoxClass

import numpy as np
import imageio
import shutil
import os


class TestAugmentation(unittest.TestCase):
    def test_basigAugmentation(self):
        test_original = 'resources/test_original.png'
        test = 'resources/test.png'
        if os.path.exists(test):
            os.remove(test)
        shutil.copyfile(test_original, test)
        path = test
        original_image = imageio.imread(path)

        figure = Figure()
        figure.figure_boundary = BoxClass(x1=250, x2=400, y1=250, y2=400)
        augment_images(path, [figure])

        augmented_image = imageio.imread(path)

        if np.array_equal(original_image, augmented_image):
            print('Failed augmentation test.')
        else:
            print("Success!")

        if os.path.exists(test):
            os.remove(test)
