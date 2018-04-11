"""Utilities for tests with deepfigures."""

import json
import logging


logger = logging.getLogger(__name__)


def test_deepfigures_json(
        self,
        expected_json,
        actual_json):
    """Run tests comparing two deepfigures JSON files.

    Compare two json files outputted from deepfigures and verify that
    they are sufficiently similar, this includes comparing the general
    structure of the files as well as specific values like the figure
    captions, intersection over union for the bounding boxes, etc.

    :param unittest.TestCase self: the TestCase to use for running the
      comparsions.
    :param str expected_json: a file path string to the
      expected / baseline deepfigures JSON on disk.
    :param str actual_json: a file path string to the
      actual / to be tested deepfigures JSON on disk.

    :returns: None
    """
    with open(expected_json, 'r') as expected_file:
        expected = json.load(expected_file)
    with open(actual_json, 'r') as actual_file:
        actual = json.load(actual_file)

    # make sure keys are the same
    self.assertEqual(
        expected.keys(),
        actual.keys())

    # compare top level attributes
    self.assertEqual(
        expected['dpi'],
        actual['dpi'])
    self.assertEqual(
        expected['error'],
        actual['error'])
    self.assertEqual(
        len(expected['figures']),
        len(actual['figures']))

    # compare generated figures
    for expected_figure, actual_figure in zip(
            expected['figures'],
            actual['figures']):
        exact_match_attrs = [
            'caption_text',
            'dpi',
            'figure_type',
            'name',
            'page',
            'page_height',
            'page_width'
        ]
        for attr in exact_match_attrs:
            self.assertEqual(
                expected_figure[attr],
                actual_figure[attr])
        bounding_box_attrs = [
            'caption_boundary',
            'figure_boundary'
        ]
        for attr in bounding_box_attrs:
            intersection = {
                'x1': max(expected_figure[attr]['x1'], actual_figure[attr]['x1']),
                'x2': min(expected_figure[attr]['x2'], actual_figure[attr]['x2']),
                'y1': max(expected_figure[attr]['y1'], actual_figure[attr]['y1']),
                'y2': min(expected_figure[attr]['y2'], actual_figure[attr]['y2'])
            }
            # check that the boxes actually do overlap
            self.assertLess(
                intersection['x1'],
                intersection['x2'],
                msg="expected and actual box for {attr} in {figname}"
                    "don't overlap".format(attr=attr, figname=expected_figure['name']))
            self.assertLess(
                intersection['y1'],
                intersection['y2'],
                msg="expected and actual box for {attr} in {figname}"
                    "don't overlap".format(attr=attr, figname=expected_figure['name']))
            union = {
                'x1': min(expected_figure[attr]['x1'], actual_figure[attr]['x1']),
                'x2': max(expected_figure[attr]['x2'], actual_figure[attr]['x2']),
                'y1': min(expected_figure[attr]['y1'], actual_figure[attr]['y1']),
                'y2': max(expected_figure[attr]['y2'], actual_figure[attr]['y2'])
            }
            i_area = (
                (intersection['x2'] - intersection['x1']) *
                (intersection['y2'] - intersection['y1'])
            )
            u_area = (
                (union['x2'] - union['x1']) *
                (union['y2'] - union['y1'])
            )
            iou = i_area / u_area
            self.assertGreater(
                iou,
                0.8,
                msg="intersection over union for {attr} on {figname} has"
                    "dropped below acceptable thresholds.".format(
                        attr=attr,
                        figname=expected_figure['name']))
