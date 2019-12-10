"""Functions for detecting and extracting figures."""

import os

from typing import List, Tuple, Iterable

import cv2  # Need to import OpenCV before tensorflow to avoid import error
from scipy.misc import imread, imsave
#from scipy.misc.pilutil import imread, imsave
import numpy as np

from deepfigures.extraction import (
    tensorbox_fourchannel,
    pdffigures_wrapper,
    figure_utils)
from deepfigures import settings
from deepfigures.extraction.datamodels import (
    BoxClass,
    Figure,
    PdfDetectionResult,
    CaptionOnly)
from deepfigures import settings
from deepfigures.utils import (
    file_util,
    settings_utils)
from deepfigures.utils import misc


PAD_FACTOR = 0.02
TENSORBOX_MODEL = settings.TENSORBOX_MODEL


# Holds a cached instantiation of TensorboxCaptionmaskDetector.
_detector = None


def get_detector() -> tensorbox_fourchannel.TensorboxCaptionmaskDetector:
    """
    Get TensorboxCaptionmaskDetector instance, initializing it on the first call.
    """
    global _detector
    if not _detector:
        _detector = tensorbox_fourchannel.TensorboxCaptionmaskDetector(
            **TENSORBOX_MODEL)
    return _detector


def extract_figures_json(
        pdf_path,
        page_image_paths,
        pdffigures_output,
        output_directory):
    """Extract information about figures to JSON and save to disk.

    :param str pdf_path: path to the PDF from which to extract
      figures.

    :returns: path to the JSON file containing the detection results.
    """
    page_images_array = np.array([
        imread(page_image_path)
        for page_image_path in page_image_paths
    ])
    detector = get_detector()
    figure_boxes_by_page = detector.get_detections(
        page_images_array)
    pdffigures_captions = pdffigures_wrapper.get_captions(
        pdffigures_output=pdffigures_output,
        target_dpi=settings.DEFAULT_INFERENCE_DPI)
    figures_by_page = []
    for page_num in range(len(page_image_paths)):
        figure_boxes = figure_boxes_by_page[page_num]
        pf_page_captions = [
            caption
            for caption in pdffigures_captions
            if caption.page == page_num
        ]
        caption_boxes = [
            caption.caption_boundary
            for caption in pf_page_captions
        ]
        figure_indices, caption_indices = figure_utils.pair_boxes(
            figure_boxes, caption_boxes)
        page_image = page_images_array[page_num]
        pad_pixels = PAD_FACTOR * min(page_image.shape[:2])
        for (figure_idx, caption_idx) in zip(figure_indices, caption_indices):
            figures_by_page.append(
                Figure(
                    figure_boundary=figure_boxes[figure_idx].expand_box(
                        pad_pixels).crop_to_page(
                        page_image.shape).crop_whitespace_edges(
                            page_image),
                    caption_boundary=caption_boxes[caption_idx],
                    caption_text=pf_page_captions[caption_idx].caption_text,
                    name=pf_page_captions[caption_idx].name,
                    figure_type=pf_page_captions[caption_idx].figure_type,
                    page=page_num))
    pdf_detection_result = PdfDetectionResult(
        pdf=pdf_path,
        figures=figures_by_page,
        dpi=settings.DEFAULT_INFERENCE_DPI,
        raw_detected_boxes=figure_boxes_by_page,
        raw_pdffigures_output=pdffigures_output)

    output_path = os.path.join(
        output_directory,
        os.path.basename(pdf_path)[:-4] + 'deepfigures-results.json')
    file_util.write_json_atomic(
        output_path,
        pdf_detection_result.to_dict(),
        indent=2,
        sort_keys=True)
    return output_path
