"""The model used to detect figures."""

import copy
import os
import tempfile
from typing import List, Tuple, Iterable

import numpy as np
import tensorflow as tf

from deepfigures import settings
from deepfigures.extraction.datamodels import (
    BoxClass,
    Figure,
    PdfDetectionResult,
    CaptionOnly)
from deepfigures.extraction import (
    figure_utils,
    pdffigures_wrapper,
    renderers)
from deepfigures.extraction.pdffigures_wrapper import pdffigures_extractor
from deepfigures.utils import (
    file_util,
    image_util,
    config,
    traits,
    settings_utils)
from tensorboxresnet import train
from tensorboxresnet.utils import train_utils


CAPTION_CHANNEL_BACKGROUND = 255
CAPTION_CHANNEL_MASK = 0

pdf_renderer = settings_utils.import_setting(
    settings.DEEPFIGURES_PDF_RENDERER)()


class TensorboxCaptionmaskDetector(object):
    """Interface for using the neural network model to detect figures.

    Instantiating this class creates a tensorflow session object as the
    self.sess attribute. When done using the instance, remember to close
    the session; however, do not open and close sessions every time you
    extract a figure because the added overhead will very negatively
    affect performance.
    """
    def __init__(
            self,
            save_dir,
            iteration,
            batch_size=1  # Batch sizes greater than 1 will change results due to batch norm in inception_v1
    ):
        self.save_dir = save_dir
        self.iteration = iteration

        self.hypes = self._get_hypes()
        self.hypes['batch_size'] = batch_size
        self.input_shape = [
            self.hypes['image_height'], self.hypes['image_width'],
            self.hypes['image_channels']
        ]  # type: Tuple[float, float, float]
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x_in = tf.placeholder(
                tf.float32, name='x_in', shape=self.input_shape
            )
            assert (self.hypes['use_rezoom'])
            pred_boxes, self.pred_logits, self.pred_confidences, self.pred_confs_deltas, pred_boxes_deltas = \
                train.build_forward(self.hypes, tf.expand_dims(self.x_in, 0), 'test', reuse=None)
            self.pred_boxes = pred_boxes + pred_boxes_deltas
            grid_area = self.hypes['grid_height'] * self.hypes['grid_width']
            pred_confidences = tf.reshape(
                tf.nn.softmax(
                    tf.reshape(
                        self.pred_confs_deltas,
                        [grid_area * self.hypes['rnn_len'], 2]
                    )
                ), [grid_area, self.hypes['rnn_len'], 2]
            )
            assert (self.hypes['reregress'])
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
        model_weights = self._get_weights()
        saver.restore(self.sess, model_weights)

    def _get_weights(self) -> str:
        suffixes = ['.index', '.meta', '.data-00000-of-00001']
        local_paths = [
            file_util.cache_file(
                self.save_dir + 'save.ckpt-%d' % self.iteration + suffix
            ) for suffix in suffixes
        ]
        local_path = local_paths[0]
        return local_path[:local_path.rfind(suffixes[0])]

    def _get_hypes(self) -> dict:
        return file_util.read_json(self.save_dir + 'hypes.json')

    def detect_page(
            self,
            page_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        feed = {self.x_in: page_tensor}
        (np_pred_boxes, np_pred_confidences) = self.sess.run(
            [self.pred_boxes, self.pred_confidences],
            feed_dict=feed)
        return (np_pred_boxes, np_pred_confidences)

    def get_detections(
            self,
            page_images: List[np.ndarray],
            crop_whitespace: bool = True,
            conf_threshold: float = .5) -> List[List[BoxClass]]:
        page_datas = [
            {
                'page_image': page_image,
                'orig_size': page_image.shape[:2],
                'resized_page_image': image_util.imresize_multichannel(
                    page_image, self.input_shape),
            }
            for page_image in page_images
        ]

        predictions = [
            self.detect_page(page_data['resized_page_image'])
            for page_data in page_datas
        ]

        for (page_data, prediction) in zip(page_datas, predictions):
            (np_pred_boxes, np_pred_confidences) = prediction
            new_img, rects = train_utils.add_rectangles(
                self.hypes,
                page_data['resized_page_image'],
                np_pred_confidences,
                np_pred_boxes,
                use_stitching=True,
                min_conf=conf_threshold,
                show_suppressed=False)
            detected_boxes = [
                BoxClass(x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2).resize_by_page(
                    self.input_shape, page_data['orig_size'])
                for r in rects if r.score > conf_threshold
            ]
            if crop_whitespace:
                detected_boxes = [
                    box.crop_whitespace_edges(page_data['page_image'])
                    for box in detected_boxes
                ]
                detected_boxes = list(filter(None, detected_boxes))
            page_data['detected_boxes'] = detected_boxes
        return [page_data['detected_boxes'] for page_data in page_datas]


def detect_figures(
    pdf: str,
    pdffigures_captions: List[CaptionOnly],
    detector: TensorboxCaptionmaskDetector,
    conf_threshold: float
) -> Tuple[List[Figure], List[List[BoxClass]]]:
    page_image_files = pdf_renderer.render(pdf, dpi=settings.DEFAULT_INFERENCE_DPI)
    page_tensors = []
    for f in page_image_files:
        page_im = image_util.read_tensor(f)
        if detector.hypes['image_channels'] == 3:
            page_tensors.append(page_im)
        else:
            im_with_mask = np.pad(
                page_im,
                pad_width=[(0, 0), (0, 0), (0, 1)],
                mode='constant',
                constant_values=CAPTION_CHANNEL_BACKGROUND
            )
            for caption in pdffigures_captions:
                (x1, y1, x2, y2) = caption.caption_boundary.get_rounded()
                im_with_mask[y1:y2, x1:x2, 3] = CAPTION_CHANNEL_MASK
            page_tensors.append(im_with_mask)
    figure_boxes_by_page = detector.get_detections(
        page_tensors, conf_threshold=conf_threshold
    )
    figures_by_page = []
    for page_num in range(len(page_image_files)):
        # Page numbers are always 0 indexed
        figure_boxes = figure_boxes_by_page[page_num]
        pf_page_captions = [
            cap for cap in pdffigures_captions if cap.page == page_num
        ]
        caption_boxes = [cap.caption_boundary for cap in pf_page_captions]
        (figure_indices, caption_indices) = figure_utils.pair_boxes(
            figure_boxes, caption_boxes
        )
        figures_by_page.extend(
            [
                Figure(
                    figure_boundary=figure_boxes[figure_idx],
                    caption_boundary=caption_boxes[caption_idx],
                    caption_text=pf_page_captions[caption_idx].caption_text,
                    name=pf_page_captions[caption_idx].name,
                    figure_type=pf_page_captions[caption_idx].figure_type,
                    page=page_num,
                )
                for (figure_idx,
                     caption_idx) in zip(figure_indices, caption_indices)
            ]
        )
    return figures_by_page, figure_boxes_by_page


def detect_batch(
        src_pdfs: List[str],
        detector: TensorboxCaptionmaskDetector,
        conf_threshold: float=.5) -> Iterable[PdfDetectionResult]:
    for src_pdf in src_pdfs:
        with tempfile.TemporaryDirectory(
                prefix='deepfigures-tensorbox') as working_dir:
            pdf_path = os.path.join(
                working_dir,
                src_pdf.replace('/', '_'))
            file_util.copy(src_pdf, pdf_path)
            pdffigures_output = pdffigures_extractor.extract(
                pdf_path,
                working_dir)
            pdffigures_captions = pdffigures_wrapper.get_captions(
                pdffigures_output)
            figures_by_page, figure_boxes_by_page = detect_figures(
                pdf_path,
                pdffigures_captions,
                detector,
                conf_threshold=conf_threshold)
            yield PdfDetectionResult(
                pdf=src_pdf,
                figures=figures_by_page,
                dpi=settings.DEFAULT_INFERENCE_DPI,
                raw_detected_boxes=figure_boxes_by_page,
                raw_pdffigures_output=pdffigures_output)
