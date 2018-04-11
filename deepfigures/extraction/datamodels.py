"""Data models for deepfigures.

This subpackage contains models for various data dealt with by the
deepfigures package.
"""

from typing import List, Optional, Tuple, Union

from matplotlib import patches
import numpy as np

from deepfigures.utils import traits
from deepfigures.utils.config import JsonSerializable

from deepfigures.settings import (DEFAULT_INFERENCE_DPI, BACKGROUND_COLOR)

# A box of the form (x1, y1, x2, y2) in pixel coordinates
IntBox = Tuple[int, int, int, int]
ImageSize = Union[Tuple[float, float], Tuple[float, float, float]
                 ]  # Page sizes may have a third color channel


class BoxClass(JsonSerializable):
    x1 = traits.Float(allow_none=False)
    y1 = traits.Float(allow_none=False)
    x2 = traits.Float(allow_none=False)
    y2 = traits.Float(allow_none=False)

    @staticmethod
    def from_tuple(t: Tuple[float, float, float, float]) -> 'BoxClass':
        return BoxClass(x1=t[0], y1=t[1], x2=t[2], y2=t[3])

    @staticmethod
    def from_tensorbox_rect(r) -> 'BoxClass':
        return BoxClass(
            x1=r.cx - .5 * r.width,
            x2=r.cx + .5 * r.width,
            y1=r.cy - .5 * r.height,
            y2=r.cy + .5 * r.height
        )

    @staticmethod
    def from_xml(word, target_dpi=DEFAULT_INFERENCE_DPI) -> 'BoxClass':
        scale_factor = DEFAULT_INFERENCE_DPI / 72
        return BoxClass(
            x1=float(word.get('xMin')),
            y1=float(word.get('yMin')),
            x2=float(word.get('xMax')),
            y2=float(word.get('yMax'))
        ).rescale(scale_factor)

    def get_width(self) -> float:
        return self.x2 - self.x1

    def get_height(self) -> float:
        return self.y2 - self.y1

    def get_plot_box(
        self, color: str='red', fill: bool=False, **kwargs
    ) -> patches.Rectangle:
        """Return a rectangle patch for plotting"""
        return patches.Rectangle(
            (self.x1, self.y1),
            self.get_width(),
            self.get_height(),
            edgecolor=color,
            fill=fill,
            **kwargs
        )

    def get_area(self) -> float:
        width = self.get_width()
        height = self.get_height()
        if width <= 0 or height <= 0:
            return 0
        else:
            return width * height

    def rescale(self, ratio: float) -> 'BoxClass':
        return BoxClass(
            x1=self.x1 * ratio,
            y1=self.y1 * ratio,
            x2=self.x2 * ratio,
            y2=self.y2 * ratio
        )

    def resize_by_page(
        self, cur_page_size: ImageSize, target_page_size: ImageSize
    ):
        (orig_h, orig_w) = cur_page_size[:2]
        (target_h, target_w) = target_page_size[:2]
        height_scale = target_h / orig_h
        width_scale = target_w / orig_w
        return BoxClass(
            x1=self.x1 * width_scale,
            y1=self.y1 * height_scale,
            x2=self.x2 * width_scale,
            y2=self.y2 * height_scale
        )

    def get_rounded(self) -> IntBox:
        return (
            int(round(self.x1)), int(round(self.y1)), int(round(self.x2)),
            int(round(self.y2))
        )

    def crop_image(self, image: np.ndarray) -> np.ndarray:
        """Return image cropped to the portion contained in box."""
        (x1, y1, x2, y2) = self.get_rounded()
        return image[y1:y2, x1:x2]

    def crop_whitespace_edges(self, im: np.ndarray) -> Optional['BoxClass']:
        (rounded_x1, rounded_y1, rounded_x2, rounded_y2) = self.get_rounded()
        white_im = im.copy()
        white_im[:, :rounded_x1] = BACKGROUND_COLOR
        white_im[:, rounded_x2:] = BACKGROUND_COLOR
        white_im[:rounded_y1, :] = BACKGROUND_COLOR
        white_im[rounded_y2:, :] = BACKGROUND_COLOR
        is_white = (white_im == BACKGROUND_COLOR).all(axis=2)
        nonwhite_columns = np.where(is_white.all(axis=0) != 1)[0]
        nonwhite_rows = np.where(is_white.all(axis=1) != 1)[0]
        if len(nonwhite_columns) == 0 or len(nonwhite_rows) == 0:
            return None
        x1 = min(nonwhite_columns)
        x2 = max(nonwhite_columns) + 1
        y1 = min(nonwhite_rows)
        y2 = max(nonwhite_rows) + 1
        assert x1 >= rounded_x1, 'ERROR:  x1:%d box[0]:%d' % (x1, rounded_x1)
        assert y1 >= rounded_y1, 'ERROR:  y1:%d box[1]:%d' % (y1, rounded_y1)
        assert x2 <= rounded_x2, 'ERROR:  x2:%d box[2]:%d' % (x2, rounded_x2)
        assert y2 <= rounded_y2, 'ERROR:  y2:%d box[3]:%d' % (y2, rounded_y2)
        # np.where returns np.int64, cast back to python types
        return BoxClass(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))

    def distance_to_other(self, other: 'BoxClass') -> float:
        x_distance = max([0, self.x1 - other.x2, other.x1 - self.x2])
        y_distance = max([0, self.y1 - other.y2, other.y1 - self.y2])
        return np.linalg.norm([x_distance, y_distance], 2)

    def intersection(self, other: 'BoxClass') -> float:
        intersection = BoxClass(
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
            x2=min(self.x2, other.x2),
            y2=min(self.y2, other.y2)
        )
        if intersection.x2 >= intersection.x1 and intersection.y2 >= intersection.y1:
            return intersection.get_area()
        else:
            return 0

    def iou(self, other: 'BoxClass') -> float:
        intersection = self.intersection(other)
        union = self.get_area() + other.get_area() - intersection
        if union == 0:
            return 0
        else:
            return intersection / union

    def contains_box(self, other: 'BoxClass', overlap_threshold=.5) -> bool:
        if other.get_area() == 0:
            return False
        else:
            return self.intersection(other
                                    ) / other.get_area() >= overlap_threshold

    def expand_box(self, amount: float) -> 'BoxClass':
        return BoxClass(
            x1=self.x1 - amount,
            y1=self.y1 - amount,
            x2=self.x2 + amount,
            y2=self.y2 + amount,
        )

    def crop_to_page(self, page_shape: ImageSize) -> 'BoxClass':
        page_height, page_width = page_shape[:2]
        return BoxClass(
            x1=max(self.x1, 0),
            y1=max(self.y1, 0),
            x2=min(self.x2, page_width),
            y2=min(self.y2, page_height),
        )


def enclosing_box(boxes: List[BoxClass]) -> BoxClass:
    assert len(boxes) > 0
    return BoxClass(
        x1=min([box.x1 for box in boxes]),
        y1=min([box.y1 for box in boxes]),
        x2=max([box.x2 for box in boxes]),
        y2=max([box.y2 for box in boxes])
    )


class Figure(JsonSerializable):
    figure_boundary = traits.Instance(BoxClass)
    caption_boundary = traits.Instance(BoxClass)
    caption_text = traits.Unicode()
    name = traits.Unicode()
    page = traits.Int()
    figure_type = traits.Unicode()
    dpi = traits.Int()
    page_width = traits.Int()
    page_height = traits.Int()
    # URI to cropped image of the figure
    uri = traits.Unicode(
        default_value=None, allow_none=True)

    def page_size(self) -> Tuple[int, int]:
        return self.page_height, self.page_width

    @staticmethod
    def from_pf_ann(ann: dict, target_page_size: Tuple[int, int]) -> 'Figure':
        """Convert an annotation in the pdffigures format"""
        cur_page_size = ann['page_height'], ann['page_width']
        if cur_page_size[0] is None:
            cur_page_size = [
                d * DEFAULT_INFERENCE_DPI / ann['dpi'] for d in target_page_size
            ]
        return Figure(
            figure_boundary=BoxClass.from_tuple(ann['region_bb'])
            .resize_by_page(cur_page_size, target_page_size),
            caption_boundary=BoxClass.from_tuple(ann['caption_bb'])
            .resize_by_page(cur_page_size, target_page_size),
            caption_text=ann['caption'],
            name=ann['name'],
            page=ann['page'],
            figure_type=ann['figure_type'],
            page_width=target_page_size[
                1
            ],
            page_height=target_page_size[
                0
            ]
        )

    @staticmethod
    def from_pf_output(res: dict, target_dpi=DEFAULT_INFERENCE_DPI) -> 'Figure':
        """Convert a pdffigures output figure to a Figure object"""
        scale_factor = target_dpi / 72
        return Figure(
            figure_boundary=BoxClass.from_dict(res['regionBoundary']
                                              ).rescale(scale_factor),
            caption_boundary=BoxClass.from_dict(res['captionBoundary'])
            .rescale(scale_factor),
            caption_text=res['caption'],
            name=res['name'],
            page=res['page'],
            figure_type=res['figType']
        )


class CaptionOnly(JsonSerializable):
    caption_boundary = traits.Instance(BoxClass)
    caption_text = traits.Unicode()
    name = traits.Unicode()
    page = traits.Int()
    figure_type = traits.Unicode()
    dpi = traits.Int()


class PdfDetectionResult(JsonSerializable):
    pdf = traits.Unicode()
    figures = traits.List(traits.Instance(Figure))
    dpi = traits.Int()
    raw_detected_boxes = traits.List(
        traits.List(traits.Instance(BoxClass)), allow_none=True
    )  # type: Optional[List[List[BoxClass]]]
    raw_pdffigures_output = traits.Dict(
        traits.Any(), allow_none=True
    )  # type: Optional[dict]
    error = traits.Unicode(
        default_value=None, allow_none=True
    )  # type: Optional[str]


class AuthorInfo(JsonSerializable):
    bounding_box = traits.Instance(BoxClass)
    name = traits.Unicode()


class TitleAndAuthorInfo(JsonSerializable):
    pdf = traits.Unicode()
    pdf_sha1 = traits.Unicode()
    image_path = traits.Unicode()
    title_bounding_box = traits.Instance(BoxClass)
    title_text = traits.Unicode()
    authors = traits.List(traits.Instance(AuthorInfo))
