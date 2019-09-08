import collections
import os
import subprocess
from typing import Callable, Dict, Iterable, List, Tuple, TypeVar

from matplotlib import axes
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import optimize
from deepfigures.utils import file_util
from deepfigures.extraction.renderers import PDFRenderer
from deepfigures.extraction.exceptions import LatexException
from deepfigures.extraction.datamodels import (BoxClass, Figure)
from deepfigures.settings import DEFAULT_INFERENCE_DPI


def call_pdflatex(
    src_tex: str, src_dir: str, dest_dir: str, timeout: int=1200
) -> str:
    """
    Call pdflatex on the tex source file src_tex, save its output to dest_dir, and return the path of the
    resulting pdf.
    """
    # Need to be in the same directory as the file to compile it
    file_util.safe_makedirs(dest_dir)
    # Shell-escape required due to https://www.scivision.co/pdflatex-error-epstopdf-output-filename-not-allowed-in-restricted-mode/
    cmd = [
        'pdflatex', '-interaction=nonstopmode', '-shell-escape',
        '-output-directory=' + dest_dir, src_tex
    ]
    # Run twice so that citations are built correctly
    # Had some issues getting latexmk to work
    try:
        subprocess.run(
            cmd, stdout=subprocess.PIPE, cwd=src_dir, timeout=timeout
        )
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, cwd=src_dir, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        raise LatexException(
            ' '.join(cmd), -1, 'Timeout exception after %d' % timeout
        )
    if res.returncode != 0:
        raise LatexException(' '.join(cmd), res.returncode, res.stdout)
    paperid = os.path.splitext(os.path.basename(src_tex))[0]
    return dest_dir + paperid + '.pdf'


def im_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Returns a copy of image 'a' with all pixels where 'a' and 'b' are equal set to white."""
    assert (np.array_equal(np.shape(a), np.shape(b)))
    diff = a - b
    mask = np.any(diff != 0, axis=2)  # Check if any channel is different
    rgb_mask = np.transpose(np.tile(mask, (3, 1, 1)), axes=[1, 2, 0])
    diff_image = np.copy(a)
    diff_image[np.logical_not(rgb_mask)] = 255
    return diff_image


def pair_boxes(a_boxes: List[BoxClass],
               b_boxes: List[BoxClass]) -> Tuple[List[int], List[int]]:
    """
    Find the pairing between boxes with the lowest total distance, e.g. for matching figures to their captions.
    This is an instance of the linear assignment problem and can be solved efficiently using the Hungarian algorithm.
    Return the indices of matched boxes. If a_boxes and b_boxes are of unequal length, not all boxes will be paired.
    Length of returned lists is min(len(a_boxes), len(b_boxes)).
    """
    a_len = len(a_boxes)
    b_len = len(b_boxes)
    cost_matrix = np.zeros([a_len, b_len])
    cost_matrix[:] = np.nan
    for (a_idx, a_box) in enumerate(a_boxes):
        for (b_idx, b_box) in enumerate(b_boxes):
            cost_matrix[a_idx, b_idx] = a_box.distance_to_other(b_box)
    assert (cost_matrix != np.nan).all()
    (a_indices, b_indices) = optimize.linear_sum_assignment(cost_matrix)
    assert len(a_indices) == len(b_indices)
    return a_indices, b_indices


def load_figures_json(filename: str) -> Dict[str, List[Figure]]:
    d = file_util.read_json(filename)
    res = {
        page: [Figure.from_dict(dict_fig) for dict_fig in page_dicts]
        for (page, page_dicts) in d.items()
    }
    return res


T = TypeVar('T')
S = TypeVar('S')


def group_by(l: Iterable[T],
             key: Callable[[T], S]=lambda x: x) -> Dict[S, List[T]]:
    """Like itertools.groupby but doesn't require first sorting by the key function. Returns a dict."""
    d = collections.defaultdict(list)
    assert (callable(key))
    for item in l:
        d[key(item)].append(item)
    return d


def ordered_group_by(l: Iterable[T],
                     key: Callable[[T], S]=lambda x: x) -> Dict[S, List[T]]:
    """Keys are returned in order of first occurrence."""
    d = collections.OrderedDict()
    assert (callable(key))
    for item in l:
        k = key(item)
        if k not in d:
            d[k] = []
        d[k].append(item)
    return d


def group_figures_by_pagenum(figs: Iterable[Figure]
                            ) -> Dict[int, List[Figure]]:
    return group_by(figs, lambda x: x.page)


def make_axes(size: Tuple[float, float]=(20, 20)) -> axes.Subplot:
    fig, ax = plt.subplots(1, figsize=size)
    return ax


def pagename_to_pagenum(pagename: str) -> int:
    """Takes a page name with a 1-indexed number and returns the 0-indexed page number."""
    return int(
        PDFRenderer.IMAGE_FILENAME_RE.fullmatch(pagename).group('page_num')
    ) - 1


def pagenum_to_pagename(pdf: str, pagenum: int, dpi: int=DEFAULT_INFERENCE_DPI) -> str:
    """Takes a pdf and a page with 0-indexed number and returns the 1-indexed page image name."""
    return os.path.join(
        os.path.dirname(pdf),
        (PDFRenderer.IMAGE_FILENAME_PREFIX_TEMPLATE +
            '{page_num:04d}.png').format(
                pdf_name=os.path.split(pdf)[-1], dpi=dpi, page_num=pagenum + 1
            ))


def pagename_to_pdf(pagename: str) -> str:
    """Takes a page image name and returns the name of the pdf it came from."""
    return PDFRenderer.IMAGE_FILENAME_RE.fullmatch(pagename).group('pdf_name')
