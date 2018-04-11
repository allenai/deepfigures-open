import os
import subprocess
from typing import List, Optional, Iterable
import tempfile
from deepfigures.utils import file_util
from deepfigures.extraction import datamodels
from deepfigures import settings
import logging
import shlex
import contextlib
import more_itertools


# DPI used by pdffigures for json outputs; this is hard-coded as 72
PDFFIGURES_DPI = 72


class PDFFiguresExtractor(object):
    """Extract figure and caption information from a PDF."""

    def extract(self, pdf_path, output_dir, use_cache=True):
        """Return results from extracting a PDF with pdffigures2.

        :param str pdf_path: path to the PDF to extract.
        :param str output_dir: path to the output directory.
        :param bool use_cache: whether or not to use cached data from
          disk if it's available.

        :returns: results from running pdffigures2 on the PDF.
        """
        pdffigures_dir = os.path.join(output_dir, 'pdffigures/')
        if not os.path.exists(pdffigures_dir):
            os.makedirs(pdffigures_dir)

        success_file_path = os.path.join(pdffigures_dir, '_SUCCESS')

        pdffigures_jar_path = file_util.cache_file(
            settings.PDFFIGURES_JAR_PATH)

        if not os.path.exists(success_file_path) or not use_cache:
            subprocess.check_call(
                'java'
                ' -jar {pdffigures_jar_path}'
                ' --figure-data-prefix {pdffigures_dir}'
                ' --save-regionless-captions'
                ' {pdf_path}'.format(
                    pdffigures_jar_path=pdffigures_jar_path,
                    pdf_path=pdf_path,
                    pdffigures_dir=pdffigures_dir),
                shell=True)

            # add a success file to verify that the operation completed
            with open(success_file_path, 'w') as f_out:
                f_out.write('')

        return file_util.read_json(
            os.path.join(
                pdffigures_dir,
                os.path.basename(pdf_path)[:-4] + '.json'))


pdffigures_extractor = PDFFiguresExtractor()


def figure_to_caption(figure: dict) -> datamodels.CaptionOnly:
    return datamodels.CaptionOnly(
        caption_boundary=datamodels.BoxClass.
        from_dict(figure['captionBoundary']),
        page=figure['page'],
        caption_text=figure['caption'],
        name=figure['name'],
        figure_type=figure['figType'],
    )


def regionless_to_caption(regionless: dict) -> datamodels.CaptionOnly:
    return datamodels.CaptionOnly(
        caption_boundary=datamodels.BoxClass.from_dict(regionless['boundary']),
        page=regionless['page'],
        caption_text=regionless['text'],
        name=regionless['name'],
        figure_type=regionless['figType'],
    )


def get_captions(
    pdffigures_output: dict, target_dpi: int=settings.DEFAULT_INFERENCE_DPI
) -> List[datamodels.CaptionOnly]:
    figures = pdffigures_output.get('figures', [])
    regionless_captions = pdffigures_output.get('regionless-captions', [])
    captions = (
        [figure_to_caption(fig) for fig in figures] +
        [regionless_to_caption(reg) for reg in regionless_captions]
    )
    for caption in captions:
        caption.caption_boundary = caption.caption_boundary.rescale(
            target_dpi / PDFFIGURES_DPI
        )
    return captions


def get_figures(pdffigures_output: dict, target_dpi: int=settings.DEFAULT_INFERENCE_DPI
               ) -> List[datamodels.Figure]:
    return [
        datamodels.Figure.from_pf_output(figure, target_dpi)
        for figure in pdffigures_output.get('figures', [])
    ]


def detect_batch(src_pdfs: List[str], target_dpi: int = settings.DEFAULT_INFERENCE_DPI, chunksize=1) -> \
        Iterable[datamodels.PdfDetectionResult]:
    for chunk in more_itertools.chunked(src_pdfs, chunksize):
        results = [
            pdffigures_extractor.extract(pdf_path, os.path.dirname(pdf_path))
            for pdf_path in chunk
        ]
        for (result, pdf) in zip(results, chunk):
            figs = get_figures(result, target_dpi=target_dpi)
            yield datamodels.PdfDetectionResult(
                pdf=pdf,
                figures=figs,
                dpi=target_dpi,
                raw_detected_boxes=None,
                raw_pdffigures_output=None,
                error=None
            )
