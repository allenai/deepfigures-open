"""PDF Rendering engines for deepfigures."""

import glob
import json
import logging
import lxml
import os
import re
import shutil
import string
import subprocess
import typing

import bs4

from deepfigures.utils import file_util
from deepfigures.extraction import exceptions
from deepfigures import settings


logger = logging.getLogger(__name__)

# constant for use in the isprintable function
_PRINTABLES = set(string.printable)


def isprintable(s):
    """Return True if all characters in s are printable, else False.

    Parameters
    ----------
    :param str s: a string.

    Returns
    -------
    :return: True if s has only printable characters, otherwise False.
    """
    return set(s).issubset(_PRINTABLES)


class PDFRenderer(object):
    """Render PDFs and extract text from them.

    PDFRenderers are used to generate data and as part of the figure
    extraction pipeline for deepfigures. PDFRenderers must implement
    methods to render PDFs as images to disk and to extract text with
    bounding boxes that may later be parsed into classes from
    deepfigures.datamodels.

    Usage
    -----
    Subclass PDFRenderer and override:

      - RENDERING_ENGINE_NAME: a class variable giving a unique name
        that signals what backend was used to process the PDFs.
      - _rasterize_pdf: a method (see _rasterize_pdf on this class for
        details).
      - _extract_text: a method (see _extract_text on this class for
        details).

    """
    RENDERING_ENGINE_NAME = None
    IMAGE_FILENAME_RE = re.compile(
        r'(?P<pdf_name>.*)-dpi(?P<dpi>\d+)-page(?P<page_num>\d+).(?P<ext>png|jpg)'
    )
    IMAGE_FILENAME_PREFIX_TEMPLATE = \
        '{pdf_name}-dpi{dpi:d}-page'

    def __init__(self):
        """Initialize the PDFRenderer."""
        # check that subclasses override cls.RENDERING_ENGINE_NAME
        assert self.RENDERING_ENGINE_NAME is not None, (
            "class variable RENDERING_ENGINE_NAME must not be None"
        )

    def render(
        self,
        pdf_path: str,
        output_dir: typing.Optional[str]=None,
        dpi: int=settings.DEFAULT_INFERENCE_DPI,
        ext: str='png',
        max_pages: typing.Optional[int]=None,
        use_cache: bool=True,
        check_retcode: bool=False
    ) -> typing.List[str]:
        """Render pdf_path, save to disk and return the file paths.

        Render the pdf at pdf_path, save the generated image to disk
        in output dir using a file name matching the
        PDFRenderer.IMAGE_FILENAME_RE pattern, and return a list of
        paths to the generated files.

        Parameters
        ----------
        :param str pdf_path: path to the pdf that should be rendered.
        :param Optional[str] output_dir: path to the directory in which
          to save output. If None, then output is saved in the same
          directory as the PDF.
        :param int dpi: the dpi at which to render the PDF.
        :param str ext: the extension or file type of the generated
          image, should be either 'png' or 'jpg'.
        :param Optional[int] max_pages: the maximum number of pages to
          render from the PDF.
        :param bool use_cache: whether or not to skip the rendering
          operation if the pdf has already been rendered.
        :param bool check_retcode: whether or not to check the return
          code from the subprocess used to render the PDF.

        Returns
        -------
        :return: the list of generated paths
        """
        image_types = ['png', 'jpg']
        if ext not in image_types:
            raise ValueError(
                "ext must be one of {}".format(', '.join(image_types)))

        if output_dir is None:
            output_dir = os.path.dirname(pdf_path)

        if not os.path.isdir(output_dir):
            raise IOError(
                "Output directory ({}) does not exist.".format(output))

        pdf_name = os.path.basename(pdf_path)

        # engines_dir: directory used for storing the output from
        # different rendering engines.
        engines_dir = os.path.join(
            output_dir, '{pdf_name}-images'.format(pdf_name=pdf_name))
        # images_dir: directory used for storing images output by this
        # specific PDFRenderer / engine.
        images_dir = os.path.join(
            engines_dir,
            self.RENDERING_ENGINE_NAME,
            'dpi{}'.format(dpi))

        image_filename_prefix = self.IMAGE_FILENAME_PREFIX_TEMPLATE.format(
            pdf_name=pdf_name, dpi=dpi)
        image_output_path_prefix = os.path.join(
            images_dir, image_filename_prefix)
        success_file_path = os.path.join(images_dir, '_SUCCESS')

        if not os.path.exists(success_file_path) or not use_cache:
            if os.path.exists(images_dir):
                logger.info("Overwriting {}.".format(images_dir))
                shutil.rmtree(images_dir)
            os.makedirs(images_dir)

            self._rasterize_pdf(
                pdf_path=pdf_path,
                image_output_path_prefix=image_output_path_prefix,
                dpi=dpi,
                ext=ext,
                max_pages=max_pages,
                check_retcode=check_retcode)

            # add a success file to verify that the operation completed
            with open(success_file_path, 'w') as f_out:
                f_out.write('')

        generated_image_paths = glob.glob(
            image_output_path_prefix + '*.' + ext)

        return sort_by_page_num(generated_image_paths)

    def _rasterize_pdf(
        self,
        pdf_path: str,
        image_output_path_prefix: str,
        dpi: int,
        ext: str,
        max_pages: typing.Optional[int],
        check_retcode: bool,
    ) -> typing.List[str]:
        """Rasterize the PDF at PDF path and save it to disk.

        Rasterize the PDF at PDF path and save it to disk using
        image_output_path_prefix. Each page of the PDF should be
        rasterized separately and saved to the path formed by
        appending '{page_num:04d}.{ext}' to
        image_output_path_prefix.

        Parameters
        ----------
        :param str pdf_path: path to the pdf that should be rendered.
        :param str image_output_path_prefix: prefix for the output
          path of each rendered pdf page.
        :param int dpi: the dpi at which to render the pdf.
        :param int max_pages: the maximum number of pages to render
          from the pdf.

        Returns
        -------
        :return: None
        """
        raise NotImplementedError(
            "Subclasses of PDFRenderer must implement _rasterize_pdf."
        )

    def extract_text(self, pdf_path: str, encoding: str='UTF-8'
                    ) -> typing.Optional[bs4.BeautifulSoup]:
        """Extract info about a PDF as XML returning the parser for it.

        Extract information about the text, bounding boxes and pages of
        a PDF as XML, saving the XML to disk and returning a parser for
        it.

        Parameters
        ----------
        :param str pdf_path: the path to the pdf from which to extract
          information.
        :param str encoding: the encoding to use for the XML.

        Returns
        -------
        :return: A parser for the XML that is saved to disk.
        """
        # generate the html files
        self._extract_text(pdf_path=pdf_path, encoding=encoding)

        html = pdf_path[:-4] + '.html'
        if not os.path.isfile(html):
            html_soup = None
        try:
            with open(html, 'r') as f:
                html_soup = bs4.BeautifulSoup(f, 'xml')
        except UnicodeDecodeError:
            html_soup = None

        if html_soup is None:
            raise exceptions.PDFProcessingError(
                "Error in extracting xml for {}.".format(pdf_path)
            )

        return html_soup

    def _extract_text(self, pdf_path: str, encoding: str='UTF-8') -> None:
        """Extract text from a PDF and save to disk as xml.

        Parameters
        ----------
        :param str pdf_path: path to the PDF to be extracted.
        :param str encoding: the encoding to use for saving the XML.

        Returns
        -------
        :return: None
        """
        raise NotImplementedError(
            "Subclasses of PDFRenderer must implement _extract_text."
        )


class GhostScriptRenderer(PDFRenderer):
    """Render PDFs using GhostScript."""
    RENDERING_ENGINE_NAME = 'ghostscript'

    def _rasterize_pdf(
        self,
        pdf_path: str,
        image_output_path_prefix: str,
        dpi: int,
        ext: str,
        max_pages: typing.Optional[int],
        check_retcode: bool
    ) -> typing.List[str]:
        """Rasterize a PDF using GhostScript."""
        # ghostscript requires a template string for the output path
        image_output_path_template = image_output_path_prefix + '%04d.{ext}'.format(
            ext=ext)
        sdevice = 'png16m' if ext == 'png' else 'jpeg'
        gs_args = [
            'gs', '-dGraphicsAlphaBits=4', '-dTextAlphaBits=4', '-dNOPAUSE', '-dBATCH', '-dSAFER', '-dQUIET',
            '-sDEVICE=' + sdevice,
            '-r%d' % dpi, '-sOutputFile=' + image_output_path_template,
            '-dBufferSpace=%d' % int(1e9),
            '-dBandBufferSpace=%d' % int(5e8), '-sBandListStorage=memory',
            '-c',
            '%d setvmthreshold' % int(1e9), '-dNOGC',
            '-dNumRenderingThreads=4', "-f", pdf_path
        ]
        if max_pages is not None:
            gs_args.insert(-2, '-dLastPage=%d' % max_pages)
        subprocess.run(gs_args, check=check_retcode)

    def _extract_text(self, pdf_path: str, encoding: str) -> None:
        """Extract text using pdftotext."""
        subprocess.run(['pdftotext', '-bbox', '-enc', encoding, pdf_path])


def sort_by_page_num(file_paths: typing.List[str]) -> typing.List[str]:
    """Sort file_paths by the page number.

    Sort file_paths by the page number where file_paths is a list
    of rendered output image file paths generated by a
    PDFRenderer.

    Parameters
    ----------
    :param List[str] file_paths: a list of file paths generated by
    a PDFRenderer.

    Returns
    -------
    file_paths sorted by page number.
    """
    return sorted(
        file_paths,
        key=lambda file_path: int(PDFRenderer.IMAGE_FILENAME_RE.fullmatch(
            os.path.split(file_path)[-1]).group('page_num')))
