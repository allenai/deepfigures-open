"""The figure extraction pipeline for deepfigures.

The ``deepfigures.extraction.pipeline`` module defines the figure
extraction pipeline for deepfigures, including copying the PDF to a
location, rendering a PDF, finding captions for figures, detecting the
figures and cropping out the images.
"""

import hashlib
import os
import shutil

from PIL import Image

from deepfigures import settings
from deepfigures.extraction import (
    detection,
    pdffigures_wrapper,
    renderers)
from deepfigures.utils import (
    misc,
    settings_utils)


class FigureExtraction(object):
    """A class representing the data extracted from a PDF.

    The ``FigureExtraction`` class represents the data extracted from
    a single PDF and is generated through the ``extract`` method of
    the ``FigureExtractionPipeline`` class.

    The data stored for a ``FigureExtraction`` instance sits on disk
    in a directory. See `Attributes`_ for more information.

    Attributes
    ----------
    path_templates : Dict[str, str]
        A class attribute providing the templates for the paths to the
        extracted data on disk, relative to the data directory.
    paths : Dict[str, str]
        A dictionary mapping path names to their actual absolute paths
        on disk.
    parent_directory : str
        The parent directory for the directory containing the extracted
        data.
    low_res_rendering_paths : Optional[str]
        Paths to the low resolution renderings of the PDF (used for
        predicting the bounding boxes).
    hi_res_rendering_paths : Optional[str]
        Paths to the high resolution renderings of the PDF (used for
        cropping out the figure images).
    pdffigures_output_path : Optional[str]
        Path to the output of running pdffigures2 on the PDF.
    deepfigures_json_path : Optional[str]
        Path to the deepfigures JSON predicting the bounding boxes.
    """

    """Templates for paths to the data extracted from a PDF."""
    path_templates = {
        'BASE': '{pdf_hash}',
        'PDF_PATH': '{base}/{pdf_name}',
        'RENDERINGS_PATH': '{base}/page-renderings',
        'PDFFIGURES_OUTPUT_PATH': '{base}/pdffigures-output',
        'DEEPFIGURES_OUTPUT_PATH': '{base}/deepfigures-output',
        'FIGURE_IMAGES_PATH': '{base}/figure-images'
    }

    def __init__(self, pdf_path, parent_directory):
        """Initialize a ``FigureExtraction`` instance.

        Parameters
        ----------
        pdf_path : str
            The path to the PDF locally on disk.
        parent_directory : str
            The parent directory for the directory in which the figure
            extraction results will be stored.
        """
        # compute strings to fill in the path templates
        pdf_hash = misc.hash_out_of_core(hashlib.sha1, pdf_path)
        pdf_name = os.path.basename(pdf_path)
        base = self.path_templates['BASE'].format(pdf_hash=pdf_hash)
        template_vars = {
            'pdf_hash': pdf_hash,
            'pdf_name': pdf_name,
            'base': base
        }
        # set the paths attribute
        self.paths = {
            k: os.path.join(parent_directory, v.format(**template_vars))
            for k, v in self.path_templates.items()
        }
        self.parent_directory = parent_directory
        self.low_res_rendering_paths = None
        self.hi_res_rendering_paths = None
        self.pdf_figures_output_path = None
        self.deepfigures_json_path = None


class FigureExtractionPipeline(object):
    """A class for extracting figure data from PDFs.

    The ``FigureExtractionPipeline`` class's main function is to
    generate instances of ``FigureExtraction``. Each instance of a
    ``FigureExtraction`` represents the data extracted from processing a
    single PDF.

    See the ``FigureExtraction`` class's doc string for details on
    the format that this extracted data takes.
    """

    def extract(self, pdf_path, output_directory):
        """Return a ``FigureExtraction`` instance for ``pdf_path``.

        Extract the figures and additional information from the PDF at
        ``pdf_path``, saving the results to disk in ``output_directory``
        and returning the corresponding ``FigureExtraction`` instance.

        Parameters
        ----------
        pdf_path : str
            The path to the PDF.
        output_directory : str
            The directory in which to save the results from extraction.

        Returns
        -------
        FigureExtraction
            A ``FigureExtraction`` instance for the PDF at ``pdf_path``.
        """
        figure_extraction = FigureExtraction(
            pdf_path=pdf_path,
            parent_directory=output_directory)

        # create the extraction results directory
        if os.path.exists(figure_extraction.paths['BASE']) and os.path.isdir(figure_extraction.paths['BASE']):
            pass
            # os.chmod(figure_extraction.paths['BASE'], 0o777)
            # shutil.rmtree(figure_extraction.paths['BASE'])
        else:
            os.makedirs(figure_extraction.paths['BASE'])

        # copy the PDF into the extraction results directory
        shutil.copy(pdf_path, figure_extraction.paths['PDF_PATH'])

        pdf_renderer = settings_utils.import_setting(
            settings.DEEPFIGURES_PDF_RENDERER)()

        # render the PDF into low-res images
        figure_extraction.low_res_rendering_paths = \
            pdf_renderer.render(
                pdf_path=figure_extraction.paths['PDF_PATH'],
                output_dir=figure_extraction.paths['BASE'],
                dpi=settings.DEFAULT_INFERENCE_DPI)

        # render the PDF into hi-res images
        figure_extraction.hi_res_rendering_paths = \
            pdf_renderer.render(
                pdf_path=figure_extraction.paths['PDF_PATH'],
                output_dir=figure_extraction.paths['BASE'],
                dpi=settings.DEFAULT_CROPPED_IMG_DPI)

        # extract captions from PDF using pdffigures2
        figure_extraction.pdffigures_output_path = \
            pdffigures_wrapper.pdffigures_extractor.extract(
                pdf_path=figure_extraction.paths['PDF_PATH'],
                output_dir=figure_extraction.paths['BASE'])

        # run deepfigures / neural networks on the PDF images
        figure_extraction.deepfigures_json_path = \
            detection.extract_figures_json(
                pdf_path=figure_extraction.paths['PDF_PATH'],
                page_image_paths=figure_extraction.low_res_rendering_paths,
                pdffigures_output=figure_extraction.pdffigures_output_path,
                output_directory=figure_extraction.paths['BASE'])

        return figure_extraction
