"""Detect the figures in a PDF."""

import logging
import os

import click


logger = logging.getLogger(__name__)


@click.command(
    context_settings={
        'help_option_names': ['-h', '--help']
        })
@click.argument(
    'output_directory',
    type=click.Path(file_okay=False))
@click.argument(
    'pdf_path',
    type=click.Path(exists=True, dir_okay=False))
def rundetection(output_directory, pdf_path):
    """Detect figures from the pdf at PDF_PATH.

    Detect the figures from the pdf located at PDF_PATH and write the
    detection results to the directory specified by OUTPUT_DIRECTORY.
    """
    # import lazily to speed up response time for returning help text
    from deepfigures.extraction import pipeline

    figure_extractor = pipeline.FigureExtractionPipeline()

    figure_extractor.extract(pdf_path, output_directory)


if __name__ == '__main__':
    rundetection()
