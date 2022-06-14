"""Run figure detection on a PDF.

See ``detectfigures.py --help`` for more information.
"""

import logging
import os

import click

from deepfigures import settings
from scripts import build, execute


logger = logging.getLogger(__name__)


@click.command(
    context_settings={
        'help_option_names': ['-h', '--help']
    })
@click.option(
    '--skip-dependencies', '-s',
    is_flag=True,
    help='skip running dependency commands.')
@click.argument(
    'output_directory',
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        resolve_path=True))
@click.argument(
    'pdf_path',
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True))
def detectfigures(
        output_directory,
        pdf_path,
        skip_dependencies=False):
    """Run figure extraction on the PDF at PDF_PATH.

    Run figure extraction on the PDF at PDF_PATH and write the results
    to OUTPUT_DIRECTORY.
    """
    if not skip_dependencies:
        build.build.callback()

    cpu_docker_img = settings.DEEPFIGURES_IMAGES['cpu']

    pdf_directory, pdf_name = os.path.split(pdf_path)

    internal_output_directory = '/work/host-output'
    internal_pdf_directory = '/work/host-input'

    internal_pdf_path = internal_pdf_directory + '/' + pdf_name
    
    execute(
        'docker run'
        ' --rm'
        ' --env-file deepfigures-local.env'
        ' --volume {output_directory}:{internal_output_directory}'
        ' --volume {pdf_directory}:{internal_pdf_directory}'
        ' {tag}:{version}'
        ' python3 /work/scripts/rundetection.py'
        '   {internal_output_directory}'
        '   {internal_pdf_path}'.format(
            tag=cpu_docker_img['tag'],
            version=settings.VERSION,
            output_directory=output_directory,
            internal_output_directory=internal_output_directory,
            pdf_directory=pdf_directory,
            internal_pdf_directory=internal_pdf_directory,
            internal_pdf_path=internal_pdf_path),
        logger,
        raise_error=True)


if __name__ == '__main__':
    detectfigures()
