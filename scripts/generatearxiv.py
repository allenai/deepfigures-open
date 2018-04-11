"""Generate arxiv data for deepfigures.

Generate the arxiv data for deepfigures. This data generation process
requires pulling down all the arxiv source files from S3 which the
requester (person executing this script) must pay for.

See ``generatearxiv.py --help`` for more information.
"""

import logging

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
def generatearxiv(skip_dependencies=True):
    """Generate arxiv data for deepfigures.

    Generate the arxiv data for deepfigures, which involves pulling the
    data from S3 (which the requestor has to pay for).
    """
    if not skip_dependencies:
        build.build.callback()

    cpu_docker_img = settings.DEEPFIGURES_IMAGES['cpu']

    execute(
        'docker run'
        ' --rm'
        ' --env-file deepfigures-local.env'
        ' --volume {ARXIV_DATA_TMP_DIR}:{ARXIV_DATA_TMP_DIR}'
        ' --volume {ARXIV_DATA_OUTPUT_DIR}:{ARXIV_DATA_OUTPUT_DIR}'
        ' {tag}:{version}'
        ' python3'
        ' /work/deepfigures/data_generation/arxiv_pipeline.py'.format(
            tag=cpu_docker_img['tag'],
            version=settings.VERSION,
            ARXIV_DATA_TMP_DIR=settings.ARXIV_DATA_TMP_DIR,
            ARXIV_DATA_OUTPUT_DIR=settings.ARXIV_DATA_OUTPUT_DIR),
        logger,
        raise_error=True)


if __name__ == '__main__':
    generatearxiv()
