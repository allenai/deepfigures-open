"""Generate pubmed data for deepfigures.

See ``generatepubmed.py --help`` for more information.
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
def generatepubmed(skip_dependencies=True):
    """Generate pubmed data for deepfigures.

    Generate the pubmed data for deepfigures, which can involve pulling
    the data from S3 (which the requestor has to pay for).
    """
    if not skip_dependencies:
        build.build.callback()

    cpu_docker_img = settings.DEEPFIGURES_IMAGES['cpu']

    execute(
        'docker run'
        ' --rm'
        ' --env-file deepfigures-local.env'
        ' --volume {LOCAL_PUBMED_DISTANT_DATA_DIR}:{LOCAL_PUBMED_DISTANT_DATA_DIR}'
        ' {tag}:{version}'
        ' python3'
        ' /work/deepfigures/data_generation/pubmed_pipeline.py'.format(
            tag=cpu_docker_img['tag'],
            version=settings.VERSION,
            LOCAL_PUBMED_DISTANT_DATA_DIR=settings.LOCAL_PUBMED_DISTANT_DATA_DIR),
        logger,
        raise_error=True)


if __name__ == '__main__':
    generatepubmed()
