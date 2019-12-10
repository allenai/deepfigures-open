"""Run unit tests for deepfigures.

Run unit tests for deepfigures locally in a docker container, building
the required docker images before hand.

See ``testunits.py --help`` for more information.
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
def testunits(skip_dependencies=False):
    """Run unit tests for deepfigures."""
    if not skip_dependencies:
        build.build.callback()

    cpu_docker_img = settings.DEEPFIGURES_IMAGES['cpu']

    execute(
        'docker run'
        ' --rm'
        ' --env-file deepfigures-local.env'
        ' {tag}:{version}'
        ' python3 /work/scripts/runtests.py'.format(
            tag=cpu_docker_img['tag'],
            version=cpu_docker_img['version_prefix'] + settings.VERSION),
        logger,
        raise_error=True)


if __name__ == '__main__':
    testunits()
