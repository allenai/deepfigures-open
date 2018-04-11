"""Build docker images for deepfigures.

See ``build.py --help`` for more information.
"""

import logging

import click

from deepfigures import settings
from scripts import execute


logger = logging.getLogger(__name__)


@click.command(
    context_settings={
        'help_option_names': ['-h', '--help']
    })
def build():
    """Build docker images for deepfigures."""
    for _, docker_img in settings.DEEPFIGURES_IMAGES.items():
        tag = docker_img['tag']
        dockerfile_path = docker_img['dockerfile_path']

        execute(
            'docker build'
            ' --tag {tag}:{version}'
            ' --file {dockerfile_path} .'.format(
                tag=tag,
                version=settings.VERSION,
                dockerfile_path=dockerfile_path),
            logger)


if __name__ == '__main__':
    build()
