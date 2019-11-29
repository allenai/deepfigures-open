"""
Train the deepfigures model.
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
    'hypes',
    type=click.Path()
)
@click.argument(
    'output_directory',
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        resolve_path=True))
@click.argument(
    'input_directory',
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        resolve_path=True))
def train(output_directory,
          input_directory,
          hypes='/work/weights/hypes.json',
          skip_dependencies=False):
    """ Traing the deepfigures model.
    :param output_directory: The directory on your host where you want to store the output. (weights, etc.)
    :param input_directory: The directory which contains the data for training, hypes, etc.
    :param hypes: The JSO file which contains the hyper-parameters for training.
    :param skip_dependencies: Set this to True if you do not want to pre-check if the dependencies before running.
    :return: Nothing.
    """

    if not skip_dependencies:
        build.build.callback()

    cpu_docker_img = settings.DEEPFIGURES_IMAGES['gpu']

    docker_output_directory = '/work/host-output'
    docker_input_directory = '/work/host-input'

    execute(
        'docker run'
        ' --rm'
        ' --env-file deepfigures-local.env'
        ' --volume {host_input_path}:{docker_input_path}'
        ' --volume {host_output_path}:{docker_output_path}'
        ' {tag}:{version}'
        ' python vendor/tensorboxresnet/tensorboxresnet/train.py'
        ' --hypes {hypes_path}'
        ' --gpu 1'
        ' --logdir {docker_output_path}'.format(
            host_input_path=input_directory,
            docker_input_path=docker_input_directory,
            host_output_path=output_directory,
            docker_output_path=docker_output_directory,
            tag=cpu_docker_img['tag'],
            version=cpu_docker_img['version_prefix'] + settings.VERSION,
            hypes_path=hypes),
        logger,
        raise_error=True)


if __name__ == "__main__":
    train()
