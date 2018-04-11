"""Run tests for deepfigures."""

import logging

import click

from scripts import execute


logger = logging.getLogger(__name__)


@click.command(
    context_settings={
        'help_option_names': ['-h', '--help']
    })
def runtests():
    """Run tests for deepfigures."""

    # init logging
    logger.setLevel(logging.INFO)
    logging.basicConfig()

    logger.info('Running tests for deepfigures.')
    execute(
        'pytest -n auto /work/deepfigures',
        logger)


if __name__ == '__main__':
    runtests()
