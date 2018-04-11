"""Management commands for the deepfigures project.

``manage.py`` provides an interface to the scripts
automating development activities found in the `scripts`
directory.

See the ``scripts`` directory for examples.
"""

import logging
import sys

import click

from scripts import (
    build,
    detectfigures,
    generatearxiv,
    generatepubmed,
    testunits)


logger = logging.getLogger(__name__)

LOG_FORMAT = '%(asctime)s:%(levelname)s:%(name)s:%(message)s'


@click.group(
    context_settings={
        'help_option_names': ['-h', '--help']
    })
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Turn on verbose logging for debugging purposes.')
@click.option(
    '--log-file', '-l',
    type=str,
    help='Log to the provided file path instead of stdout.')
def manage(verbose, log_file):
    """A high-level interface to admin scripts for deepfigures."""
    log_level = logging.DEBUG if verbose else logging.INFO

    if log_file:
        logging.basicConfig(
            filename=log_file,
            filemode='a',
            format=LOG_FORMAT,
            level=log_level)
    else:
        logging.basicConfig(
            stream=sys.stdout,
            format=LOG_FORMAT,
            level=log_level)


subcommands = [
    build.build,
    detectfigures.detectfigures,
    generatearxiv.generatearxiv,
    generatepubmed.generatepubmed,
    testunits.testunits
]

for subcommand in subcommands:
    manage.add_command(subcommand)


if __name__ == '__main__':
    manage()
