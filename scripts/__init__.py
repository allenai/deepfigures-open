"""Scripts automating development tasks for deepfigures."""

import subprocess


def execute(
        command,
        logger,
        quiet=False,
        raise_error=True):
    """Execute ``command``.

    Parameters
    ----------
    command : str
        The command to execute in the shell.
    logger : logging.RootLogger
        The logger to use for logging output about the command.
    quiet : bool
        Prevent the subprocess from printing output to stdout.
    raise_error : bool
        If ``True`` then raise an error when the command returns a
        non-zero exit status, else log the error as a warning.

    Returns
    -------
    None
    """
    if quiet :
        logger.info(
            'Executing command and supressing stdout: {command}'.format(
                command=command))

        p = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            shell=True)
    else:
        logger.info(
            'Executing: {command}'.format(
                command=command))

        p = subprocess.Popen(
            command,
            shell=True)

    p.communicate()

    returncode = p.returncode

    if raise_error and returncode != 0:
        raise subprocess.CalledProcessError(
            returncode=returncode,
            cmd=command)
    elif not raise_error and returncode != 0:
        logger.warning(
            'Command: "{command}" exited with returncode'
            ' {returncode}'.format(
                command=command,
                returncode=returncode))

    return None
