"""Utilities for managing settings."""

from importlib import import_module


def import_setting(import_string):
    """Import and return the object defined by import_string.

    This function is helpful because by the nature of settings files,
    they often end up with circular imports, i.e. ``foo`` will import
    ``settings`` to get configuration information but ``settings`` will
    have some setting set to an object imported from ``foo``. Because
    python can't do circular imports, we make the settings strings and
    then import them at runtime from the string using this function.

    Parameters
    ----------
    :param str import_string: the python path to the object you wish to
      import. ``import_string`` should be a dot separated path the same
      as you would use in a python import statement.

    Returns
    -------
    :returns: any module or object located at ``import_string`` or
      ``None`` if no module exists.
    """
    try:
        module = import_module(import_string)
    except ImportError:
        module = None

    if not module:
        mod_string, obj_string = import_string.rsplit('.', 1)
        obj = getattr(import_module(mod_string), obj_string)

    return module or obj
