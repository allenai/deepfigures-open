"""Miscellaneous utilities."""

import hashlib


def read_chunks(input_path, block_size):
    """Iterate over ``block_size`` chunks of file at ``input_path``.

    :param str input_path: the path to the input file to iterate over.
    :param int block_size: the size of the chunks to return at each
      iteration.

    :yields: a binary chunk of the file at ``input_path`` of size
      ``block_size``.
    """
    with open(input_path, 'rb') as f_in:
        while True:
            chunk = f_in.read(block_size)
            if chunk:
                yield chunk
            else:
                return


def hash_out_of_core(hash_func, input_path):
    """Return hexdigest of file at ``input_path`` using ``hash_func``.

    Hash the file at ``input_path`` using ``hash_func`` in an
    out-of-core way, allowing hashing of arbitrarily large files, and
    then return the hexdigest.

    :param _hashlib.HASH hash_func: a hashing function from hashlib such
      as sha1 or md5.
    :param str input_path: path to the input file.

    :returns: the hexdigest of the file at ``input_path`` hashed using
      ``hash_func``.

    Example
    -------
    To use SHA256 to compute the hash of a file out of core:

        hash_out_of_core(hashlib.sha256, '/path/to/file')

    """
    hf = hash_func()
    for chunk in read_chunks(input_path, 256 * (128 * hf.block_size)):
        hf.update(chunk)
    return hf.hexdigest()
