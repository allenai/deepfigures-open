import os
import typing
import numpy as np
from scipy import misc
from deepfigures.utils import file_util
import logging

class FileTooLargeError(Exception):
    pass


def read_tensor(path: str, maxsize: int=None) -> typing.Optional[np.ndarray]:
    """
    Load a saved a tensor, saved either as an image file for standard RGB images or as a numpy archive for more general
    tensors.
    """
    path = file_util.cache_file(path)
    if maxsize is not None:
        if os.path.getsize(path) > maxsize:
            raise FileTooLargeError
    (_, ext) = os.path.splitext(path)
    ext = ext.lower()
    if ext in {'.png', '.jpg', '.jpeg'}:
        res = misc.imread(path, mode='RGB')
        assert len(res.shape) == 3
        assert res.shape[2] == 3
        return res
    elif ext in {'.npz'}:
        try:
            data = np.load(path)
            assert len(list(data.items())) == 1
        except Exception as e:
            logging.exception('Error unzipping %s' % path)
            return None
        return data['arr_0']
    else:
        raise RuntimeError('Extension %s for file %s not supported' % (ext, path))


def write_tensor(dst: str, value: np.ndarray) -> None:
    """Save a numpy tensor to a given location."""
    (_, ext) = os.path.splitext(dst)
    assert (ext == '' or ext == '.npz')
    with open(dst, 'wb') as f:
        np.savez_compressed(f, value)


def imresize_multichannel(im: np.ndarray, target_size: typing.Tuple[int, int],
                          **kwargs) -> np.ndarray:
    n_channels = im.shape[2]
    resized_channels = [
        misc.imresize(im[:, :, n], target_size, **kwargs) for n in range(n_channels)
    ]
    return np.stack(resized_channels, axis=2)


def imrescale_multichannel(im: np.ndarray, scale_factor: float, **kwargs) -> np.ndarray:
    n_channels = im.shape[2]
    resized_channels = [
        misc.imresize(im[:, :, n], scale_factor, **kwargs) for n in range(n_channels)
    ]
    return np.stack(resized_channels, axis=2)
