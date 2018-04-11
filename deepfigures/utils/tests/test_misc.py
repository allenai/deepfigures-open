"""Test miscellaneous utilities."""

import hashlib
import os
import unittest

from deepfigures.utils import misc


class TestReadChunks(unittest.TestCase):
    """Test deepfigures.utils.misc.read_chunks."""

    def test_read_chunks(self):
        """Test read_chunks."""
        chunks_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data/chunks.txt')
        # read in the file as a string
        with open(chunks_path, 'rb') as f_in:
            contents = f_in.read()
        # verify that we iterate through the file correctly
        for i, chunk in enumerate(misc.read_chunks(chunks_path, block_size=1)):
            self.assertEqual(chunk, contents[i:i+1])
        for i, chunk in enumerate(misc.read_chunks(chunks_path, block_size=4)):
            self.assertEqual(chunk, contents[4*i:4*(i+1)])


class TestHashOutOfCore(unittest.TestCase):
    """Test deepfigures.utils.misc.hash_out_of_core."""

    def test_hash_out_of_core(self):
        """Test hash_out_of_core."""
        bigfile_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data/bigfile.txt')
        self.assertEqual(
            misc.hash_out_of_core(hashlib.sha1, bigfile_path),
            "329f37bbe1d7f23caf4f1868a4a256f168d84f15")
        self.assertEqual(
            misc.hash_out_of_core(hashlib.sha256, bigfile_path),
            "cbe4b71d97967575d12084b3702467f9dec2b22859c9a2407ea671fe17ed3d4a")
        self.assertEqual(
            misc.hash_out_of_core(hashlib.md5, bigfile_path),
            "ad4b675109d472d8c1ed006e395f8f14")
