"""Tests for deepfigures.extraction.renderers"""

import contextlib
import logging
import os
import shutil
import time
import tempfile
import unittest

import numpy as np
from scipy.misc import imread
import pytest

from deepfigures.extraction import renderers
from deepfigures import settings


logger = logging.getLogger(__name__)


class IsPrintableTest(unittest.TestCase):
    """Test deepfigures.renderers.isprintable."""

    def test_returns_correct_values(self):
        """Test isprintable returns the correct values."""
        # test empty string
        self.assertTrue(renderers.isprintable(''))

        # test single printable characters
        self.assertTrue(renderers.isprintable('a'))
        self.assertTrue(renderers.isprintable('#'))
        self.assertTrue(renderers.isprintable('!'))
        self.assertTrue(renderers.isprintable('|'))

        # test multicharacter strings
        self.assertTrue(renderers.isprintable('aasfd'))
        self.assertTrue(renderers.isprintable('a*&($#asdf!'))

        # test nonprintable chars
        self.assertFalse(renderers.isprintable('\x0e'))
        self.assertFalse(renderers.isprintable('afj\x0eqq'))


class PDFRendererTest(unittest.TestCase):
    """Tests for deepfigures.renderers.PDFRenderer.

    Since PDFRenderer is only meant as a base class for classes that
    actually use a rendering backend, most of it's functionality is
    tested through tests of it's subclasses (GhostScriptRenderer).
    """

    def test_init(self):
        """Test init asserts RENDERING_ENGINE_NAME exists."""
        with self.assertRaises(AssertionError):
            renderers.PDFRenderer()


class PDFRendererSubclassTestMixin(object):
    """A mixin for making tests of PDFRenderer subclasses.

    Usage
    -----
    To test a PDFRenderer, mix this class into a unittest.TestCase
    subclass, provide PDF_RENDERER and MANUALLY_INSPECTED_RENDERINGS_DIR
    class attributes on that subclass, and render / manually inspect
    images of each page for
    deepfigures/tests/data/pdfrenderer/paper.pdf.

    PDF_RENDERER should be an instance of the pdf renderer class you
    wish to test, and MANUALLY_INSPECTED_RENDERINGS_DIR should be a
    directory containing renderings using PDF_RENDERER that have been
    manually inspected and match the paths in
    deepfigures/tests/data/pdfrenderer/pdfbox-renderings/.

    Example
    -------

        class GhostScriptRendererTest(
                PDFRendererSubclassTestMixin,
                unittest.TestCase):
            '''... documentation ...'''
            PDF_RENDERER = GhostScriptRenderer()
            MANUALLY_INSPECTED_RENDERINGS_DIR = os.path.join(
                settings.TEST_DATA_DIR,
                'pdfrenderer/ghostscript-renderings/')

            def ghostscript_renderer_specific_test(self):
                ...
    """
    PDF_RENDERER = None
    MANUALLY_INSPECTED_RENDERINGS_DIR = None

    def mixin_test_setup(self, ext):
        """Set up for unittests.

        Parameters
        ----------
        :param str ext: 'png' or 'jpg', the extension for the image type
          for which you wish to setup a test.
        """
        # implement this test setup as a method that is explicitly
        # called rather than trying to use setUp from unittest.TestCase
        # because we don't want to require users to call super in their
        # setUp methods.
        self.pdf_renderer = self.PDF_RENDERER

        self.pdf_path = os.path.join(
            settings.TEST_DATA_DIR,
            'pdfrenderer/paper.pdf')
        self.pdf_num_pages = 6
        self.pdf_rendered_page_template = \
            'paper.pdf-dpi100-page{page_num:04d}.{ext}'

        # add random bits to the path so that separate instances
        # of this test writing in parallel don't collide.
        self.tmp_output_dir = tempfile.mkdtemp()

        self.expected_dir_structure = [
            os.path.join(
                self.tmp_output_dir,
                'paper.pdf-images',
                self.pdf_renderer.RENDERING_ENGINE_NAME,
                'dpi{}'.format(settings.DEFAULT_INFERENCE_DPI),
                '_SUCCESS')
        ]
        self.expected_dir_structure.extend([
            os.path.join(
                self.tmp_output_dir,
                'paper.pdf-images/',
                self.pdf_renderer.RENDERING_ENGINE_NAME,
                'dpi{}'.format(settings.DEFAULT_INFERENCE_DPI),
                self.pdf_rendered_page_template.format(
                    page_num=i, ext=ext))
            for i in range(1, 7)
        ])

    def mixin_test_teardown(self):
        """Tear down for unittests."""
        shutil.rmtree(self.tmp_output_dir)

    @contextlib.contextmanager
    def setup_and_teardown(self, ext):
        """Setup and tear down resources for a test as a context manager.

        Parameters
        ----------
        :param str ext: either 'png' or 'jpg', the type of image for
        which you want to write the test.
        """
        try:
            self.mixin_test_setup(ext=ext)
            yield
        finally:
            self.mixin_test_teardown()

    def _test_render_image_ext(self, ext):
        """Test the render method with a png extension."""
        self.pdf_renderer.render(
            pdf_path=self.pdf_path,
            output_dir=self.tmp_output_dir,
            ext=ext,
            check_retcode=True)
        # check that all and only the expected paths are in the output
        # dir
        output_dir_paths = [
            os.path.join(dir_path, file_name)
            for dir_path, dir_names, file_names in os.walk(
                    self.tmp_output_dir)
            for file_name in file_names
        ]
        self.assertEqual(
            sorted(output_dir_paths),
            sorted(self.expected_dir_structure))
        # since it's a little complicated to debug bad renderings,
        # provide a useful help message.
        bad_render_help_msg = (
            "\n"
            "\n HINT!: Use the render method on {pdf_renderer} to generate"
            "\n   and inspect renderered output, and if the rendered"
            "\n   output looks good move it into "
            "\n   ``{renderings_dir}`` in place of"
            "\n   the existing files. If using docker you'll need to run"
            "\n   the following command after mounting ``/tmp`` as a volume:"
            "\n"
            "\n   python3 -c 'from deepfigures.extraction import renderers;"
                         " renderers.{pdf_renderer}().render("
                             "\"tests/data/pdfrenderer/paper.pdf\","
                            " \"/tmp/\","
                            " ext=\"{ext}\","
                            " use_cache=False)'".format(
                                renderings_dir=self.MANUALLY_INSPECTED_RENDERINGS_DIR,
                                pdf_renderer=self.pdf_renderer.__class__.__name__,
                                ext=ext))

        # compare the renderings against manually inspected renderings
        for path in output_dir_paths:
            if path[-3:] == ext:
                test_image = imread(path)
                reference_image = imread(
                    os.path.join(
                        self.MANUALLY_INSPECTED_RENDERINGS_DIR,
                        os.path.split(path)[-1]))
                # test that the average absolute difference between the pixels is
                # less than 5.
                self.assertLess(
                    np.sum(np.abs(test_image - reference_image)) / test_image.size, 5.0,
                    msg=bad_render_help_msg)

    def test_render_png(self):
        """Test the render method with a png extension."""
        ext = 'png'
        with self.setup_and_teardown(ext=ext):
            self._test_render_image_ext(ext=ext)

    def test_render_jpg(self):
        """Test the render method with a jpg extension."""
        ext = 'jpg'
        with self.setup_and_teardown(ext=ext):
            self._test_render_image_ext(ext=ext)

    def test_uses_cache(self):
        """Test that the rendered uses existing copies of the files."""
        ext = 'png'
        with self.setup_and_teardown(ext=ext):
            self.pdf_renderer.render(
                pdf_path=self.pdf_path,
                output_dir=self.tmp_output_dir,
                ext=ext,
                check_retcode=True)
            output_dir_paths = [
                os.path.join(dir_path, file_name)
                for dir_path, dir_names, file_names in os.walk(
                        self.tmp_output_dir)
                for file_name in file_names
            ]
            mtimes = {}
            for path in output_dir_paths:
                mtimes[path] = os.path.getmtime(path)
            time.sleep(1)
            # render the PDF again and verify the mtimes haven't changed
            self.pdf_renderer.render(
                pdf_path=self.pdf_path,
                output_dir=self.tmp_output_dir,
                ext=ext,
                check_retcode=True)
            output_dir_paths = [
                os.path.join(dir_path, file_name)
                for dir_path, dir_names, file_names in os.walk(
                        self.tmp_output_dir)
                for file_name in file_names
            ]
            for path in output_dir_paths:
                self.assertEqual(mtimes[path], os.path.getmtime(path))

    def test_busts_cache(self):
        """Test that passing use_cache False busts the cache."""
        ext = 'png'
        with self.setup_and_teardown(ext=ext):
            self.pdf_renderer.render(
                pdf_path=self.pdf_path,
                output_dir=self.tmp_output_dir,
                ext=ext,
                check_retcode=True)
            output_dir_paths = [
                os.path.join(dir_path, file_name)
                for dir_path, dir_names, file_names in os.walk(
                        self.tmp_output_dir)
                for file_name in file_names
            ]
            mtimes = {}
            for path in output_dir_paths:
                mtimes[path] = os.path.getmtime(path)
            # render the PDF again and verify the mtimes have changed
            time.sleep(1)
            self.pdf_renderer.render(
                pdf_path=self.pdf_path,
                output_dir=self.tmp_output_dir,
                ext=ext,
                use_cache=False,
                check_retcode=True)
            output_dir_paths = [
                os.path.join(dir_path, file_name)
                for dir_path, dir_names, file_names in os.walk(
                        self.tmp_output_dir)
                for file_name in file_names
            ]
            for path in output_dir_paths:
                if path[-3:] == 'png' or path[-8:] == '_SUCCESS':
                    self.assertNotEqual(
                        mtimes[path],
                        os.path.getmtime(path),
                        msg="{path} mtime did not change.".format(path=path))


class GhostScriptRendererTest(
        PDFRendererSubclassTestMixin,
        unittest.TestCase):
    """Test deepfigures.renderers.GhostScriptRenderer."""
    PDF_RENDERER = renderers.GhostScriptRenderer()
    MANUALLY_INSPECTED_RENDERINGS_DIR = os.path.join(
        settings.TEST_DATA_DIR,
        'pdfrenderer/ghostscript-renderings/')
