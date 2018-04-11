"""Test deepfigures.extraction.pipeline"""

import logging
import tempfile
import unittest

from deepfigures.extraction import pipeline
from deepfigures.utils import test


logger = logging.getLogger(__name__)


class TestFigureExtractionPipeline(unittest.TestCase):
    """Test ``FigureExtractionPipeline``."""

    def test_extract(self):
        """Test extract against a known extraction."""
        pdf_path = "/work/tests/data/endtoend/paper.pdf"
        figure_extractor = pipeline.FigureExtractionPipeline()

        with tempfile.TemporaryDirectory() as tmp_dir:
            figure_extraction = figure_extractor.extract(
                pdf_path, tmp_dir)

            test.test_deepfigures_json(
                self,
                expected_json='/work/tests/data/endtoend/_work_tests_data_endtoend_paper.pdf-result.json',
                actual_json=figure_extraction.deepfigures_json_path)
