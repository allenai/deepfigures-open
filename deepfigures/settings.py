"""Constants and settings for deepfigures."""

import logging
import os


logger = logging.getLogger(__name__)


# path to the deepfigures project root
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__)))

# version number for the current release
VERSION = '0.0.1'

# descriptions of the docker images deepfigures builds
DEEPFIGURES_IMAGES = {
    'cpu': {
        'tag': 'sampyash/vt_cs_6604_digital_libraries',
        'dockerfile_path': os.path.join(BASE_DIR, 'dockerfiles/cpu/Dockerfile'),
        'version_prefix': 'deepfigures_cpu_'
    },
    'gpu': {
        'tag': 'sampyash/vt_cs_6604_digital_libraries',
        'dockerfile_path': os.path.join(BASE_DIR, 'dockerfiles/gpu/Dockerfile'),
        'version_prefix': 'deepfigures_gpu_'
    }
}

# path to the directory containing all the project-level test data.
TEST_DATA_DIR = os.path.join(BASE_DIR, 'tests/data')

# settings for PDFRenderers
DEFAULT_INFERENCE_DPI = 100
DEFAULT_CROPPED_IMG_DPI = 200
BACKGROUND_COLOR = 255

# weights for the model
TENSORBOX_MODEL = {
    'save_dir': os.path.join(BASE_DIR, 'weights/'),
    'iteration': 500000
}

# paths to binary dependencies
PDFFIGURES_JAR_NAME = 'pdffigures2-assembly-0.0.12-SNAPSHOT.jar'
PDFFIGURES_JAR_PATH = os.path.join(
    BASE_DIR,
    'bin/',
    PDFFIGURES_JAR_NAME)

# PDF Rendering backend settings
DEEPFIGURES_PDF_RENDERER = 'deepfigures.extraction.renderers.GhostScriptRenderer'


# settings for data generation

# The location to temporarily store arxiv source data
ARXIV_DATA_TMP_DIR = ''
# The location to store the final output labels
ARXIV_DATA_OUTPUT_DIR = ''

# The location of the PMC open access data
PUBMED_INPUT_DIR = ''
# A directory for storing intermediate results
PUBMED_INTERMEDIATE_DIR = ''
# A directory for storing the output pubmed data
PUBMED_DISTANT_DATA_DIR = ''

# a local directory for storing the output data
LOCAL_PUBMED_DISTANT_DATA_DIR = ''
