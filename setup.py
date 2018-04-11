#!/usr/bin/python

import os
import setuptools

requirements_file = os.path.join(
    os.path.dirname(__file__),
    'requirements.txt')
requirements = open(requirements_file).read().split('\n')
requirements = [r for r in requirements if not '-e' in r]

setuptools.setup(
    name='deepfigures-open',
    version='0.0.1',
    url='http://github.com/allenai/deepfigures-open',
    packages=setuptools.find_packages(),
    install_requires=requirements,
    tests_require=[],
    zip_safe=False,
    test_suite='py.test',
    entry_points='',
    cffi_modules=['deepfigures/utils/stringmatch/stringmatch_builder.py:ffibuilder']
)
