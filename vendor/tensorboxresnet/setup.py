#!/usr/bin/env python

import sys

from setuptools import setup, Extension, find_packages

tf_include = '/'.join(sys.executable.split('/')[:-2]) + \
                '/lib/python%d.%d/site-packages/tensorflow/include' % sys.version_info[:2]

import os
extra_defs = []
if os.uname().sysname == 'Darwin':
    extra_defs.append('-D_GLIBCXX_USE_CXX11_ABI=0')
else:
    os.environ['CC'] = 'g++'
    os.environ['CXX'] = 'g++'

setup(
    name='tensorboxresnet',
    version='0.20',
    packages=find_packages(),
    setup_requires=['Cython'],
    ext_modules=[
        Extension(
            'tensorboxresnet.utils.stitch_wrapper',
            [
                './tensorboxresnet/utils/stitch_wrapper.pyx',
                './tensorboxresnet/utils/stitch_rects.cpp',
                './tensorboxresnet/utils/hungarian/hungarian.cpp'
            ],
            language='c++',
            extra_compile_args=[
                '-std=c++11', '-Itensorbox/utils',
                '-I%s' % tf_include
            ] + extra_defs,
        )
    ]
)
