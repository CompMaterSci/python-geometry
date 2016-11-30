#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages
import re
import ast

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))

_version_re = re.compile(r'__version__\s+=\s+(.*)')
with open(os.path.join(root_dir, 'python_geometry', '__init__.py'), 'r') as f:
    version = str(ast.literal_eval(_version_re.search(f.read()).group(1)))

setup(
    name='python-geometry',
    version=version,
    packages=find_packages(),
    url='https://github.com/CompMaterSci/python-geometry',
    license='BSD',
    author='Dominik Steinberger',
    author_email='dominik.steinberger@fau.de',
    description='Dealing with geometrical problems using Python.',
    install_requires=[
        'numpy'
    ],
    setup_requires=[
        'pytest-runner'
    ],
    tests_require=[
        'pytest'
    ],
    extras_require={
        'docs': [
            'sphinx',
            'sphinx_rtd_theme',
            'numpydoc'
        ]
    }
)
