#!/usr/bin/env python
# setup.py
#
# (C) 2019, Daniel Mouritzen

import os
import subprocess
import sys

from setuptools import find_packages, setup

requirements = [
    'click',
    'gin-config',
    'gym',
    'habitat',
    'loguru',
    'numpy<1.17',  # 1.17 results in deprecation warnings with TF 1.13
    'planetrl',
    'wandb',
]

test_requirements = [
    'mypy',
    'pytest',
    'pytest-flake8',
    'pytest-isort',
    'pytest-mypy',
]

# check if system has CUDA enabled GPU (method borrowed from
# https://github.com/NervanaSystems/coach/blob/master/setup.py)
p = subprocess.Popen(['command -v nvidia-smi'], stdout=subprocess.PIPE, shell=True)
out = p.communicate()[0].decode('UTF-8')
using_GPU = out != ''

if not using_GPU:
    # For linux wth no GPU, we install the Intel optimized version of TensorFlow
    if sys.platform in ['linux', 'linux2']:
        requirements.append('intel-tensorflow==1.13.1')
    else:
        requirements.append('tensorflow==1.13.1')
else:
    requirements.append('tensorflow-gpu==1.13.1')

setup(
    name='master_thesis_mouritzen',
    version='0.0.0',
    description='',
    author='Daniel Mouritzen',
    author_email='dmrtzn@gmail.com',
    url='https://github.com/uzh-rpg/master_thesis_mouritzen',
    packages=find_packages(),
    install_requires=requirements,
    setup_requires=['pytest-runner'],
    tests_require=test_requirements,
    extras_require={'test': test_requirements},
    dependency_links=[
        f'file://{os.getcwd()}/planet#egg=planetrl',
    ],
)
