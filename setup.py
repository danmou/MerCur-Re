#!/usr/bin/env python
# setup.py
#
# (C) 2019, Daniel Mouritzen

import subprocess
import sys

from setuptools import setup

requirements = [
    'click',
    'gin-config',
    'gym==0.10.9',
    'habitat>=0.1.3',
    'imageio==2.4.1',  # https://github.com/Zulko/moviepy/issues/960
    'loguru',
    'matplotlib',
    'moviepy<1.0',  # required for wandb video summaries
    'numpy<1.17',  # 1.17 results in deprecation warnings with TF 1.15
    'plotly',  # required for wandb plot summaries
    'psutil',
    'scikit-image',
    'scipy',
    'tqdm',
    'wandb',
]

test_requirements = [
    'mypy>=0.730',
    'pyflakes>=2.2.0',
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
        requirements.append('intel-tensorflow==1.15.*')
    else:
        requirements.append('tensorflow==1.15.*')
else:
    requirements.append('tensorflow-gpu==1.15.*')

setup(
    name='master_thesis_mouritzen',
    version='0.0.0',
    description='',
    author='Daniel Mouritzen',
    author_email='dmrtzn@gmail.com',
    url='https://github.com/uzh-rpg/master_thesis_mouritzen',
    packages=['project', 'configs'],
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=requirements,
    setup_requires=['pytest-runner'],
    tests_require=test_requirements,
    extras_require={'test': test_requirements},
    dependency_links=[
        'https://github.com/facebookresearch/habitat-api/tarball/master#egg=habitat',
        'https://github.com/deepmind/dm_control/tarball/master#egg=dm_control',
    ],
    entry_points={'console_scripts': ['thesis=project.cli:cli']},
)
