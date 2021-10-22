# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import re
from distutils.core import setup

from setuptools import find_packages


def find_version() -> str:
    with open('bisk/__init__.py', 'r') as f:
        version_file = f.read()
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name='bipedal-skills',
    version=find_version(),
    author='Facebook AI Research',
    author_email='jgehring@fb.com',
    url='https://facebookresearch.github.io/hsd3',
    license='MIT License',
    description='Bipedal Skills RL Benchmark',
    python_requires='>=3.7',
    install_requires=[
        'dm-control>=0.0.32',
        'gym>=0.18',
        'numpy>=1.9.0',
    ],
    packages=find_packages(),
    package_data={'bisk': ['assets/*.xml', 'assets/*.png']},
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
)
