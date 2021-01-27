#!/usr/bin/env python
from distutils.core import setup

setup(
      name='mnist',
      version='1.3',
      description='mnist recognition app',
      author='omegaml',
      author_email='info@omegaml.io',
      url='http://omegaml.io',
      packages=['mnist'],
      install_requires=[ # List of specific python modules names after pip install
          'dash_canvas',
          'dash_bootstrap_components',
          'scikit-image',
          'tensorflow'
            ]
      )
