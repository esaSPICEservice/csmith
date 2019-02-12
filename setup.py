#!/usr/bin/python
"""

@author: mcosta

"""

#from distutils.command.build import build as DisuitlsBuild
#from shutil import copyfile
#import os
#import json
#import glob
#import subprocess

from setuptools import setup, find_packages
from codecs import open
from os import path

# Needed for PyPi
#from devflow.versioning import get_python_version


here = path.abspath(path.dirname(__file__))


# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()



setup(
        name='spirec',


        version='0.1.0',


        description='Auxiliary Data Conversion System Next Generation',
        url="https://mcosta@repos.cosmos.esa.int/socci/scm/spice/adcsng.git",

        author='Marc Costa Sitja (ESA SPICE Service)',
        author_email='esa-spice@sciops.esa.int',

        # Classifiers
        classifiers=[
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 3 - Alpha',
            'Intended Audience :: SPICE and Ancillary Data Engineers, Science Operations Engineers and Developers',
            'Topic :: Geometry Pipeline :: Planetary Science :: Geometry Computations',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
        ],

        # Keywords
        keywords=['esa', 'spice', 'naif', 'planetary', 'space', 'geometry'],

        # Packages
        packages=find_packages(),

        # Include additional files into the package
        include_package_data=True,

        # Dependent packages (distributions)
        python_requires='>=3',

        # Scripts
        scripts=['bin/spirec']

      )

# cmdclass={'build': MyBuild},
# install_requires = [
#                     'configparser',
#                     'ddt',
#                     'numpy'
# setup_requires=['setuptools>=21.2.1'],
# zip_safe=False