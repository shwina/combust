#!/usr/bin/env python

from setuptools import setup, find_packages
setup(name='combust',
        version='0.1',
        description='',
        author='Ashwin Srinath',
        packages=find_packages(exclude=['tests']),
        install_requires = ['numpy', 'pytest'],
        )
