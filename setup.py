#!/usr/bin/env python

from distutils.core import setup

setup(
    name='Content Rating System',
    version='0.1',
    description='Content Rating System in Polish TV',
    author='Marek Piechula',
    author_email='marpiechula@gmail.com',
    packages=['contentrating'],
    install_requires=[
        'matplotlib==1.2.1',
        'scipy==0.12.0',
        'scikit-image==0.8.2',
        'pybrain',
        'numpy==1.7.1',
        'pip install -e git://github.com/pybrain/pybrain.git'
    ]
)
