from setuptools import find_packages
from setuptools import setup

setup(
    name='triago_control',
    version='0.0.1',
    packages=find_packages(
        include=('triago_control', 'triago_control.*')),
)
