# setup.py
from setuptools import setup, find_packages

setup(
    name="eurosat_project",
    version="0.2",
    packages=find_packages(),  # will pick up simclr, yaware, etc.
)
