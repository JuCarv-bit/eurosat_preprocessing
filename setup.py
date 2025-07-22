 # setup.py
from setuptools import setup, find_packages

setup(
    name="yaware_project",
    version="0.2",
    package_dir={ "": "exploratory_notebooks" },
    packages=find_packages(where="exploratory_notebooks"),
)