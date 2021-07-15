#!/usr/bin/env python
from setuptools import setup, find_packages

with open("README.md") as file:
    read_me_description = file.read()

setup(
    name="fiola",
    version="0.1",
    author="Changjia Cai, Cynthia Dong, Andrea Giovannucci",
    author_email="changjia@live.unc.edu",
    description="Real-time analysis of fluorescence imaging data",
    long_description=read_me_description,
    long_description_content_type="text/markdown",
    url="package_github_page",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: GPL-2 License',
        "Operating System :: Linux",
        'Intended Audience :: Researchers',
        'Topic :: Fluorescence Imaging :: Analysis Tools',
        'Intended Audience :: Researchers', 
    ],
    python_requires='>=3.7',
)
