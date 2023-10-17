#!/usr/bin/env python
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from distutils.command.build_ext import build_ext

with open("README.md") as file:
    read_me_description = file.read()

ext_modules = [
    Extension(
        "fiola.oasis",
        ["fiola/oasis.pyx"],
    )
]

setup(
    name="fiola",
    version="1.0",
    license="GPL-2.0",
    author="Changjia Cai, Cynthia Dong, Andrea Giovannucci",
    author_email="changjia@live.unc.edu",
    description="Real-time analysis of fluorescence imaging data",
    long_description=read_me_description,
    long_description_content_type="text/markdown",
    url="package_github_page",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        "Operating System :: Linux",
        'Intended Audience :: Researchers',
        'Topic :: Fluorescence Imaging :: Analysis Tools',
        'Intended Audience :: Researchers', 
    ],
    python_requires='>=3.7',
    ext_modules=cythonize(ext_modules),
    cmdclass={'build_ext': build_ext}
)
