#!/usr/bin/env python
# metadata and dependencies are in pyproject.toml, this only builds the Cython extension
from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "fiola.oasis",
        ["fiola/oasis.pyx"],
    )
]

setup(
    ext_modules=cythonize(ext_modules),
)
