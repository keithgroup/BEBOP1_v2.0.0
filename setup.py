#!/usr/bin/env python3
"""
Setup script for BEBOP1 package
"""

from setuptools import setup, find_packages
import os

# Read version from __init__.py
version = "1.0.1"
if os.path.exists("__init__.py"):
    with open("__init__.py", "r") as f:
        for line in f:
            if line.startswith("__version__"):
                version = line.split("=")[1].strip().strip("'\"")
                break

# Read long description from README
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="bebop1",
    version=version,
    description="BEBOP1: Bond Energy from Bond Order Population analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Barbaro Zulueta",
    license="MIT",
    py_modules=[
        "bebop1",
        "bebop1_params",
        "bebop1_equation",
        "read_output",
        "spatial_geom",
        "cli"
    ],
    install_requires=[
        "numpy>=1.19.0",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "bebop1=cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    keywords="chemistry, bond energy, quantum chemistry, computational chemistry",
)
