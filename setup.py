from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="torchlite",
    version="0.1.0",
    author="Ryan Tobin",
    author_email="ryantobin119@gmail.com",
    description="A lightweight, educational deep learning framework with PyTorch-like API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ryan-tobin/torchlite",
    project_urls={
        "Bug Tracker": "https://github.com/ryan-tobin/torchlite/issues",
        "Source Code": "https://github.com/ryan-tobin/torchlite",
        "Documentation": "https://github.com/ryan-tobin/torchlite/docs",
    },
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Indended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Pytohn :: 3.11",
        "Programming Language :: Pytohn :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">3.8",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.3.0",
        "Pillow>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.9",
            "mypy>=0.910",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
        "cuda": [
            "cupy>=9.0",
        ],
        "visualization": [
            "tensorboard>=2.6",
            "graphviz>=0.17",
        ],
    },
)
