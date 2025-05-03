from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="phyto_nas_tsc",
    version="0.1.8",
    author="Carmely Reiska",
    author_email="reiskacarmely@gmail.com",
    description="Phyto Neural Architecture Search for Time Series Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carmelyr/Phyto-NAS-T",
    packages=find_packages(),
    package_data={
        'phyto_nas_tsc': ['data/*.csv'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.0.0",
        "torch>=1.8.0",
        "pytorch-lightning>=1.4.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.0.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0",
            "flake8>=3.9.0"
        ]
    },
    include_package_data=True,
)