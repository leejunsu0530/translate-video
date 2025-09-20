from setuptools import setup, find_packages  # type: ignore
from translatevideo.version import __version__

setup(
    name="translate-video",            # Package name
    version=__version__,             # Version number
    packages=find_packages(),    # Automatically find subpackages
    install_requires=[           # Dependencies
        "git+https://github.com/absadiki/pywhispercpp",

    ],
    author="leejunsu0530",
    author_email="leejunsu0530@gmail.com",
    description="translate video images and audio",
    long_description=open("README.md", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/leejunsu0530/translate-video",
    # classifiers=[                # Metadata for PyPI
        # "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License"
    # ],
    python_requires=">=3.11",     # Minimum Python version
)
