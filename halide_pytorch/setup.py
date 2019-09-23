"""Synthesizes the cpp wrapper code and builds dynamic Python extension."""
import re
import setuptools

with open('halide_pytorch/version.py') as fid:
    try:
        __version__, = re.findall( '__version__ = "(.*)"', fid.read() )
    except:
        raise ValueError("could not find version number")

with open("README.rst", "r") as fh:
    long_description = fh.read()

packages = setuptools.find_packages(exclude=["tests"])

if __name__ == "__main__":
    # Build the Python extension module
    # packages = setuptools.find_packages()
    packages = ["halide_pytorch"]
    setuptools.setup(
        name="halide_pytorch",
        verbose=True,
        url="",
        author="Michaël Gharbi",
        author_email="mgharbi@adobe.com",
        version=__version__,
        packages=packages,
        license="MIT",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: POSIX",
        ],
    )
