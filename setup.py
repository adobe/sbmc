# encoding: utf-8
# Sample-based Monte Carlo Denoising using a Kernel-Splatting Network
# Michaël Gharbi Tzu-Mao Li Miika Aittala Jaakko Lehtinen Frédo Durand
# Siggraph 2019
#
# Copyright (c) 2019 Michaël Gharbi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Synthesizes the cpp wrapper code and builds dynamic Python extension."""
import os
import platform
import re
import setuptools
import subprocess

import torch as th


def main():
    dirname = os.path.dirname(os.path.abspath(__file__))

    try:
        import halide_pytorch as hlpt
    except ModuleNotFoundError:
        print("Required module `halide pytorch` not found. "
              "You can install the module using the following command:\n\n"
              "\t`pip install halide_pytorch`\n\n"
              "Aborting setup.")
        return

    with open(os.path.join(dirname, "sbmc", "version.py")) as fid:
        try:
            __version__, = re.findall( '__version__ = "(.*)"', fid.read() )
        except:
            raise ValueError("could not find version number")

    halide_root = os.getenv("HALIDE_DISTRIB_DIR")
    if not halide_root:
        # TODO: try local download
        print("The HALIDE_DISTRIB_DIR environment variable is not set, and"
              " Halide was not found in the default `vendor` path, would you"
              " like to download a pre-compiled distribution?")
        ret = input("yes/no?")
        if ret in ["y", "yes"]:
            pass
            return
        else:
            print("Not downloading Halide, installation aborted.")
            return

    if platform.system() == "Windows":
        raise RuntimeError("Windows platform not supported.")

    # CPU Halide extensions
    generators = [
        hlpt.HalideOp("src/scatter2gather.cpp", "scatter2gather",
                      "scatter2gather_cpu_float32"),
        hlpt.HalideOp("src/kernel_weighting.cpp", "kernel_weighting",
                      "kernel_weighting_cpu_float32"),
        hlpt.HalideOp("src/kernel_weighting.cpp", "kernel_weighting_grad",
                      "kernel_weighting_grad_cpu_float32"),
    ]

    try:
        subprocess.check_output(["nvcc", "--version"])
        print("Found an nvcc compiler: compiling the CUDA Halide extensions.")
        generators += [
            hlpt.HalideOp("src/scatter2gather.cpp", "scatter2gather",
                          "scatter2gather_cuda_float32", cuda=True),
            hlpt.HalideOp("src/kernel_weighting.cpp", "kernel_weighting",
                          "kernel_weighting_cuda_float32", cuda=True),
            hlpt.HalideOp("src/kernel_weighting.cpp", "kernel_weighting_grad",
                          "kernel_weighting_grad_cuda_float32", cuda=True),
        ]
    except FileNotFoundError:
        print("CUDA not found: not compiling the CUDA Halide extensions.")

    ext_name = "sbmc.halide_ops"
    extension = hlpt.HalidePyTorchExtension(
        halide_root, ext_name, generators=generators)

    # Build the Python extension module
    packages = ["sbmc", "sbmc.scene_generator"]
    setuptools.setup(
        name="sbmc",
        verbose=True,
        url="",
        author_email="mgharbi@adobe.com",
        author="Michaël Gharbi",
        version=__version__,
        packages=packages,
        ext_modules=[extension],
        install_requires=["torch-tools", "bridson", "pandas", "pyexr",
                          "scikit-image", "lz4", "wget", "torch==1.2.0"],
        cmdclass=dict(build_ext=hlpt.HalideBuildExtension),
    )

if __name__ == "__main__":
    main()
