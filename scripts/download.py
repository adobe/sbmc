#!/usr/bin/env python
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
"""Download the .pbrt description and assets for our test scenes."""
import argparse
import os
import wget

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="url of the file to download.")
    parser.add_argument("dst", help="path to the output")
    args = parser.parse_args()
    args.dst = os.path.abspath(args.dst)

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    if os.path.exists(args.dst):
        print(args.dst, "already exists, skipping download.")
    else:
        print("Downloading", args.src, "to", args.dst)
        wget.download(args.src, args.dst)
