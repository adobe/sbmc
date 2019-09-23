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
"""Compute metrics from a set of .exr images."""
import os
import argparse

from sbmc import evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ref", help="path to the root of the reference .exr"
                        " files")
    parser.add_argument("output", help="path to store the stats")
    parser.add_argument("--methods", nargs="+", help="list of methods to"
                        " compare, their folders are expected to sit next to"
                        " 'ref/'")
    parser.add_argument("--scenes", nargs="+", help="list of scenes to"
                        " evaluate")
    parser.add_argument("--pad", type=int, default=21, help="how many pixels"
                        " to remove on each side")
    args = parser.parse_args()

    evaluation.compute(args.ref, args.output, args.methods, args.scenes,
                       pad=args.pad)

    fname, ext = os.path.splitext(args.output)
    evaluation.stats([args.output], fname + "_stats.csv")
