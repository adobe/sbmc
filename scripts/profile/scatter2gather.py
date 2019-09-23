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
"""Profiles the scatter2gather extension."""
import argparse
import torch as th
from torch.autograd import profiler


import sbmc.functions as funcs


def profile_scatter2gather(gpu):
    print("Profiling `Scatter2Gather`")
    bs = 4
    c = 3
    h = 128
    w = 128
    ksize = 1

    weights = th.ones(bs, ksize, ksize, h, w, requires_grad=True)

    if gpu:
        print("Running on GPU")
        weights = weights.cuda()
    else:
        print("Running on CPU")

    print("Burning 5 iterations")
    for i in range(5):
        w2 = funcs.Scatter2Gather.apply(weights)
    th.cuda.synchronize()

    print("Profiling")
    with profiler.profile(use_cuda=gpu) as prof:
        for i in range(1):
            w2 = funcs.Scatter2Gather.apply(weights)
            loss = w2.mean()
            loss.backward()
    print(prof)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--gpu", dest="gpu", action="store_true", default=False,
                      help="if True, run GPU implementation")
  args = parser.parse_args()
  profile_scatter2gather(args.gpu)
