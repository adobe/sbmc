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
"""Visualize the content of a dataset of .bin sample files."""
import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
import ttools
import skimage.io as skio

from sbmc.datasets import FullImagesDataset


LOG = ttools.get_logger(__name__)


def _save(output, imname, imdata, dtype=np.uint8):
    os.makedirs(output, exist_ok=True)
    path = os.path.join(output, imname)
    skio.imsave(path, ttools.utils.tensor2image(imdata, dtype=dtype))


def main(args):
    data = FullImagesDataset(args.data_dir, spp=args.spp)

    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=4)

    LOG.info("Visualizing dataset with %d spp (gt_spp = %d)",
             data.spp, data.gt_sample_count)
    for idx, sample in enumerate(dataloader):
        LOG.info("Processing data sample %d", idx)
        im = sample["target_image"]
        LOG.info("  target radiance: %.2f -- %.2f", im.min().item(), im.max().item())
        _save(args.output, "%04d_reference.png" % idx, im)

        if not args.dump_features:
            continue

        for i, k in enumerate(data.labels):
            if k in ["diffuse_r", "specular_r", "albedo_r"]:
                k = k.split("_r")[0]
                im = sample["features"][:, :, i:i+3, ...]
            elif k in ["depth", "depth_first", "visibility", "hasHit"]:
                im = sample["features"][:, :, i:i+1, ...]
            elif k in ["normal_x", "normal_first_x"]:
                im = sample["features"][:, :, i:i+3, ...]
                im += 1.0
                im *= 0.5
            else:
                continue

            im = im.mean(1)  # average samples
            LOG.info("  %s (feature): %.2f -- %.2f", k, im.min().item(), im.max().item())
            os.makedirs(args.output, exist_ok=True)
            _save(args.output, "%04d_%s.png" % (idx, k), im, dtype=np.uint16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help="folder containing the scenes"
                        " to visualize")
    parser.add_argument('output', help="folder where the visualization will" 
                        " be saved.")
    parser.add_argument('--spp', type=int)
    parser.add_argument('--dump_features', dest="dump_features",
                        action="store_true", default=False)
    args = parser.parse_args()

    ttools.set_logger()

    main(args)
