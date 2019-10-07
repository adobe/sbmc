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
"""Denoise an image using a previously trained model."""

import os
import argparse
import shutil
import tempfile
import time
import pyexr
import skimage.io as skio

import torch as th
import numpy as np
from torch.utils.data import DataLoader

import ttools

import sbmc


LOG = ttools.get_logger(__name__)


def _pad(batch, out_, kpcn_mode):
    if kpcn_mode:
        pad_h = (batch["kpcn_diffuse_in"].shape[-2] - out_.shape[-2]) // 2
        pad_w = (batch["kpcn_diffuse_in"].shape[-1] - out_.shape[-1]) // 2
    else:
        pad_h = (batch["features"].shape[-2] - out_.shape[-2]) // 2
        pad_w = (batch["features"].shape[-1] - out_.shape[-1]) // 2
    pad = max(pad_h, pad_w)
    out_ = th.nn.functional.pad(out_, (pad_w, pad_w, pad_h, pad_h))
    return out_


def _split_tiles(batch, max_sz=1024, pad=256):
    h, w = batch["low_spp"].shape[-2:]
    keys = ["radiance", "features", "kpcn_diffuse_in", "kpcn_specular_in",
            "kpcn_diffuse_buffer", "kpcn_specular_buffer", "kpcn_albedo"]
    unchanged = ["global_features"]
    if h <= max_sz and w <= max_sz:  # no tiling
        tilepad = (0, 0, 0, 0)
        return [(batch, 0, h, 0, w, tilepad)]
    else:
        ret = []
        for start_y in range(0, h, max_sz-2*pad):
            pad_y = pad
            pad_y2 = pad
            if start_y == 0:
                pad_y = 0
            end_y = start_y + max_sz
            if end_y > h:
                end_y = h
                pad_y2 = 0
            for start_x in range(0, w, max_sz-2*pad):
                pad_x = pad
                pad_x2 = pad
                end_x = start_x + max_sz
                if start_x == 0:
                    pad_x = 0
                if end_x > w:
                    end_x = w
                    pad_x2 = 0
                b_ = {}
                for k in unchanged:
                    if k not in batch.keys():
                        continue
                for k in keys:
                    if k not in batch.keys():
                        continue
                    b_[k] = batch[k][..., start_y:end_y, start_x:end_x]
                    tilepad = (pad_y, pad_y2, pad_x, pad_x2)
                    ret.append((b_, start_y+pad_y, end_y-pad_y2,
                                start_x+pad_x, end_x-pad_x2, tilepad))
        return ret


def main(args):
    start = time.time()
    if not os.path.exists(args.input):
        raise ValueError("input {} does not exist".format(args.input))

    data_root = os.path.abspath(args.input)
    name = os.path.basename(data_root)
    tmpdir = tempfile.mkdtemp()
    os.symlink(data_root, os.path.join(tmpdir, name))

    LOG.info("Loading model {}".format(args.checkpoint))
    meta_params = ttools.Checkpointer.load_meta(args.checkpoint)

    LOG.info("Setting up dataloader")
    data_params = meta_params["data_params"]
    if args.spp:
        data_params["spp"] = args.spp
    data = sbmc.FullImagesDataset(tmpdir, **data_params)
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)

    LOG.info("Denoising input with {} spp".format(data_params["spp"]))

    kpcn_mode = meta_params["kpcn_mode"]
    if kpcn_mode:
        LOG.info("Using [Bako2017] denoiser.")
        model = sbmc.KPCN(data.num_features)
    else:
        model = sbmc.Multisteps(data.num_features, data.num_global_features)

    model.train(False)
    device = "cpu"
    cuda = th.cuda.is_available()
    if cuda:
        LOG.info("Using CUDA")
        model.cuda()
        device = "cuda"

    checkpointer = ttools.Checkpointer(args.checkpoint, model, None)
    extras, meta = checkpointer.load_latest()
    LOG.info("Loading latest checkpoint {}".format(
        "failed" if meta is None else "success"))

    elapsed = (time.time() - start) * 1000
    LOG.info("setup time {:.1f} ms".format(elapsed))

    LOG.info("starting the denoiser")
    for scene_id, batch in enumerate(dataloader):
        for k in batch.keys():
            batch[k] = batch[k].to(device)
        scene = os.path.basename(data.scenes[scene_id])
        LOG.info("  scene {}".format(scene))
        tile_sz = args.tile_size
        tile_pad = args.tile_pad
        batch_parts = _split_tiles(batch, max_sz=tile_sz, pad=tile_pad)
        out_radiance = th.zeros_like(batch["low_spp"])

        if cuda:
            th.cuda.synchronize()
        start = time.time()
        for part, start_y, end_y, start_x, end_x, pad_ in batch_parts:
            with th.no_grad():
                out_ = model(part)
                out_ = _pad(part, out_["radiance"], kpcn_mode)
                out_ = out_[..., pad_[0]:out_.shape[-2] -
                            pad_[1], pad_[2]:out_.shape[-1]-pad_[3]]
                out_radiance[..., start_y:end_y, start_x:end_x] = out_
        if cuda:
            th.cuda.synchronize()
        elapsed = (time.time() - start)*1000
        LOG.info("    denoising time {:.1f} ms".format(elapsed))

        out_radiance = out_radiance[0, ...].cpu().numpy().transpose([1, 2, 0])
        outdir = os.path.dirname(args.output)
        os.makedirs(outdir, exist_ok=True)
        pyexr.write(args.output, out_radiance)

        png = args.output.replace(".exr", ".png")
        skio.imsave(png, (np.clip(out_radiance, 0, 1)*255).astype(np.uint8))
    shutil.rmtree(tmpdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help="folder containing the sample .bin files.")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="folder containing the model checkpoint.")
    parser.add_argument('--output', type=str, required=True,
                        help="output destination.")
    parser.add_argument('--spp', type=int,
                        help="number of samples to use as input.")
    parser.add_argument("--tile_size", default=1024, help="We process in tiles"
                        " to limit GPU memory usage. This is the tile size.")
    parser.add_argument("--tile_pad", default=256, help="We process in tiles"
                        " to limit GPU memory usage. This is the padding"
                        " around tiles, for overlapping tiles.")
    args = parser.parse_args()
    ttools.set_logger(True)
    main(args)
