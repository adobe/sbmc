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
"""Denoise rendered .bin samples using [Bitterli2016] NFOR."""
import argparse
import os
import subprocess
import shutil
import tempfile

import numpy as np
import pyexr

import ttools

import sbmc

LOG = ttools.get_logger(__name__)


def bins2exr(data_dir, output, spp):
    batch_size = 1
    data = sbmc.FullImagesDataset(
        data_dir, mode=sbmc.TilesDataset.RAW_MODE, spp=spp)

    if len(data.scenes) != 1:
        LOG.error("Expected a single scene, got %d", len(data.scenes))
        raise RuntimeError("Invalid number of scenes in .bin folder.")

    im = data[0]

    os.makedirs(output, exist_ok=True)

    exr_data = {}
    half_spp = spp // 2

    # Keys to features
    labels = data.tiles_dset.labels
    albedo_k = labels.index("albedo_first_r")
    diffuse_k = labels.index("diffuse_r")
    specular_k = labels.index("specular_r")
    normal_k = labels.index("normal_first_x")
    depth_k = labels.index("depth_first")
    visibility_k = labels.index("visibility")

    df = im['features']

    diffuse_ = df[:, diffuse_k:diffuse_k+3].transpose([2, 3, 1, 0])
    specular_ = df[:, specular_k:specular_k+3].transpose([2, 3, 1, 0])
    specular_[specular_ < 0.0] = 0
    diffuse_[diffuse_ < 0.0] = 0
    radiance_ = diffuse_ + specular_
    radiance_[radiance_ < 0.0] = 0
    exr_data['color'] = radiance_.mean(-1)
    exr_data['colorA'] = radiance_[..., :half_spp].mean(-1)
    exr_data['colorB'] = radiance_[..., half_spp:].mean(-1)
    # MC estimate's variance = 1/n * var
    exr_data['colorVariance'] = radiance_.var(-1) / spp
    exr_data['diffuse'] = diffuse_.mean(-1)
    exr_data['diffuseA'] = diffuse_[..., :half_spp].mean(-1)
    exr_data['diffuseB'] = diffuse_[..., half_spp:].mean(-1)
    # MC estimate's variance = 1/n * var
    exr_data['diffuseVariance'] = diffuse_.var(-1) / spp
    exr_data['specular'] = specular_.mean(-1)
    exr_data['specularA'] = specular_[..., :half_spp].mean(-1)
    exr_data['specularB'] = specular_[..., half_spp:].mean(-1)
    # MC estimate's variance = 1/n * var
    exr_data['specularVariance'] = specular_.var(-1) / spp

    albedo_ = df[:, albedo_k:albedo_k+3].transpose([2, 3, 1, 0])
    exr_data['albedo'] = albedo_.mean(-1)
    exr_data['albedoA'] = albedo_[..., :half_spp].mean(-1)
    exr_data['albedoB'] = albedo_[..., half_spp:].mean(-1)
    exr_data['albedoVariance'] = albedo_.var(-1) / spp

    normal_ = df[:, normal_k:normal_k+3].transpose([2, 3, 1, 0])
    exr_data['normal'] = normal_.mean(-1)
    exr_data['normalA'] = normal_[..., :half_spp].mean(-1)
    exr_data['normalB'] = normal_[..., half_spp:].mean(-1)
    exr_data['normalVariance'] = normal_.var(-1) / spp

    depth_ = df[:, depth_k:depth_k+1].transpose([2, 3, 1, 0])
    depth_ = np.tile(depth_, [1, 1, 3, 1])
    scene_radius = im["scene_radius"]
    depth_[depth_ < 0.0] = 0
    depth_ /= scene_radius
    exr_data['depth'] = depth_.mean(-1)
    exr_data['depthA'] = depth_[..., :half_spp].mean(-1)
    exr_data['depthB'] = depth_[..., half_spp:].mean(-1)
    exr_data['depthVariance'] = depth_.var(-1) / spp

    for k in exr_data.keys():
        LOG.debug("  %s %.2f --- %.2f (mean = %.2f)", k, exr_data[k].min(),
                  exr_data[k].max(), exr_data[k].mean())
        pyexr.write(os.path.join(output, k + ".exr"), exr_data[k])


def main(args):
    ttools.set_logger(args.verbose)
    args.denoiser_exe = os.path.abspath(args.denoiser_exe)

    # Create a working directory for the temporary output
    LOG.debug("Saving temporary data to: %s.", args.tmp_dir)
    os.makedirs(args.tmp_dir, exist_ok=True)
    os.chdir(args.tmp_dir)

    scene_name = os.path.basename(args.scene)

    # Link folder so that the Dataset class gets the expected folder structure,
    # with one scene
    bins_folder = os.path.join(args.tmp_dir, "bins")
    os.makedirs(bins_folder, exist_ok=True)
    try:
        os.symlink(os.path.abspath(args.scene),
                   os.path.join(bins_folder, scene_name))
    except FileExistsError as e:
        LOG.warning("scene path in %s already exists.", bins_folder)

    # Convert .bin to .exr in the format expected by NFOR
    exr_folder = os.path.join(args.tmp_dir, "exr_channels")
    bins2exr(bins_folder, exr_folder, args.spp)

    LOG.info("Denoising %s", args.scene)
    if args.verbose:
        stderr = None
    else:
        stderr = subprocess.DEVNULL
    os.makedirs(args.output, exist_ok=True)
    subprocess.call([args.denoiser_exe, exr_folder+"/", args.output],
                    stderr=stderr)
    shutil.rmtree(args.tmp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "denoiser_exe", help="path to Tungsten's `denoiser` executable.")
    parser.add_argument(
        "scene", help="path to the folder containing the scene .bin files.")
    parser.add_argument("output", help="output path.")
    parser.add_argument("--tmp_dir", default=tempfile.mkdtemp(),
                        help="temporary working directory.")
    parser.add_argument("--spp", type=int, default=8,
                        help="number of samples to save.")
    parser.add_argument("--verbose", dest="verbose", action="store_true",
                        default=False, help="Use verbose log messages.")
    main(parser.parse_args())
