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
"""Sample-Based Monte-Carlo denoising using a Kernel-Splatting Network."""
import os
import json

import numpy as np
import torch as th
import torch.nn as nn

from ttools.modules.image_operators import crop_like

from . import modules as ops


__all__ = ["Multisteps", "KPCN"]


class Multisteps(nn.Module):
    """
    Model from Sample-Based Monte-Carlo denoising using a Kernel-Splatting
    Network. [Gharbi2019] <http://groups.csail.mit.edu/graphics/rendernet/>

    Args:
        n_features(int): number of input features per sample, used to predict
            the splatting kernels.
        n_global_features(int): number of global features, also used to
            predict the splats.
        width(int): number of features per conv layer.
        embedding_width(int): number of intermediate per-sample features.
        ksize(int): spatial extent of the splatting kernel (square).
        splat(bool): if True, uses splatting kernel. Otherwise uses gather
            kernels (useful for ablation).
        nsteps(int): number of sample/pixel coordination steps. ("n" in
            Algorithm 1 in the paper)
        pixel(bool): if True, collapses samples by averaging and treat the
            resulting image as a 1spp input. Useful for ablation.
    """

    def __init__(self, n_features, n_global_features, width=128,
                 embedding_width=128, ksize=21, splat=True, nsteps=3,
                 pixel=False):
        super(Multisteps, self).__init__()

        if ksize < 3 or (ksize % 2 == 0):
            LOG.error("Kernel size should be odd and > 3.")
            raise ValueError("Kernel size should be odd and > 3.")

        if nsteps < 1:
            LOG.error("Multisteps requires at least one sample/pixel step.")
            raise ValueError("Multisteps requires at least one sample/pixel "
                             "step.")

        self.ksize = ksize
        self.splat = splat
        self.pixel = pixel
        self.width = width
        self.embedding_width = embedding_width
        self.eps = 1e-8  # for kernel normalization

        # We repeat the pixel/sample alternate processing for `nsteps` steps.
        self.nsteps = nsteps
        for step in range(self.nsteps):
            if step == 0:
                n_in = n_features + n_global_features
            else:
                n_in = self.embedding_width + width

            # 1x1 convolutions implement the per-sample transformation
            self.add_module("embedding_{:02d}".format(step),
                            ops.ConvChain(n_in, self.embedding_width,
                                          width=width, depth=3, ksize=1,
                                          pad=False))

            # U-net implements the pixel spatial propagation step
            self.add_module("propagation_{:02d}".format(step), ops.Autoencoder(
                self.embedding_width, width, num_levels=3, increase_factor=2.0,
                num_convs=3, width=width, ksize=3, output_type="leaky_relu",
                pooling="max"))

        # Final regression for the per-sample kernel (also 1x1 convs)
        self.kernel_regressor = ops.ConvChain(width + self.embedding_width,
                                              ksize*ksize, depth=3,
                                              width=width, ksize=1,
                                              activation="leaky_relu",
                                              pad=False, output_type="linear")

        # This module aggregates the sample contributions
        self.kernel_update = ops.ProgressiveKernelApply(splat=self.splat)

    def forward(self, samples):
        """Forward pass of the model.

        Args:
            data(dict) with keys:
                "radiance": (th.Tensor[bs, spp, 3, h, w]) sample radiance.
                "features": (th.Tensor[bs, spp, nf, h, w]) sample features.
                "global_features": (th.Tensor[bs, ngf, h, w]) global features.

        Returns:
            (dict) with keys:
                "radiance": (th.Tensor[bs, 3, h, w]) denoised radiance
        """
        radiance = samples["radiance"]
        features = samples["features"]
        gfeatures = samples["global_features"].to(radiance.device)

        if self.pixel:
            # Make the pixel-average look like one sample
            radiance = radiance.mean(1, keepdim=True)
            features = features.mean(1, keepdim=True)

        bs, spp, nf, h, w = features.shape

        modules = {n: m for (n, m) in self.named_modules()}

        limit_memory_usage = not self.training

        # -- Embed the samples then collapse to pixel-wise summaries ----------
        if limit_memory_usage:
            gf = gfeatures.repeat([1, 1, h, w])
            new_features = th.zeros(bs, spp, self.embedding_width, h, w)
        else:
            gf = gfeatures.repeat([spp, 1, h, w])

        for step in range(self.nsteps):
            if limit_memory_usage:
                # Go through the samples one by one to preserve memory for
                # large images
                for sp in range(spp):
                    f = features[:, sp].to(radiance.device)
                    if step == 0:  # Global features at first iteration only
                        f = th.cat([f, gf], 1)
                    else:
                        f = th.cat([f, propagated], 1)

                    f = modules["embedding_{:02d}".format(step)](f)

                    new_features[:, sp].copy_(f, non_blocking=True)

                    if sp == 0:
                        reduced = f
                    else:
                        reduced.add_(f)

                    del f
                    if th.cuda.is_available():
                        th.cuda.empty_cache()

                features = new_features
                reduced.div_(spp)
                if th.cuda.is_available():
                    th.cuda.empty_cache()
            else:
                flat = features.view([bs*spp, nf, h, w])
                if step == 0:  # Global features at first iteration only
                    flat = th.cat([flat, gf], 1)
                else:
                    flat = th.cat([flat, propagated.unsqueeze(1).repeat(
                        [1, spp, 1, 1, 1]).view(spp*bs, self.width, h, w)], 1)
                flat = modules["embedding_{:02d}".format(step)](flat)
                flat = flat.view(bs, spp, self.embedding_width, h, w)
                reduced = flat.mean(1)
                features = flat
                nf = self.embedding_width

            # Propagate spatially the pixel context
            propagated = modules["propagation_{:02d}".format(step)](reduced)

            if limit_memory_usage:
                del reduced
                if th.cuda.is_available():
                    th.cuda.empty_cache()

        # Predict kernels based on the context information and
        # the current sample's features
        sum_r, sum_w, max_w = None, None, None

        for sp in range(spp):
            f = features[:, sp].to(radiance.device)
            f = th.cat([f, propagated], 1)
            r = radiance[:, sp].to(radiance.device)
            kernels = self.kernel_regressor(f)
            if limit_memory_usage:
                if th.cuda.is_available():
                    th.cuda.empty_cache()

            # Update radiance estimate
            sum_r, sum_w, max_w = self.kernel_update(
                crop_like(r, kernels), kernels, sum_r, sum_w, max_w)
            if limit_memory_usage:
                if th.cuda.is_available():
                    th.cuda.empty_cache()

        # Normalize output with the running sum
        output = sum_r / (sum_w + self.eps)

        # Remove the invalid boundary data
        crop = (self.ksize - 1) // 2
        output = output[..., crop:-crop, crop:-crop]

        return {"radiance": output}


class KPCN(nn.Module):
    """Re-implementation of [Bako 2017].

    Kernel-Predicting Convolutional Networks for Denoising Monte Carlo
    Renderings: <http://cvc.ucsb.edu/graphics/Papers/SIGGRAPH2017_KPCN/>

    Args:
        n_in(int): number of input channels in the diffuse/specular streams.
        ksize(int): size of the gather reconstruction kernel.
        depth(int): number of conv layers in each branch.
        width(int): number of feature channels in each branch.
    """

    def __init__(self, n_in, ksize=21, depth=9, width=100):
        super(KPCN, self).__init__()

        self.ksize = ksize

        self.diffuse = ops.ConvChain(
            n_in, ksize*ksize, depth=depth, width=width, ksize=5,
            activation="relu", weight_norm=False, pad=False,
            output_type="linear")

        self.specular = ops.ConvChain(
            n_in, ksize*ksize, depth=depth, width=width, ksize=5,
            activation="relu", weight_norm=False, pad=False,
            output_type="linear")

        self.kernel_apply = ops.KernelApply(softmax=True, splat=False)

    def forward(self, data):
        """Forward pass of the model.

        Args:
            data(dict) with keys:
                "kpcn_diffuse_in":
                "kpcn_specular_in":
                "kpcn_diffuse_buffer":
                "kpcn_specular_buffer":
                "kpcn_albedo":

        Returns:
            (dict) with keys:
                "radiance":
                "diffuse":
                "specular":
        """
        # Process the diffuse and specular channels independently
        k_diffuse = self.diffuse(data["kpcn_diffuse_in"])
        k_specular = self.specular(data["kpcn_specular_in"])

        # Match dimensions
        b_diffuse = crop_like(data["kpcn_diffuse_buffer"],
                              k_diffuse).contiguous()
        b_specular = crop_like(data["kpcn_specular_buffer"],
                               k_specular).contiguous()

        # Kernel reconstruction
        r_diffuse, _ = self.kernel_apply(b_diffuse, k_diffuse)
        r_specular, _ = self.kernel_apply(b_specular, k_specular)

        # Combine diffuse/specular/albedo
        albedo = crop_like(data["kpcn_albedo"], r_diffuse)
        final_specular = th.exp(r_specular) - 1
        final_diffuse = albedo * r_diffuse
        final_radiance = final_diffuse + final_specular

        output = dict(radiance=final_radiance, diffuse=r_diffuse,
                      specular=r_specular)

        return output
