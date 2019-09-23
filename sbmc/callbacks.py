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
"""Callbacks used a train time."""
import torch as th

import ttools
from ttools.modules.image_operators import crop_like


__all__ = ["DenoisingDisplayCallback"]


class DenoisingDisplayCallback(ttools.ImageDisplayCallback):
    """A callback that periodically displays denoising results.

    Shows a single batch every few training steps, as well as another
    set of images during validation.

    See :class:`ttools.ImageDisplayCallback`'s documentation for more info.
    """
    def caption(self, batch, fwd_result):
        spp = batch["spp"][0].item()
        return "vertically: %dspp, ours, target, difference" % spp

    def visualized_image(self, batch, fwd_result):
        lowspp = batch["low_spp"].detach()
        target = batch["target_image"].detach()
        output = fwd_result["radiance"].detach()

        # Make sure images have the same size
        lowspp = crop_like(lowspp, output)
        target = crop_like(target, output)

        # Assemble a display gallery
        diff = (output-target).abs()
        data = th.cat([lowspp, output, target, diff], -2)

        # Clip and tonemap
        data = th.clamp(data, 0)
        data /= 1 + data
        data = th.pow(data, 1.0/2.2)
        data = th.clamp(data, 0, 1)

        return data
