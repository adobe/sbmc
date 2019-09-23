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
"""Interface for running and training the model."""
import numpy as np

import ttools
import torch as th

from sbmc import losses
from ttools.modules.image_operators import crop_like


LOG = ttools.get_logger(__name__)


__all__ = ["SampleBasedDenoiserInterface"]


class SampleBasedDenoiserInterface(ttools.ModelInterface):
    """Training interface for Sample-based Monte Carlo Denoising.

    Args:
      model(th.nn.Module): neural network that implements the denoising
          operation.
      lr(float): learning rate for training.
      cuda(bool): if True, run on CUDA device.
    """

    def __init__(self, model, lr=1e-4, cuda=False):
        super(SampleBasedDenoiserInterface, self).__init__()

        self.device = "cpu"
        self.model = model
        self.loss_fn = losses.TonemappedRelativeMSE()
        self.rmse_fn = losses.RelativeMSE()

        if cuda:
            LOG.debug("Using CUDA")
            self.device = "cuda"
            self.model.cuda()
            self.loss_fn.cuda()
            self.rmse_fn.cuda()

        self.optimizer = th.optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, batch):
        """Runs a forward pass of the model.

        Args:
          batch(dict):

        Returns:
          output(dict):
        """
        for k in batch:
            if not batch[k].__class__ == th.Tensor:
                continue
            batch[k] = batch[k].to(self.device)
        output = self.model(batch)
        return output

    def backward(self, batch, fwd):
        self.optimizer.zero_grad()

        out = fwd["radiance"]
        tgt = crop_like(batch["target_image"], out)  # make sure sizes match

        loss = self.loss_fn(out, tgt)
        loss.backward()

        # Couple checks to pick up on outliers in the data.
        if not np.isfinite(loss.data.item()):
            LOG.error("Loss is infinite, there might be outliers in the data.")
            raise RuntimeError("Infinite loss at train time.")

        if np.isnan(loss.data.item()):
            LOG.error("NaN in the loss, there might be outliers in the data.")
            raise RuntimeError("NaN loss at train time.")

        clip = 1000
        actual = th.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
        if actual > clip:
            LOG.info("Clipped gradients {} -> {}".format(clip, actual))

        self.optimizer.step()

        with th.no_grad():
            rmse = self.rmse_fn(out, tgt)

        return {"loss": loss.item(), "rmse": rmse.item()}

    def init_validation(self):
        """Initializes the data to measure and display during validation.

        Returns:
          dict: loss, rmse are scalars, n is the number of images evaluated.
        """
        return {"loss": 0.0, "rmse": 0.0, "n": 0}

    def update_validation(self, batch, fwd, running):
        """Updates running statistics for the validation."""
        with th.no_grad():
            out = fwd["radiance"]
            tgt = crop_like(batch["target_image"], out)
            loss = self.loss_fn(out, tgt).item()
            rmse = self.rmse_fn(out, tgt).item()

        # Make sure our statistics accound for potentially varying batch size
        b = out.shape[0]

        # Update the running means
        n = running["n"] + b
        new_loss = running["loss"] - (1.0/n)*(running["loss"] - b*loss)
        new_rmse = running["rmse"] - (1.0/n)*(running["rmse"] - b*rmse)

        return {"loss": new_loss, "rmse": new_rmse, "n": n}
