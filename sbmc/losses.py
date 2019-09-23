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
"""Loss functions and metrics."""
import torch as th


__all__ = ["RelativeMSE", "SMAPE", "TonemappedMSE", "TonemappedRelativeMSE"]


class RelativeMSE(th.nn.Module):
    """Relative Mean-Squared Error.

    :math:`0.5 * \\frac{(x - y)^2}{y^2 + \epsilon}`

    Args:
        eps(float): small number to avoid division by 0.
    """
    def __init__(self, eps=1e-2):
        super(RelativeMSE, self).__init__()
        self.eps = eps

    def forward(self, im, ref):
        """Evaluate the metric.

        Args:
            im(th.Tensor): image.
            ref(th.Tensor): reference.
        """
        mse = th.pow(im-ref, 2)
        loss = mse/(th.pow(ref, 2) + self.eps)
        loss = 0.5*th.mean(loss)
        return loss


class SMAPE(th.nn.Module):
    """Symmetric Mean Absolute error.

    :math:`\\frac{|x - y|} {|x| + |y| + \epsilon}`

    Args:
        eps(float): small number to avoid division by 0.
    """

    def __init__(self, eps=1e-2):
        super(SMAPE, self).__init__()
        self.eps = eps

    def forward(self, im, ref):
        # NOTE: the denominator is used to scale the loss, but does not
        # contribute gradients, hence the '.detach()' call.
        loss = (th.abs(im-ref) / (
            self.eps + th.abs(im.detach()) + th.abs(ref.detach()))).mean()

        return loss


class TonemappedMSE(th.nn.Module):
    """Mean-squared error on tonemaped images.

    Args:
        eps(float): small number to avoid division by 0.
    """

    def __init__(self, eps=1e-2):
        super(TonemappedMSE, self).__init__()
        self.eps = eps  # avoid division by zero

    def forward(self, im, ref):
        im = _tonemap(im)
        ref = _tonemap(ref)
        loss = th.pow(im-ref, 2)
        loss = 0.5*th.mean(loss)
        return loss


class TonemappedRelativeMSE(th.nn.Module):
    """Relative mean-squared error on tonemaped images.

    Args:
        eps(float): small number to avoid division by 0.
    """
    def __init__(self, eps=1e-2):
        super(TonemappedRelativeMSE, self).__init__()
        self.eps = eps  # avoid division by zero

    def forward(self, im, ref):
        im = _tonemap(im)
        ref = _tonemap(ref)
        mse = th.pow(im-ref, 2)
        loss = mse/(th.pow(ref, 2) + self.eps)
        loss = 0.5*th.mean(loss)
        return loss


def _tonemap(im):
    """Helper Reinhards tonemapper.

    Args:
        im(th.Tensor): image to tonemap.

    Returns:
        (th.Tensor) tonemaped image.
    """
    im = th.clamp(im, min=0)
    return im / (1+im)
