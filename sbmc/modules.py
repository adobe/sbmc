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
"""Auxiliary modules used in the main models."""
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import ttools

from . import functions as funcs

__all__ = ["ConvChain", "Autoencoder", "KernelApply", "ProgressiveKernelApply"]

LOG = ttools.get_logger(__name__)


class ConvChain(nn.Module):
    """A simple stack of convolution layers.

    Args:
        ninputs(int): number of input channels.
        noutputs(int): number of output channels.
        ksize(int): size of all the convolution kernels.
        width(int): number of channels per intermediate layer.
        depth(int): number of intermadiate layers.
        stride(int): stride of the convolution.
        pad(bool): if True, maintains spatial resolution by 0-padding,
            otherwise keep only the valid part.
        normalize(bool): applies normalization if True.
        normalization_type(str): either batch or instance.
        output_type(str): one of linear, relu, leaky_relu, tanh, elu.
        activation(str): one of relu, leaky_relu, tanh, elu.
        weight_norm(bool): applies weight normalization if True.
    """
    def __init__(self, ninputs, noutputs, ksize=3, width=64, depth=3, stride=1,
                 pad=True, normalize=False, normalization_type="batch",
                 output_type="linear", activation="relu", weight_norm=True):
        super(ConvChain, self).__init__()

        if depth <= 0:
            LOG.error("ConvChain should have non-negative depth.")
            raise ValueError("negative network depth.")

        if pad:
            padding = ksize//2
        else:
            padding = 0

        layers = []
        for d in range(depth-1):
            if d == 0:
                _in = ninputs
            else:
                _in = width
            layers.append(
                ConvChain._ConvBNRelu(_in, ksize, width, normalize=normalize,
                                      normalization_type=normalization_type,
                                      padding=padding, stride=stride,
                                      activation=activation,
                                      weight_norm=weight_norm))

        # Last layer
        if depth > 1:
            _in = width
        else:
            _in = ninputs

        conv = nn.Conv2d(_in, noutputs, ksize, bias=True, padding=padding)
        if weight_norm:
            conv = nn.utils.weight_norm(conv)
        conv.bias.data.zero_()
        if output_type == "elu" or output_type == "softplus":
            nn.init.xavier_uniform_(
                conv.weight.data, nn.init.calculate_gain("relu"))
        else:
            nn.init.xavier_uniform_(
                conv.weight.data, nn.init.calculate_gain(output_type))
        layers.append(conv)

        # Rename layers
        for im, m in enumerate(layers):
            if im == len(layers)-1:
                name = "prediction"
            else:
                name = "layer_{}".format(im)
            self.add_module(name, m)

        if output_type == "linear":
            pass
        elif output_type == "relu":
            self.add_module("output_activation", nn.ReLU(inplace=True))
        elif output_type == "leaky_relu":
            self.add_module("output_activation", nn.LeakyReLU(inplace=True))
        elif output_type == "sigmoid":
            self.add_module("output_activation", nn.Sigmoid())
        elif output_type == "tanh":
            self.add_module("output_activation", nn.Tanh())
        elif output_type == "elu":
            self.add_module("output_activation", nn.ELU())
        elif output_type == "softplus":
            self.add_module("output_activation", nn.Softplus())
        else:
            raise ValueError("Unknon output type '{}'".format(output_type))

    def forward(self, x):
        for m in self.children():
            x = m(x)
        return x

    class _ConvBNRelu(nn.Module):
        """Helper class that implements a simple Conv-(Norm)-Activation group.

        Args:
            ninputs(int): number of input channels.
            ksize(int): size of all the convolution kernels.
            noutputs(int): number of output channels.
            stride(int): stride of the convolution.
            pading(int): amount of 0-padding.
            normalize(bool): applies normalization if True.
            normalization_type(str): either batch or instance.
            activation(str): one of relu, leaky_relu, tanh, elu.
            weight_norm(bool): if True applies weight normalization.
        """
        def __init__(self, ninputs, ksize, noutputs, normalize=False,
                     normalization_type="batch", stride=1, padding=0,
                     activation="relu", weight_norm=True):
            super(ConvChain._ConvBNRelu, self).__init__()

            if activation == "relu":
                act_fn = nn.ReLU
            elif activation == "leaky_relu":
                act_fn = nn.LeakyReLU
            elif activation == "tanh":
                act_fn = nn.Tanh
            elif activation == "elu":
                act_fn = nn.ELU
            else:
                LOG.error("Incorrect activation %s", activation)
                raise ValueError("activation should be one of: "
                                 "relu, leaky_relu, tanh, elu")

            if normalize:
                print("nrm", normalization_type)
                conv = nn.Conv2d(ninputs, noutputs, ksize,
                                 stride=stride, padding=padding, bias=False)
                if normalization_type == "batch":
                    nrm = nn.BatchNorm2d(noutputs)
                elif normalization_type == "instance":
                    nrm = nn.InstanceNorm2D(noutputs)
                else:
                    LOG.error("Incorrect normalization %s", normalization_type)
                    raise ValueError(
                        "Unkown normalization type {}".format(
                            normalization_type))
                nrm.bias.data.zero_()
                nrm.weight.data.fill_(1.0)
                self.layer = nn.Sequential(conv, nrm, act_fn())
            else:
                conv = nn.Conv2d(ninputs, noutputs, ksize,
                                 stride=stride, padding=padding)
                if weight_norm:
                    conv = nn.utils.weight_norm(conv)
                conv.bias.data.zero_()
                self.layer = nn.Sequential(conv, act_fn())

            if activation == "elu":
                nn.init.xavier_uniform_(
                    conv.weight.data, nn.init.calculate_gain("relu"))
            else:
                nn.init.xavier_uniform_(
                    conv.weight.data, nn.init.calculate_gain(activation))

        def forward(self, x):
            out = self.layer(x)
            return out


class Autoencoder(nn.Module):
    """A U-net style autoencoder.

    Args:
        ninputs(int): number of input channels.
        noutputs(int): number of output channels.
        ksize(int): size of all the convolution kernels.
        width(int): number of channels per intermediate layer at the finest
            scale.
        num_levels(int): number of spatial scales.
        num_convs(int): number of conv layers per scale.
        max_width(int): max number of features per conv layer.
        increase_factor(float): each coarsest scale increases the number of
            feature channels by this factor, up to `max_width`.
        normalize(bool): applies normalization if True.
        normalization_type(str): either batch or instance.
        output_type(str): one of linear, relu, leaky_relu, tanh, elu.
        activation(str): one of relu, leaky_relu, tanh, elu.
    """
    def __init__(self, ninputs, noutputs, ksize=3, width=64, num_levels=3,
                 num_convs=2, max_width=512, increase_factor=1.0,
                 normalize=False, normalization_type="batch",
                 output_type="linear",
                 activation="relu", pooling="max"):
        super(Autoencoder, self).__init__()

        next_level = None
        for lvl in range(num_levels-1, -1, -1):
            n_in = min(int(width*(increase_factor)**(lvl-1)), max_width)
            w = min(int(width*(increase_factor)**(lvl)), max_width)
            n_us = min(int(width*(increase_factor)**(lvl+1)), max_width)
            n_out = w
            o_type = activation

            if lvl == 0:
                n_in = ninputs
                o_type = output_type
                n_out = noutputs
            elif lvl == num_levels-1:
                n_us = None

            next_level = Autoencoder._Level(
                n_in, n_out, next_level=next_level, num_us=n_us,
                ksize=ksize, width=w, num_convs=num_convs,
                output_type=o_type, normalize=normalize,
                normalization_type=normalization_type,
                activation=activation, pooling=pooling)

        self.add_module("net", next_level)

    def forward(self, x):
        return self.net(x)

    class _Level(nn.Module):
        """One scale of the autoencoder processor.

        Args:
            num_inputs(int): number of input channels.
            num_outputs(int): number of output channels.
            next_level(Autoencoder._Level or None): the coarser level after
                this one, or None if this is the coarsest level.
            num_us(int): number of features in the upsampling branch.
            ksize(int): size of all the convolution kernels.
            width(int): number of channels per intermediate layer at the finest
                scale.
            num_convs(int): number of conv layers per scale.
            output_type(str): one of linear, relu, leaky_relu, tanh, elu.
            normalize(bool): applies normalization if True.
            normalization_type(str): either batch or instance.
            pooling(str): type of downsampling operator: "max", "average" or
                "conv".
            activation(str): one of relu, leaky_relu, tanh, elu.
        """
        def __init__(self, num_inputs, num_outputs, next_level=None,
                     num_us=None, ksize=3, width=64, num_convs=2,
                     output_type="linear", normalize=True,
                     normalization_type="batch", pooling="max",
                     activation="relu"):
            super(Autoencoder._Level, self).__init__()

            self.is_last = (next_level is None)

            if self.is_last:
                self.left = ConvChain(
                    num_inputs, num_outputs, ksize=ksize, width=width,
                    depth=num_convs, stride=1, pad=True,
                    normalize=normalize, normalization_type=normalization_type,
                    output_type=output_type)
            else:
                assert num_us is not None

                self.left = ConvChain(
                    num_inputs, width, ksize=ksize, width=width,
                    depth=num_convs, stride=1, pad=True, normalize=normalize,
                    normalization_type=normalization_type,
                    output_type=activation, activation=activation)
                if pooling == "max":
                    self.downsample = nn.MaxPool2d(2, 2)
                elif pooling == "average":
                    self.downsample = nn.AvgPool2d(2, 2)
                elif pooling == "conv":
                    self.downsample = nn.Conv2d(width, width, 2, stride=2)
                else:
                    raise ValueError("unknown pooling'{}'".format(pooling))

                self.next_level = next_level
                self.right = ConvChain(
                    num_us + width, num_outputs, ksize=ksize, width=width,
                    depth=num_convs, stride=1, pad=True, normalize=normalize,
                    normalization_type=normalization_type,
                    output_type=output_type)

        def forward(self, x):
            left = self.left(x)
            if self.is_last:
                return left

            ds = self.downsample(left)
            next_level = self.next_level(ds)
            us = F.interpolate(
                next_level, size=left.shape[-2:], mode='bilinear',
                align_corners=False)
            # Concat skip connection
            concat = th.cat([us, left], 1)
            output = self.right(concat)
            return output


class KernelApply(nn.Module):
    """Applies kernel-based averaging to input tensor.

    Args:
        softmax(bool): if True, the kernels are softmax-normalized with respect
            to the output pixel, i.e. the contributions to a pixel sum to one.
        splat(bool): if True, the module expects splatting as input. Those are
            'transposed' into gather kernels and optionally normalized
            (`softmax`), before combining contributions.
    """
    def __init__(self, softmax=True, splat=True):
        super(KernelApply, self).__init__()
        self.softmax = softmax
        self.splat = splat

    def forward(self, data, kernels):
        """Forward pass.

        Args:
            data(th.Tensor[bs, chans, h, w]): input data to filter.
            kernels(th.Tensor[bs, k*k, h, w]): kernels to apply.
        Returns:
            output(th.Tensor[bs, chans, h, w]): filtered output.
            sum_w(th.Tensor[bs, h, w]): sum of weights (equivalent to filtering
                an all-ones tensor).
        """
        bs, k2, h, w = kernels.shape
        k = int(np.sqrt(k2))
        kernels = kernels.view(bs, k, k, h, w)
        if self.splat:
            kernels = funcs.Scatter2Gather.apply(kernels)
        if self.softmax:
            kernels = kernels.view(bs, k*k, h, w)
            kernels = F.softmax(kernels, dim=1)
            kernels = kernels.view(bs, k, k, h, w)

        output, sum_w = funcs.KernelWeighting.apply(data, kernels)
        sum_w = sum_w.unsqueeze(1)
        return output, sum_w


class ProgressiveKernelApply(nn.Module):
    """Applies progressive kernel-based averaging to input tensor.

    Args:
        splat(bool): if True, the module expects splatting as input. Those are
            'transposed' into gather kernels and optionally normalized
            (`softmax`), before combining contributions.
    """
    def __init__(self, splat=False):
        super(ProgressiveKernelApply, self).__init__()
        self.splat = splat

    def forward(self, data, kernels, sum_r, sum_w, max_w):
        """Add contributions.

        Initialization: pass sum_r=None, sum_w=None, max_w=None, this applies
            the kernels on the data. And initializes, sum_r, sum_w, max_w.
        Update: add the contributions from `data`, weighted by `kernels` onto
            `sum_r`. Also updates the running sum of weights and maximum.

        The final reconstruction, properly normalized is given by
        `sum_r/sum_w`.

            sum_r = sum_(i=1...n_updates) sum_(kernel_size) data_i * k_i,
            sum_w = sum_(i=1...n_updates) sum_(kernel_size) k_i,

        where k_i = exp(kernels_i) / sum_j(exp(kernels_j))

        max_w is used for numerical stability. We apply the softmax
        normalization to the weights iteratively.

            so k_i = exp(kernels_i - max_j(kernels_j)) / sum_l(exp(kernels_l -
                max_j(kernels_j)))

        Args:
            data(th.Tensor[bs, chans, h, w]): data to filter with the
                `kernels`.
            kernels(th.Tensor[bs, k*k, h, w]): gather/splat kernels to apply to
                `data`.
            sum_r(None or th.Tensor[bs, chans, h, w]): running sum of past
                contributions, expects None for the first call
                (initialization).
            sum_w(None or th.Tensor[bs, 1, h, w]): running sum of weights,
                expects None for the first call (initialization).
            max_w(None or th.Tensor[bs, 1, h, w]): running maximum of weights,
                expects None for the first call (initialization).

        Returns:
            sum_r(th.Tensor[bs, chans, h, w]): updated running sum of past
                contributions
            sum_w(th.Tensor[bs, 1, h, w]): updated running sum of weights
            max_w(th.Tensor[bs, 1, h, w]): updated running maximum of weights.
        """

        # Get kernel dimensions
        bs, k2, h, w = kernels.shape
        k = int(np.sqrt(k2))

        kernels = kernels.view(bs, k, k, h, w)
        # Convert to the gather representation if we have splat kernels
        if self.splat:
            kernels = funcs.Scatter2Gather.apply(kernels)
        kernels_view = kernels.view(bs, k*k, h, w)

        # Maximum of the current kernels
        kmax = kernels_view.max(1, keepdim=True)[0]

        if sum_r is None:  # first time we update
            if sum_w is not None or max_w is not None:
                LOG.error("sum_r is None, this is the initialization step: "
                          "sum_w and max_w should be None as well.")
                raise RuntimeError("all of sum_r, sum_w, max_w should be none")

            # Subtract max for numerical stability
            max_w = kmax
            kernels.sub_(max_w.unsqueeze(1))

            # Softmax weights
            kernels.exp_()

            # Computed weighted contributions
            sum_r, sum_w = funcs.KernelWeighting.apply(
                data.contiguous(), kernels.contiguous())
            sum_w = sum_w.unsqueeze(1)
        else:
            # Update previous weights's max and normalization
            new_max = th.max(kmax, max_w)

            # max_w is the old_max at this point
            scaler = th.exp(max_w - new_max)

            # Rescale running data with the new max
            sum_r = sum_r*scaler
            sum_w = sum_w*scaler
            max_w = new_max

            # softmax weights
            kernels.sub_(max_w.unsqueeze(1))
            kernels.exp_()

            # Computed weighted contributions
            new_sum_r, new_sum_w = funcs.KernelWeighting.apply(
                data.contiguous(), kernels.contiguous())
            new_sum_w = new_sum_w.unsqueeze(1)

            # Update running sums
            sum_r = sum_r + new_sum_r
            sum_w = sum_w + new_sum_w

        return sum_r, sum_w, max_w
