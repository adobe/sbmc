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
"""Autograd functional extensions that use Halide as backend."""
import torch as th  # need to import PyTorch before the extension
import ttools

LOG = ttools.get_logger(__name__)
try:  # TODO(mgharbi): this is a hack so that Sphinx can compile the doc
    from . import halide_ops as ops
except Exception as e:
    LOG.error("Halide extension not loaded!\n %s", e)


def _is_cuda(*args):
    """Returns True is any of the argument is on a CUDA device, False
    otherwise."""
    for arg in args:
        if arg.is_cuda:
            return True
    return False


class Scatter2Gather(th.autograd.Function):
    """Converts (transposes) scatter kernels into gather kernels.

    Kernel weights at (x, y) for offset (dx, dy) (i.e. scatter[., dy, dx, y,
    x]) are put at gather[., -dy, -dx, y+dy, x+dx].

    Args:
      data(th.Tensor)[bs, k_h, k_w, h, w]: scatter kernel weights.

    Returns:
      (th.Tensor)[bs, k_h, k_w, h, w]: gather kernel weights.
    """
    @staticmethod
    def forward(ctx, data):
        output = data.new()
        output.resize_as_(data)
        assert len(data.shape) == 5, "data should be 5d"
        if _is_cuda(data):
            ops.scatter2gather_cuda_float32(data, output)
        else:
            ops.scatter2gather_cpu_float32(data, output)
        return output

    @staticmethod
    def backward(ctx, d_output):
        d_data = d_output.new()
        d_data.resize_as_(d_output)
        _, kh, kw, _, _ = d_data.shape
        if _is_cuda(d_output):
            ops.scatter2gather_cuda_float32(d_output, d_data)
        else:
            ops.scatter2gather_cpu_float32(d_output, d_data)
        return d_data


class KernelWeighting(th.autograd.Function):
    """Locally-weighted average of the input values using kernel weights.

    Args:
      data(th.Tensor)[bs, c, h, w]: input values to be locally averaged.
          weights(th.Tensor)[bs, k_h, k_w, h, w]: kernel weights. k_h, k_w are
          the kernel's dimensions. Channels are filtered independently.

    Returns:
      output(th.Tensor)[bs, c, h, w]: weighted average of data using weights.
          output[., c, y, x] = sum_{dx, dy} weights[., dy, dx, x, y]*data[., c,
          y+dy, x+dx].
      sum_w(th.Tensor)[bs, h, w]: sum of weights per pixel
    """
    @staticmethod
    def forward(ctx, data, weights):
        bs, c, h, w = data.shape
        output = data.new()
        sum_w = data.new()
        output.resize_as_(data)
        sum_w.resize_(bs, h, w)
        if _is_cuda(data, weights):
            ops.kernel_weighting_cuda_float32(data, weights, output, sum_w)
        else:
            ops.kernel_weighting_cpu_float32(data, weights, output, sum_w)
        ctx.save_for_backward(data, weights, sum_w)
        return output, sum_w

    @staticmethod
    def backward(ctx, d_output, d_sum_w):
        data, weights, sum_w = ctx.saved_tensors
        d_data = data.new()
        d_weights = weights.new()
        d_data.resize_as_(data)
        d_weights.resize_as_(weights)
        if _is_cuda(d_output, d_sum_w):
            ops.kernel_weighting_grad_cuda_float32(
                data, weights, sum_w, d_output, d_sum_w, d_data, d_weights)
        else:
            ops.kernel_weighting_grad_cpu_float32(
                data, weights, sum_w, d_output, d_sum_w, d_data, d_weights)
        return d_data, d_weights
