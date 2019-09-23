# encoding: utf-8
"""Tests for Halide extension functions."""
import unittest
import warnings

import torch as th
from torch.autograd import gradcheck 

import ttools

import sbmc.functions as funcs

LOG = ttools.get_logger(__name__)


class TestKernelWeighting(unittest.TestCase):
    def test_forward_cpu(self):
        self._forward(False)

    def test_forward_cuda(self):
        if not th.cuda.is_available():
            LOG.info("CUDA not available skipping.")
            return
        self._forward(True)

    def test_backward_cpu(self):
        self._backward(False)

    def test_backward_cuda(self):
        if not th.cuda.is_available():
            LOG.info("CUDA not available skipping.")
            return
        self._backward(True)

    def test_kernel_weighting_grad_cpu(self):
        self._kernel_weighting_grad(False)

    def test_kernel_weighting_grad_cuda(self):
        if not th.cuda.is_available():
            return
        self._kernel_weighting_grad(True)

    def _forward(self, gpu):
        bs = 4
        c = 5
        h = 16
        w = 16

        ksize = 5

        data = th.zeros(bs, c, h, w)
        weights = th.zeros(bs, ksize, ksize, h, w)

        for idx in range(1, 2):
            data.zero_()
            y, x = h//2, w//2
            data[idx, 0, y, x] = 1.4
            data[idx, 1, y, x] = 2.4
            data[idx, 2, y, x] = 3.4
            for dy in range(-(ksize//2), ksize//2 + 1):
                for dx in range(-(ksize//2), ksize//2 + 1):
                    weights.zero_()
                    kx = ksize // 2 + dx
                    ky = ksize // 2 + dy
                    weights[idx, ky, kx, y-dy, x-dx] = 0.5
                    o, s = funcs.KernelWeighting.apply(data, weights)
                    self.assertAlmostEqual(o[idx, 0, y-dy, x-dx].item(), 1.4*0.5)
                    self.assertAlmostEqual(o[idx, 1, y-dy, x-dx].item(), 2.4*0.5)
                    self.assertAlmostEqual(o[idx, 2, y-dy, x-dx].item(), 3.4*0.5)
                    self.assertAlmostEqual(s[idx, y-dy, x-dx].item(), 0.5)

    def _backward(self, gpu):
        bs = 3
        chans = 5
        h = 16
        w = 16

        x = w // 2
        y = h // 2
        device = "cpu"
        if gpu:
            device = "cuda"

        for ksize in [3, 5, 7]:
            for b in range(bs):
                for c in range(chans):
                    # Forward
                    data = th.ones(bs, chans, h, w, device=device, requires_grad=True)
                    data.data.fill_(7.0)
                    weights = th.ones(bs, ksize, ksize, h, w, device=device, requires_grad=True)
                    o, s = funcs.KernelWeighting.apply(data, weights)

                    # Backward
                    o_grad = th.zeros_like(o)
                    o_grad[b, c, x, y] = 1.1
                    o.backward(o_grad)

                    for dy in range(-(ksize//2), ksize//2 + 1):
                        for dx in range(-(ksize//2), ksize//2 + 1):
                            self.assertAlmostEqual(data.grad[b, c, y+dy, x+dx].item(), 1.1)
                            data.grad[b, c, y+dy, x+dx] = 0.0  # set to zero for next test
                    self.assertAlmostEqual(data.grad.abs().max().item(), 0.0)
                    self.assertAlmostEqual(weights.grad[b, ksize//2 + dy, ksize//2 + dx, x, y].item(), 7.0*1.1, places=3)

    def _kernel_weighting_grad(self, gpu):
        bs = 2
        c = 3
        h = 16
        w = 16

        ksize = 3

        data = 2*th.randn(bs, c, h, w, requires_grad=True)
        weights = th.randn(bs, ksize, ksize, h, w, requires_grad=False)

        if gpu:
            data = data.cuda()
            weights = weights.cuda()

        o, s = funcs.KernelWeighting.apply(data, weights)

        with warnings.catch_warnings():
            # Inputs are float, the gradient checker wants double inputs and
            # will issue a warning.
            warnings.filterwarnings(
                "ignore", message="At least one of the inputs that requires "
                "gradient is not of double precision")
            gradcheck(
                funcs.KernelWeighting.apply,
                (data, weights),
                eps=1e-4, atol=5e-2, rtol=5e-4,
            )

            data = 2*th.randn(bs, c, h, w, requires_grad=False)
            weights = th.randn(bs, ksize, ksize, h, w, requires_grad=True)
            if gpu:
                data = data.cuda()
                weights = weights.cuda()

            gradcheck(
                funcs.KernelWeighting.apply,
                (data, weights),
                eps=1e-4, atol=5e-2, rtol=5e-4,
            )


class TestScatter2Gather(unittest.TestCase):
    def test_scatter2gather_cpu(self):
        self._scatter2gather(False)

    def test_scatter2gather_cuda(self):
        if not th.cuda.is_available():
            return
        self._scatter2gather(True)

    def test_scatter2gather_grad_cpu(self):
        self._scatter2gather_grad(False)

    def test_scatter2gather_grad_cuda(self):
        if not th.cuda.is_available():
            return
        self._scatter2gather_grad(True)

    def _scatter2gather(self, gpu):
        bs = 4
        h = 32
        w = 32

        for ksize in [3, 5, 7, 9]:
            scatter = th.zeros(bs, ksize, ksize, h, w)
            for idx in range(bs):
                for y in range(h//2-ksize//2, h//2+ksize//2+1):
                    for x in range(w//2-ksize//2, w//2+ksize//2+1):
                        for ky in range(ksize):
                            for kx in range(ksize):
                                scatter.zero_()
                                dx = kx - ksize // 2
                                dy = ky - ksize // 2
                                kx2 = ksize-1 - kx
                                ky2 = ksize-1 - ky
                                scatter[idx, ky, kx, y, x] = 0.5
                                gather = funcs.Scatter2Gather.apply(scatter)
                                self.assertAlmostEqual(
                                    gather[idx, ky2, kx2, y+dy, x+dx].item(),
                                    0.5)

    def _scatter2gather_grad(self, gpu):
        bs = 2
        c = 2
        h = 32
        w = 32
        ksize = 3

        weights = th.randn(bs, ksize, ksize, h, w, requires_grad=True)
        if gpu:
            weights = weights.cuda()

        with warnings.catch_warnings():
            # Inputs are float, the gradient checker wants double inputs and
            # will issue a warning.
            warnings.filterwarnings(
                "ignore", message="At least one of the inputs that requires "
                "gradient is not of double precision")
            gradcheck(
                funcs.Scatter2Gather.apply,
                (weights, ),
                eps=1e-4, atol=5e-2, rtol=5e-4,
            )


if __name__ == "__main__":
    unittest.main()
