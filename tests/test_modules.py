# encoding: utf-8
"""Tests for helper PyTorch modules."""
import unittest
import warnings

import torch as th
import numpy as np

import ttools

import sbmc.modules as modules


LOG = ttools.get_logger(__name__)


class TestConvChain(unittest.TestCase):
    def test_basic(self):
        self.assertRaises(ValueError, modules.ConvChain, 3, 3, depth=0)
        self.assertRaises(ValueError, modules.ConvChain, 3, 3, depth=-1)
        self.assertRaises(ValueError, modules.ConvChain, 3, 3, output_type="randomstring")
        self.assertRaises(ValueError, modules.ConvChain, 3, 3, activation="randomstring")
        self.assertRaises(ValueError, modules.ConvChain, 3, 3, normalize=True,
                          normalization_type="randomstring")

        for nrm in [False, True]:
            net = modules.ConvChain(3, 3, depth=3, width=32, normalize=nrm)
            idx = 0
            if nrm:
                idx = 1
                print(net)
            self.assertIsInstance(net.layer_0, modules.ConvChain._ConvBNRelu)
            self.assertIsInstance(net.layer_1, modules.ConvChain._ConvBNRelu)
            self.assertIsInstance(net.prediction, th.nn.Conv2d)

            l0 = list(net.layer_0.layer.children())
            self.assertIsInstance(l0[0], th.nn.Conv2d)
            self.assertIsInstance(l0[1+idx], th.nn.ReLU)
            self.assertEqual(l0[0].kernel_size, (3, 3))
            self.assertEqual(l0[0].stride, (1, 1))
            self.assertEqual(l0[0].in_channels, 3)
            self.assertEqual(l0[0].out_channels, 32)

            l1 = list(net.layer_1.layer.children())
            self.assertIsInstance(l1[0], th.nn.Conv2d)
            self.assertIsInstance(l1[1+idx], th.nn.ReLU)
            self.assertEqual(l1[0].kernel_size, (3, 3))
            self.assertEqual(l1[0].stride, (1, 1))
            self.assertEqual(l1[0].in_channels, 32)
            self.assertEqual(l1[0].out_channels, 32)

            self.assertEqual(net.prediction.in_channels, 32)
            self.assertEqual(net.prediction.out_channels, 3)
            self.assertEqual(net.prediction.kernel_size, (3, 3))
            self.assertEqual(net.prediction.stride, (1, 1))

            # check batch norm
            if nrm:
                self.assertIsInstance(l0[1], th.nn.BatchNorm2d)
                self.assertIsInstance(l1[1], th.nn.BatchNorm2d)


class TestKernelApply(unittest.TestCase):
    def setUp(self):
        self.bs = 4
        self.c = 5
        self.h = 16
        self.w = 16
        self.ksize = 3

        self.data = th.zeros(self.bs, self.c, self.h, self.w)
        self.weights = th.zeros(self.bs, self.ksize*self.ksize, self.h,
                                self.w)

    def test_basic(self):
        y = self.h // 2
        x = self.w // 2
        cc = 0
        k = self.ksize

        val = 1.43
        self.data[0, cc, y, x] = val 
        self.weights[0, :, y, x] = 1.0

        for splat in [True, False]:
            func = modules.KernelApply(softmax=False, splat=splat)
            output, sum_w = func(self.data, self.weights)

            self.assertAlmostEqual(output[0, cc, y, x].item(), val, places=4)

            if splat:
                for dy in range(-(k//2), k//2+1):
                    for dx in range(-(k//2), k//2+1):
                        self.assertAlmostEqual(
                            output[0, cc, y+dy, x+dx].item(), val, places=4)
                        self.assertAlmostEqual(
                            sum_w[0, 0, y+dy, x+dx].item(), 1, places=4)
            else:
                self.assertAlmostEqual(sum_w[0, 0, y, x].item(), k*k, places=4)


class TestProgressiveKernelApply(unittest.TestCase):
    def setUp(self):
        self.bs = 4
        self.c = 5
        self.h = 16
        self.w = 16
        self.ksize = 3

        self.data = th.zeros(self.bs, self.c, self.h, self.w)
        self.weights = th.zeros(self.bs, self.ksize*self.ksize, self.h,
                                self.w)

    def test_init(self):
        y = self.h // 2
        x = self.w // 2
        cc = 0
        k = self.ksize

        val = 1.43
        self.data[0, cc, y, x] = val 
        self.weights[0, :, y, x] = 1.0

        for splat in [True, False]:
            func = modules.ProgressiveKernelApply(splat=splat)
            output, sum_w, max_w = func(self.data, self.weights, None, None, None)

            self.assertAlmostEqual(output[0, cc, y, x].item(), val, places=4)

            expected = np.exp(1.0)

            if splat:
                for dy in range(-(k//2), k//2+1):
                    for dx in range(-(k//2), k//2+1):
                        self.assertAlmostEqual(
                            output[0, cc, y+dy, x+dx].item(), val, places=4)
                        # self.assertAlmostEqual(
                        #     sum_w[0, 0, y+dy, x+dx].item(), 1, places=4)
            else:
                self.assertAlmostEqual(sum_w[0, 0, y, x].item(), k*k, places=4)


if __name__ == "__main__":
    unittest.main()
