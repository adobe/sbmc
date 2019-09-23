# encoding: utf-8
"""Tests for loss functions."""
import unittest
import warnings

import torch as th
import numpy as np

import ttools

import sbmc.losses as losses


LOG = ttools.get_logger(__name__)


class TestRelativeMSE(unittest.TestCase):
    def setUp(self):
        self.eps = 1e-2
        self.loss_fn = losses.RelativeMSE(eps=self.eps)

    def test_basic(self):
        for sz in [[1, 3, 4, 5]]:
            im = th.zeros(*sz)
            ref = th.zeros(*sz)

            loss = self.loss_fn(im, ref).item()
            self.assertAlmostEqual(loss, 0.0)

            val = 0.34
            val2 = 0.7
            n = np.array(sz).prod()
            target = ((val-val2)**2 / (val**2 + self.eps)) * 0.5 / n

            for dx in range(n):
                ref.zero_()
                im.zero_()
                ref.view(-1)[dx] = val
                im.view(-1)[dx] = val2
                loss = self.loss_fn(im, ref).item()
                self.assertAlmostEqual(loss, target, places=4)


class TestSMAPE(unittest.TestCase):
    def setUp(self):
        self.eps = 1e-2
        self.loss_fn = losses.SMAPE(eps=self.eps)

    def test_basic(self):
        for sz in [[1, 3, 4, 5]]:
            im = th.zeros(*sz)
            ref = th.zeros(*sz)

            loss = self.loss_fn(im, ref).item()
            self.assertAlmostEqual(loss, 0.0)

            val = 0.34
            val2 = 0.7
            n = np.array(sz).prod()
            target = ((val2-val) / (val + val2 + self.eps)) / n

            for dx in range(n):
                ref.zero_()
                im.zero_()
                ref.view(-1)[dx] = val
                im.view(-1)[dx] = val2
                loss = self.loss_fn(im, ref).item()
                self.assertAlmostEqual(loss, target, places=4)


class TestTonemappedMSE(unittest.TestCase):
    def setUp(self):
        self.eps = 1e-2
        self.loss_fn = losses.TonemappedMSE(eps=self.eps)

    def test_basic(self):
        for sz in [[1, 3, 4, 5]]:
            im = th.zeros(*sz)
            ref = th.zeros(*sz)

            loss = self.loss_fn(im, ref).item()
            self.assertAlmostEqual(loss, 0.0)

            val = 0.34
            val2 = 0.7
            n = np.array(sz).prod()
            # tonemap
            val_t = val / (1 + val)
            val2_t = val2 / (1 + val2)
            target = (val_t-val2_t)**2 * 0.5 / n

            for dx in range(n):
                ref.zero_()
                im.zero_()
                ref.view(-1)[dx] = val
                im.view(-1)[dx] = val2
                loss = self.loss_fn(im, ref).item()
                self.assertAlmostEqual(loss, target, places=4)


class TestTonemappedRelativeMSE(unittest.TestCase):
    def setUp(self):
        self.eps = 1e-2
        self.loss_fn = losses.TonemappedRelativeMSE(eps=self.eps)

    def test_basic(self):
        for sz in [[1, 3, 4, 5]]:
            im = th.zeros(*sz)
            ref = th.zeros(*sz)

            loss = self.loss_fn(im, ref).item()
            self.assertAlmostEqual(loss, 0.0)

            val = 0.34
            val2 = 0.7
            n = np.array(sz).prod()
            # tonemap
            val_t = val / (1 + val)
            val2_t = val2 / (1 + val2)
            target = ((val_t-val2_t)**2 / (val_t**2 + self.eps)) * 0.5 / n

            for dx in range(n):
                ref.zero_()
                im.zero_()
                ref.view(-1)[dx] = val
                im.view(-1)[dx] = val2
                loss = self.loss_fn(im, ref).item()
                self.assertAlmostEqual(loss, target, places=4)
