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
"""Train a model."""
import numpy as np
import torch as th
from torch.utils.data import DataLoader

import ttools

import sbmc


LOG = ttools.get_logger(__name__)


def main(args):
    # Fix seed
    np.random.seed(0)
    th.manual_seed(0)

    # Parameterization of the dataset (shared between train/val)
    data_args = dict(spp=args.spp, mode=sbmc.TilesDataset.KPCN_MODE if
                     args.kpcn_mode else sbmc.TilesDataset.SBMC_MODE,
                     load_coords=args.load_coords,
                     load_gbuffer=args.load_gbuffer, load_p=args.load_p,
                     load_ld=args.load_ld, load_bt=args.load_bt)

    if args.randomize_spp:
        if args.bs != 1:
            LOG.error("Training with randomized spp is only valid for"
                      "batch_size=1, got %d", args.bs)
            raise RuntimeError("Incorrect batch size")
        data = sbmc.MultiSampleCountDataset(
            args.data, **data_args)
        LOG.info("Training with randomized sample count in [%d, %d]" % (
            2, args.spp))
    else:
        data = sbmc.TilesDataset(args.data, **data_args)
        LOG.info("Training with a single sample count: %dspp" % args.spp)

    if args.kpcn_mode:
        LOG.info("Model: pixel-based comparison from [Bako2017]")
        model = sbmc.KPCN(data.num_features, ksize=args.ksize)
        model_params = dict(ksize=args.ksize)
    else:
        LOG.info("Model: sample-based [Gharbi2019]")
        model = sbmc.Multisteps(data.num_features, data.num_global_features,
                                ksize=args.ksize, splat=not args.gather,
                                pixel=args.pixel)
        model_params = dict(ksize=args.ksize, gather=args.gather,
                            pixel=args.pixel)

    dataloader = DataLoader(
        data, batch_size=args.bs, num_workers=args.num_worker_threads,
        shuffle=True)

    # Validation set uses a constant spp
    val_dataloader = None
    if args.val_data is not None:
        LOG.info("Validation set with %dspp" % args.spp)
        val_data = sbmc.TilesDataset(args.val_data, **data_args)
        val_dataloader = DataLoader(
            val_data, batch_size=args.bs, num_workers=1, shuffle=False)
    else:
        LOG.info("No validation set provided")

    meta = dict(model_params=model_params, kpcn_mode=args.kpcn_mode,
                data_params=data_args)

    LOG.info("Model configuration: {}".format(model_params))

    checkpointer = ttools.Checkpointer(args.checkpoint_dir, model, meta=meta)

    interface = sbmc.SampleBasedDenoiserInterface(
        model, lr=args.lr, cuda=args.cuda)

    extras, meta = checkpointer.load_latest()

    trainer = ttools.Trainer(interface)

    # Hook-up some callbacks to the training loop
    log_keys = ["loss", "rmse"]
    trainer.add_callback(ttools.callbacks.ProgressBarCallback(log_keys))
    trainer.add_callback(ttools.callbacks.CheckpointingCallback(checkpointer))
    trainer.add_callback(ttools.callbacks.VisdomLoggingCallback(log_keys,
                                                                env=args.env,
                                                                port=args.port,
                                                                log=True,
                                                                frequency=100))
    trainer.add_callback(sbmc.DenoisingDisplayCallback(env=args.env,
                                                       port=args.port,
                                                       win="images"))

    # Launch the training
    LOG.info("Training started, 'Ctrl+C' to abort.")
    trainer.train(dataloader, num_epochs=args.num_epochs,
                  val_dataloader=val_dataloader)


if __name__ == "__main__":
    parser = ttools.BasicArgumentParser()
    parser.add_argument(
        '--spp', type=int, default=8, help="Max number of samples per pixel.")

    # Model parameters
    parser.add_argument(
        '--kpcn_mode', dest="kpcn_mode", action="store_true", default=False,
        help="if True, use the model from [Bako2017]: useful for comparison.")
    parser.add_argument(
        '--gather', dest="gather", action="store_true", default=False,
        help="if True, use gather kernels instead of splat.")
    parser.add_argument(
        '--pixel', dest="pixel", action="store_true", default=False,
        help="if True, use per-pixel model instead of samples.")
    parser.add_argument(
        '--ksize', type=int, default=21, help="Size of the kernels")

    # Data configuration
    parser.add_argument('--constant_spp', dest="randomize_spp",
                        action="store_false", default=True)

    parser.add_argument('--dont_use_coords', dest="load_coords",
                        action="store_false", default=True)
    parser.add_argument('--dont_use_gbuffer', dest="load_gbuffer",
                        action="store_false", default=True)
    parser.add_argument('--dont_use_p', dest="load_p",
                        action="store_false", default=True)
    parser.add_argument('--dont_use_ld', dest="load_ld",
                        action="store_false", default=True)
    parser.add_argument('--dont_use_bt', dest="load_bt",
                        action="store_false", default=True)

    args = parser.parse_args()
    ttools.set_logger(args.debug)
    main(args)
