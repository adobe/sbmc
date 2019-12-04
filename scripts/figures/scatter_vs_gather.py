#!/usr/bin/env python
"""Train a model."""

import os
import argparse
import logging

import numpy as np
import scipy.ndimage.filters as filters
import torch as th
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import skimage.io as skio

import ttools
from ttools.modules.image_operators import crop_like
from models import ConvChain


import rendernet.dataset as dset
import rendernet.modules.preprocessors as pre
import rendernet.modules.models as models
import rendernet.interfaces as interfaces
import rendernet.callbacks as cb
import rendernet.modules.operators as ops


LOG = logging.getLogger(__name__)

def kernels2im(kernels):
  okernels = []
  for ker in kernels:
    ker = ker.squeeze()
    ksize, _, h, w = ker.shape
    ker = ker.permute(2, 0, 3, 1).contiguous().view(1, h*ksize, w*ksize)
    okernels.append(ker)
  okernels = th.cat(okernels, 0)
  okernels -= okernels.max()
  okernels = th.exp(okernels)
  return okernels

class ForwardModel(th.nn.Module):
  """Predict kernel weights using a stack of conv layer an apply them."""
  def __init__(self, h, w, ksize=3, depth=3, width=32, scatter=False):
    super(ForwardModel, self).__init__()
    self.scatter = scatter
    self.ksize = ksize

    self.net = ConvChain(5, ksize*ksize, width=width, depth=depth, ksize=3,
                         pad=False)
    self.kernel_update = ops.KernelApply(scatter=self.scatter, softmax=True)

  def forward(self, data, coords):
    # in_ = coords.contiguous()
    in_ = th.cat([th.log10(1.0 + data/255.0), coords], 2).contiguous()
    assert in_.shape[0] == 1, "current implementation assumes batch_size = 1"
    kernels = self.net(in_.squeeze(0))
    cdata = crop_like(data.squeeze(0), kernels).contiguous()
    output, _ = self.kernel_update(cdata, kernels)

    # Average over samples
    output = th.unsqueeze(output, 0).mean(1)

    # crop output
    k = (self.ksize-1) // 2
    output = output[..., k:-k, k:-k]

    kviz = kernels.detach().clone()
    min_ = kviz.min()
    max_ = kviz.max()
    kviz = (kviz - min_) / (max_ - min_ - 1e-8) 
    bs, k2, h, w = kviz.shape
    return output, kviz.view(bs, self.ksize, self.ksize, h, w)


class OptimizerInterface(ttools.ModelInterface):
  """ """
  def __init__(self, model, cuda=True, lr=1e-4):
    self.model = model
    self.loss_fn = th.nn.MSELoss()
    self.optimizer = th.optim.Adam(self.model.parameters(), lr=lr)

    self.cuda = cuda
    if self.cuda:
      self.model.cuda()

  def forward(self, data, coords):
    if self.cuda:
      data = data.cuda()
      coords = coords.cuda()
    output, kernels = self.model(data, coords)
    return output, kernels

  def backward(self, tgt, fwd):
    if self.cuda:
      tgt = tgt.cuda()

    self.optimizer.zero_grad()
    loss = self.loss_fn(fwd, crop_like(tgt, fwd))
    loss.backward()
    self.optimizer.step()

    return loss.item()

  def init_validation(self):
    pass

  def update_validation(self, batch, fwd, running):
    pass

  def finalize_validation(self, running):
    pass


class AADataset(Dataset):
  def __init__(self, in_, ds=4, spp=1, sigma=4, size=512, outliers=True, outliers_p=0.999):
    self.outliers = outliers
    self.outliers_p = outliers_p
    # Input/Target data ----
    self.im = skio.imread(in_)
    self.im = self.im[:size, :size]

    self.h, self.w = self.im.shape[:2]
    self.h_lr = self.h // ds
    self.w_lr = self.w // ds

    self.spp = spp

    # lowpass filter
    self.sigma = sigma
    self.lp = filters.gaussian_filter(self.im, [self.sigma, self.sigma, 0])

    self.x_lr, self.y_lr = self._lr_coords(self.w_lr, self.h_lr)

  def _lr_coords(self, w_lr, h_lr):
    """Pixel coordinates in the lowres."""
    # on-grid coordinates in the lowres
    x_lr = np.arange(0, w_lr) + 0.5  # pixel center @ 0.r5
    y_lr = np.arange(0, h_lr) + 0.5
    x_lr = np.tile(np.expand_dims(x_lr, 0), [h_lr, 1])
    y_lr = np.tile(np.expand_dims(y_lr, 1), [1, w_lr])

    # add spp dimension
    x_lr = np.tile(np.expand_dims(x_lr, 0), [self.spp, 1, 1])
    y_lr = np.tile(np.expand_dims(y_lr, 0), [self.spp, 1, 1])
    return x_lr, y_lr

  def _map_to_hr(self, x_lr, y_lr):
    """Map lowres coordinates to highres grid."""
    # h_lr, w_lr = x_lr.shape
    x = x_lr * 1.0/self.w_lr
    y = y_lr * 1.0/self.h_lr
    x_hr = x * self.w
    y_hr = y * self.h
    return x_hr, y_hr

  def __len__(self):
    return 1000000000  # large number so we never run out

  def __getitem__(self, idx):
    x_lr, y_lr  = self.x_lr, self.y_lr

    # jitter coordinates in the lowres to do non-uniform sampling
    subpixel_x = np.random.uniform(-0.5, 0.5, size=x_lr.shape)
    subpixel_y = np.random.uniform(-0.5, 0.5, size=y_lr.shape)

    # subpixel_x *= 0
    # subpixel_y *= 0

    x_jitter_lr = x_lr + subpixel_x
    y_jitter_lr = y_lr + subpixel_y

    # Sample radiance in the lowress (without LP filter)
    x_jitter_hr, y_jitter_hr = self._map_to_hr(x_jitter_lr, y_jitter_lr)
    x_sampling = np.floor(x_jitter_hr).astype(np.int32)
    y_sampling = np.floor(y_jitter_hr).astype(np.int32)
    jitter_sampled = self.im[y_sampling, x_sampling, :].astype(np.float32)

    # random outliers
    if self.outliers:
      mask = (np.random.uniform(size=(self.spp, self.h_lr, self.w_lr, 1)) > self.outliers_p).astype(np.float32)
      jitter_sampled += mask*jitter_sampled*np.power(10.0, np.random.uniform(1, 2))

    # Sample radiance in the lowress (with LP filter, this is the target)
    x_hr, y_hr = self._map_to_hr(x_lr[0], y_lr[0])
    x_grid = np.floor(x_hr).astype(np.int32)
    y_grid = np.floor(y_hr).astype(np.int32)
    target = self.lp[y_grid, x_grid, :]

    # Convert to tensor
    data = th.from_numpy(jitter_sampled).permute(0, 3, 1, 2).float().contiguous()
    subpixel_coords = th.from_numpy(np.stack([subpixel_x, subpixel_y])).permute(1, 0, 2, 3).float().contiguous()
    tgt = th.from_numpy(target).permute(2, 0, 1).float().contiguous()
    return data, subpixel_coords, tgt, mask


def main(args):
  dataset = AADataset(args.input, ds=args.ds, spp=args.spp, sigma=args.sigma,
                      size=args.size, outliers=args.outliers, outliers_p=args.outliers_p)
  loader = DataLoader(dataset, batch_size=1, num_workers=4)

  # Kernel optimization -------------
  # initialize everything to a box filter
  th.manual_seed(0)
  width = 32
  depth = 3
  gather_mdl = ForwardModel(dataset.h_lr, dataset.w_lr, ksize=args.ksize, depth=depth, width=width, scatter=False)
  th.manual_seed(0)
  # beefier gather model
  b_gather_mdl = ForwardModel(dataset.h_lr, dataset.w_lr, ksize=args.ksize, depth=depth*args.depth_factor, width=width*args.width_factor, scatter=False)
  th.manual_seed(0)
  scatter_mdl = ForwardModel(dataset.h_lr, dataset.w_lr, ksize=args.ksize, depth=depth, width=width, scatter=True)

  gather_interface = OptimizerInterface(gather_mdl)
  b_gather_interface = OptimizerInterface(b_gather_mdl)
  scatter_interface = OptimizerInterface(scatter_mdl)

  # optimize the kernels
  all_losses = np.zeros((3, args.nsteps))
  step = 0
  for step, batch in enumerate(loader):
    data, subpixel_coords, target, mask = batch
    gather_result, gather_k = gather_interface.forward(data, subpixel_coords)
    loss_gather = gather_interface.backward(target, gather_result)

    b_gather_result, b_gather_k = b_gather_interface.forward(data, subpixel_coords)
    loss_b_gather = b_gather_interface.backward(target, b_gather_result)

    scatter_result, scatter_k = scatter_interface.forward(data, subpixel_coords)
    loss_scatter = scatter_interface.backward(target, scatter_result)

    all_losses[0, step] = loss_gather
    all_losses[1, step] = loss_b_gather
    all_losses[2, step] = loss_scatter

    step += 1
    if step == args.nsteps:
      break

    if step % 10 == 0:
      print("{:05d} | Gather = {:.4f}, Gather(big) = {:.4f}, Scatter = {:.4f}, gather/scatter = {:.2f} gather(big)/scatter = {:.2f}".format(step, loss_gather, loss_b_gather, loss_scatter, loss_gather/loss_scatter, loss_b_gather/loss_scatter))

  # ---------------------------------

  gather_k_viz = np.clip(kernels2im(gather_k).detach().cpu().numpy(), 0, 1)
  b_gather_k_viz = np.clip(kernels2im(b_gather_k).detach().cpu().numpy(), 0, 1)
  scatter_k_viz = np.clip(kernels2im(scatter_k).detach().cpu().numpy(), 0, 1)

  spp, h, w = gather_k_viz.shape
  gather_k_viz = gather_k_viz.reshape([spp*h, w])
  spp, b_h, b_w = b_gather_k_viz.shape
  b_gather_k_viz = b_gather_k_viz.reshape([spp*b_h, b_w])
  scatter_k_viz = scatter_k_viz.reshape([spp*h, w])

  # Conversion to numpy -----------------
  gres_c = gather_result.clone()
  mask = crop_like(mask.permute(0, 1, 4, 2, 3).max(1)[0], gather_result)
  mask = mask[0, 0]
  im = crop_like(data.mean(1), gather_result)
  im = th.clamp(im.detach()[0], 0, 255).permute(
    1, 2, 0).cpu().numpy().astype(np.uint8)
  target = crop_like(target, gather_result)
  target = th.clamp(target.detach()[0], 0, 255).permute(
    1, 2, 0).cpu().numpy().astype(np.uint8)
  gather_result = th.clamp(gather_result.detach()[0], 0, 255).permute(
    1, 2, 0).cpu().numpy().astype(np.uint8)
  b_gather_result = th.clamp(b_gather_result.detach()[0], 0, 255).permute(
    1, 2, 0).cpu().numpy().astype(np.uint8)
  scatter_result = th.clamp(scatter_result.detach()[0], 0, 255).permute(
    1, 2, 0).cpu().numpy().astype(np.uint8)


  # Output -------------
  os.makedirs(args.output, exist_ok=True)

  path = os.path.join(args.output, "0_input.png")
  skio.imsave(path, im)

  path = os.path.join(args.output, "0b_mask.png")
  skio.imsave(path, mask)

  # path = os.path.join(args.output, "1_lowpass.png")
  # skio.imsave(path, lp)
  #
  # path = os.path.join(args.output, "2_subsampled_jitter.png")
  # skio.imsave(path, jitter_sampled)
  #
  path = os.path.join(args.output, "3_gather.png")
  skio.imsave(path, gather_result)

  path = os.path.join(args.output, "3_gather_big.png")
  skio.imsave(path, b_gather_result)

  path = os.path.join(args.output, "4_target.png")
  skio.imsave(path, target)

  path = os.path.join(args.output, "5_scatter.png")
  skio.imsave(path, scatter_result)

  kdict = {
    "gather_kernels": gather_k,
    "b_gather_kernels": b_gather_k,
    "scatter_kernels": scatter_k,
  }

  h, w = mask.shape
  for k in kdict:
    kernels = crop_like(kdict[k], gres_c)
    kernels -= kernels.max()
    kernels = th.exp(kernels)

    for s in range(args.spp):
      d = os.path.join(args.output, k, "spp%02d" % s)
      os.makedirs(d, exist_ok=True)

      for y in range(h):
        for x in range(w):
          kernel = kernels[s, :, :, y, x].detach().cpu().numpy()
          if mask[y, x] == 1:
            suff = "outlier"
          else:
            suff = ""
          skio.imsave(os.path.join(d, "y%02d_x%02d%s.png" % (
            y, x, suff)), kernel)

  path = os.path.join(args.output, "6_gather_kernels.png")
  skio.imsave(path, gather_k_viz)

  path = os.path.join(args.output, "6_b_gather_kernels.png")
  skio.imsave(path, b_gather_k_viz)

  path = os.path.join(args.output, "7_scatter_kernels.png")
  skio.imsave(path, scatter_k_viz)

  ax = plt.subplot(111)
  plt.plot(all_losses.T)
  plt.legend(["gather", "gather(big)", "scatter"])
  plt.title("loss vs. step")
  plt.xlabel("optim step")
  plt.ylabel("MSE")
  ax.set_yscale("log", nonposy='clip')
  path = os.path.join(args.output, "8_loss.png")
  plt.savefig(path)
  path = os.path.join(args.output, "8_loss.pdf")
  plt.savefig(path)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('input')
  parser.add_argument('--output', required=True)
  parser.add_argument('--nsteps', type=int, default=10000)
  parser.add_argument('--ksize', type=int, default=15)
  parser.add_argument('--ds', type=int, default=8)
  parser.add_argument('--sigma', type=int, default=32)
  parser.add_argument('--spp', type=int, default=8)
  parser.add_argument('--width_factor', type=int, default=8, help="how many more params for gather")
  parser.add_argument('--depth_factor', type=int, default=2, help="how many more params for gather")
  parser.add_argument('--size', type=int, default=512)
  parser.add_argument('--outliers', dest="outliers", action="store_true")
  parser.add_argument('--outliers_p', type=float, default=0.99)
  parser.set_defaults(outliers=False)
  args = parser.parse_args()
  ttools.set_logger()
  main(args)
