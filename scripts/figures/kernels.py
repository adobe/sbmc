import os
import argparse
import logging

import numpy as np
import torch as th
from torch.utils.data import DataLoader
from torchvision import transforms

import ttools
from ttools.modules.image_operators import crop_like

import rendernet.dataset as dset
import rendernet.modules.preprocessors as pre
import rendernet.modules.models as models
import rendernet.interfaces as interfaces
import rendernet.callbacks as cb
import rendernet.viz as viz
from sbmc.utils import make_variable

import skimage.io as skio

log = logging.getLogger("rendernet")

def main(args):
  log.info("Loading model {}".format(args.checkpoint))
  meta_params = ttools.Checkpointer.load_meta(args.checkpoint)

  spp = meta_params["spp"]
  use_p = meta_params["use_p"]
  use_ld = meta_params["use_ld"]
  use_bt = meta_params["use_bt"]
  # use_coc = meta_params["use_coc"]

  mode = "sample"
  if "DisneyPreprocessor" == meta_params["preprocessor"]:
    mode = "disney_pixel"
  elif "SampleDisneyPreprocessor" == meta_params["preprocessor"]:
    mode = "disney_sample"

  log.info("Rendering at {} spp".format(spp))

  log.info("Setting up dataloader, p:{} bt:{} ld:{}".format(use_p, use_bt, use_ld))
  data = dset.FullImageDataset(args.data, dset.RenderDataset, spp=spp, use_p=use_p, use_ld=use_ld, use_bt=use_bt)
  preprocessor = pre.get(meta_params["preprocessor"])(data)
  xforms = transforms.Compose([dset.ToTensor(), preprocessor])
  data.transform = xforms
  dataloader = DataLoader(data, batch_size=1,
                          shuffle=False, num_workers=0,
                          pin_memory=True)

  model = models.get(preprocessor, meta_params["model_params"])
  model.cuda()
  model.train(False)

  checkpointer = ttools.Checkpointer(args.checkpoint, model, None)
  extras, meta = checkpointer.load_latest()
  log.info("Loading latest checkpoint {}".format("failed" if meta is None else "success"))

  for scene_id, batch in enumerate(dataloader):
    batch_v = make_variable(batch, cuda=True)
    with th.no_grad():
      klist = []
      out_ = model(batch_v, kernel_list=klist)
    lowspp = batch["radiance"]
    target = batch["target_image"]
    out = out_["radiance"]

    cx = 70
    cy = 20
    c = 128

    target = crop_like(target, out)
    lowspp = crop_like(lowspp.squeeze(), out)
    lowspp = lowspp[..., cy:cy+c, cx:cx+c]

    lowspp = lowspp.permute(1, 2, 0, 3)
    chan, h, w, s = lowspp.shape
    lowspp = lowspp.contiguous().view(chan, h, w*s)

    sum_r = []
    sum_w = []
    max_w = []
    maxi = crop_like(klist[-1]["max_w"].unsqueeze(1), out)
    kernels = []
    updated_kernels = []
    for k in klist:
      kernels.append(th.exp(crop_like(k["kernels"], out)-maxi)) 
      updated_kernels.append(th.exp(crop_like(k["updated_kernels"], out)-maxi)) 

    out = out[..., cy:cy+c, cx:cx+c]
    target = target[..., cy:cy+c, cx:cx+c]
    updated_kernels = [k[..., cy:cy+c, cx:cx+c] for k in updated_kernels]
    kernels = [k[..., cy:cy+c, cx:cx+c] for k in kernels]

    u_kernels_im = viz.kernels2im(kernels)
    kmean = u_kernels_im.mean(0)
    kvar = u_kernels_im.std(0)

    n, h, w = u_kernels_im.shape
    u_kernels_im = u_kernels_im.permute(1, 0, 2).contiguous().view(h, w*n)

    fname = os.path.join(args.output, "lowspp.png")
    save(fname, lowspp)
    fname = os.path.join(args.output, "target.png")
    save(fname, target)
    fname = os.path.join(args.output, "output.png")
    save(fname, out)
    fname = os.path.join(args.output, "kernels_gather.png")
    save(fname, u_kernels_im)
    fname = os.path.join(args.output, "kernels_variance.png")
    print(kvar.max())
    save(fname, kvar)
    import ipdb; ipdb.set_trace()
    break

def save(fname, im):
  os.makedirs(os.path.dirname(fname), exist_ok=True)
  im = im.squeeze().cpu()
  if len(im.shape) >= 3:
    im = im.permute(1, 2, 0)
  im = th.clamp(im, 0, 1).numpy()
  skio.imsave(fname, im)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--data", required=True)
  parser.add_argument("--output", required=True)
  args = parser.parse_args()
  ttools.set_logger(True)
  main(args)
