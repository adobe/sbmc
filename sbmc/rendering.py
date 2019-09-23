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
"""Collection of helper tools for the PBRT rendering scripts."""
import argparse
import os
import subprocess
import shutil
import tempfile

import ttools

from sbmc.scene_generator import Renderer

LOG = ttools.get_logger(__name__)


class BaseRenderingParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(BaseRenderingParser, self).__init__(*args, **kwargs)
        self.add_argument("pbrt_exe", help="path to the `pbrt` executable.")
        self.add_argument("scene", help="path to the .pbrt scene file.")
        self.add_argument("output", help="output path.")
        self.add_argument("--tmp_dir", default=tempfile.mkdtemp(),
                          help="temporary working directory.")
        self.add_argument("--spp", type=int, default=8,
                          help="number of samples to save.")
        self.add_argument("--tile_size", type=int, default=128, help="size of "
                          "the rendering tiles (in pixels).")
        self.add_argument("--path_depth", type=int, default=5, help="number of"
                          " bounces for the pathtracer.")
        self.add_argument("--pixel_filter", default="box", choices=["box"],
                          help="shape of the pixel filter.")
        self.add_argument("--width", type=int, default=512, help="horizontal"
                          " resolution in pixels.")
        self.add_argument("--height", type=int, default=512, help="vertical"
                          " resolution in pixels.")
        self.add_argument("--verbose", dest="verbose", action="store_true",
                          default=False, help="Use verbose log messages.")
        self.add_argument("--kalantari2015_data", dest="kalantari_data",
                          nargs=2, help="When using Kalantari's denoiser, pass"
                          " the pretrained `Weights.dat` and "
                          "`FeatureNorm.data`.")

    def parse_args(self):
        args = super(BaseRenderingParser, self).parse_args()
        args.scene = os.path.abspath(args.scene)
        args.tmp_dir = os.path.abspath(args.tmp_dir)
        args.pbrt_exe = os.path.abspath(args.pbrt_exe)
        args.output = os.path.abspath(args.output)

        if not os.path.exists(args.scene):
            raise RuntimeError("PBRT scene %s does not exist" % args.scene)

        args.scene_root = os.path.dirname(args.scene)
        args.scene_name = os.path.splitext(
            os.path.basename(args.scene_root))[0]

        # Read .pbrt scene content
        with open(args.scene) as fid:
            args.scene_desc = self._add_scene_header(fid.read(), args)

        args.scene_path = os.path.join(args.tmp_dir, "scene.pbrt")

        if not os.path.exists(args.pbrt_exe):
            LOG.error("PBRT executable not found at %s", args.pbrt_exe)
            raise RuntimeError("PBRT executable not found at %s" %
                               args.pbrt_exe)

        return args

    def _add_scene_header(self, scene_desc, args):
        return scene_desc


class PBRTRenderingParser(BaseRenderingParser):
    def _add_scene_header(self, scene_desc, args):
        """Render a .pbrt scene to a .exr image."""
        if not os.path.splitext(args.output)[-1] == ".exr":
            raise RuntimeError("Output should be the path to a .exr image.")

        renderer = Renderer(path_depth=args.path_depth,
                            spp=args.spp,
                            pixel_filter=args.pixel_filter,
                            width=args.width,
                            height=args.height,
                            integrator="path",
                            filename="scene.exr")
        return renderer.pbrt() + scene_desc


class SamplesRenderingParser(BaseRenderingParser):
    def __init__(self, *args, **kwargs):
        super(SamplesRenderingParser, self).__init__(*args, **kwargs)
        self.add_argument("--gt_spp", type=int, default=512,
                          help="number of samples for the groundtruth target.")

    def _add_scene_header(self, scene_desc, args):
        """Render a .pbrt scene to .bin files, storing the samples."""
        if not os.path.splitext(args.output)[-1] == "":
            raise RuntimeError("Output should be the path to a folder.")
        renderer = Renderer(path_depth=args.path_depth,
                            spp=args.spp,
                            gt_spp=args.gt_spp,
                            pixel_filter=args.pixel_filter,
                            width=args.width,
                            height=args.height,
                            integrator="pathrendernet",
                            filename=args.scene_name + ".exr")
        return renderer.pbrt() + scene_desc


class PBRTRenderer(object):
    def __init__(self, args):
        self.args = args
        self._setup()

    def _setup(self):
        args = self.args
        ttools.set_logger(args.verbose)

        # Create a working directory for the temporary output
        LOG.debug("Saving temporary data to: %s.", args.tmp_dir)
        os.makedirs(args.tmp_dir, exist_ok=True)

        # Write the scene file with updated header
        with open(args.scene_path, 'w') as fid:
            fid.write(args.scene_desc)

        # Link scene resources to the temp directory
        for path in os.listdir(args.scene_root):
            if path == "scene.pbrt":
                continue
            LOG.debug("Linking %s to temp directory", path)
            dst = os.path.join(args.tmp_dir, path)
            if os.path.exists(dst):
                continue
            os.symlink(os.path.join(args.scene_root, path), dst)

        if args.kalantari_data is not None:
            LOG.info("Copying [Kalantari2015]'s pretrained weights.'")
            weights, feature_norm = args.kalantari_data
            w_ = os.path.basename(weights)
            fn_ = os.path.basename(feature_norm)
            if w_ != "Weights.dat":
                LOG.error("The first argument of `kalantari2015_data` should"
                          " be `Weights.dat`, got %s", w_)
                raise ValueError("The first argument of `kalantari2015_data`"
                                 " should be `Weights.dat`")
            if fn_ != "FeatureNorm.dat":
                LOG.error("The first argument of `kalantari2015_data` should"
                          " be `FeatureNorm.dat`, got %s", fn_)
                raise ValueError("The first argument of `kalantari2015_data`"
                                 " should be `FeatureNorm.dat`")
            os.symlink(os.path.abspath(weights),
                       os.path.join(args.tmp_dir, w_))
            os.symlink(os.path.abspath(feature_norm),
                       os.path.join(args.tmp_dir, fn_))

    def __del__(self):
        # Cleanup working dir
        if os.path.exists(self.args.tmp_dir):
            LOG.debug("Deleting temp directory %s", self.args.tmp_dir)
            shutil.rmtree(self.args.tmp_dir)

    def render(self):
        args = self.args
        LOG.info("Rendering %s", self.args.scene_name)

        LOG.debug("Working from %s", args.tmp_dir)
        os.chdir(args.tmp_dir)

        # Run the rendering
        LOG.debug("Rendering samples...")
        if args.verbose:
            stderr = None
        else:
            stderr = subprocess.DEVNULL
        subprocess.check_output([args.pbrt_exe, args.scene_path],
                                stderr=stderr)

        # import ipdb; ipdb.set_trace()
        src = os.path.join(args.tmp_dir, "scene.exr")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        dst = os.path.join(args.output)
        LOG.debug("Moving %s to %s", src, dst)
        shutil.move(src, dst)


class PBRTSamplesRenderer(PBRTRenderer):
    def _setup(self):
        super(PBRTSamplesRenderer, self)._setup()
        # Create output directory for the final .bin files
        LOG.debug("Creating output directory %s", self.args.output)
        os.makedirs(self.args.output, exist_ok=True)

    def render(self):
        args = self.args
        LOG.info("Rendering %s", self.args.scene_name)

        # The sample-dumping extension to .pbrt saves files to the current
        # working directory, navigate there
        bindir = os.path.join(args.tmp_dir, "bin")
        LOG.debug("Working from %s", bindir)
        os.makedirs(bindir, exist_ok=True)
        os.chdir(bindir)

        # Run the rendering
        LOG.debug("Rendering samples...")
        if args.verbose:
            stderr = None
        else:
            stderr = subprocess.DEVNULL
        subprocess.check_output([args.pbrt_exe, args.scene_path],
                                stderr=stderr)

        # Move bin files to output folder
        for f in os.listdir(bindir):
            if os.path.splitext(f)[-1] != ".bin":
                continue
            src = os.path.join(bindir, f)
            dst = os.path.join(args.output, f)
            LOG.debug("Moving %s to %s", src, dst)
            shutil.move(src, dst)
