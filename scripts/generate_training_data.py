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
"""Generate synthetic random scenes to be used for training."""

import uuid
import time
import argparse
from multiprocessing import Pool, JoinableQueue, cpu_count
import os
import subprocess
import shutil

import numpy as np

import ttools

import sbmc.scene_generator as scenegen


LOG = ttools.get_logger(__name__)


def _validate_render(path):
    """
    Remove all intermediate files and directories used to generate the .bin
    data.

    Args:
        path(str): path to the output directory to clean.
    """
    files = os.listdir(path)
    exts = [os.path.splitext(f)[-1] for f in files]
    exts = set(exts)
    if ".bin" not in exts:
        return False
    return True


def _clean_bin_folder(path):
    """
    Remove all intermediate files and directories used to generate the .bin
    data.

    Args:
        path(str): path to the output directory to clean.
    """
    files = os.listdir(path)
    for file in files:
        file = os.path.join(path, file)
        if os.path.splitext(file)[-1] != ".bin":
            if os.path.islink(file):
                os.remove(file)
            elif os.path.isdir(file):
                shutil.rmtree(file)
            else:
                os.remove(file)


def _random_dirname():
    """Generates a directory name for the random scene.

    Uses the host name, timestamp and a random UUID to disambiguate
    scens in a distributed rendering context.
    """
    hostname = os.uname()[1]
    date = time.strftime("%Y-%m-%d-%H%M%S", time.localtime())
    name = "%s_%s_%s" % (hostname, date, str(uuid.uuid4())[:8])
    return name


class GeneratorParams(object):
    """"""
    def __init__(self, args):
        super(GeneratorParams, self).__init__()

        self.working_dir = os.getcwd()
        self.output = args.output

        self.converter = os.path.abspath(args.obj2pbrt_exe)
        self.renderer = os.path.abspath(args.pbrt_exe)

        # if args.suncg_root is not None:
        #     self.suncg_converter = scenegen.SuncgConverter(
        #         os.path.abspath(args.suncg_root))
        # else:
        #     self.suncg_converter = None

        assets = os.path.abspath(args.assets)
        if not os.path.exists(assets):
            LOG.warning("No valid assets folder provided.")

        envmaps = os.path.join(assets, "envmaps.txt")
        textures = os.path.join(assets, "textures.txt")
        models = os.path.join(assets, "models.txt")

        self.envmaps = self._load_from_filelist(envmaps)
        self.textures = self._load_from_filelist(textures)
        self.models = self._load_from_filelist(models)

        LOG.debug("Assets in %s:", args.assets)
        LOG.debug("  - %d envmaps" % len(self.envmaps))
        LOG.debug("  - %d textures" % len(self.textures))
        LOG.debug("  - %d models" % len(self.models))

        self.gen = []
        for gen in args.generators:
            if not hasattr(scenegen, gen):
                LOG.error("Unknown generator `%s`", gen)
                raise RuntimeError("Unknown generator `%s`", gen)

            gen_args = [self.envmaps, self.textures, self.models,
                          self.converter]
        #
        #     if synth == "SunCGSynthesizer":
        #         assert args.suncg_root is not None, "SunCGSynthesizer needs a path to SunCG"
        #         synth_args.append(self.suncg_converter)
        #
            self.gen.append(getattr(scenegen, gen)(*gen_args))

    def _load_from_filelist(self, listpath):
        data = []

        if not os.path.exists(listpath):
            return data

        root = os.path.dirname(listpath)
        with open(listpath) as fid:
            for l in fid.readlines():
                path = os.path.join(root, l.strip())
                path = path.replace("./", "")  # removes relative path if any
                if os.path.exists(path):
                    data.append(path)
        return data


# class RenderingParams(object):
#     def __init__(self, args):
#         super(RenderingParams, self).__init__()
#         self.spp = args.spp
#         self.gt_spp = args.gt_spp
#         self.height = args.height
#         self.width = args.width
#         self.path_depth = args.path_depth
#         self.tile_size = args.tile_size
#
#     def __str__(self):
#         s = "RenderingParams: "
#         s += "spp = {}; ".format(self.spp)
#         s += "gt_spp = {}; ".format(self.gt_spp)
#         s += "height = {}; ".format(self.height)
#         s += "width = {}; ".format(self.width)
#         s += "path_depth = {}; ".format(self.path_depth)
#         s += "tile_size = {}".format(self.tile_size)
#         return s


def create_scene_file(q, render_queue):
    while True:
        data = q.get(block=True)

        idx = data["idx"]
        params = data["gen_params"]
        rparams = data["render_params"]

        LOG.debug("Creating scene {}".format(idx))
        np.random.seed(idx)

        # Create container
        dirname = _random_dirname()
        dst_dir = os.path.abspath(os.path.join(params.output, dirname))

        try:
            LOG.debug("Setting up folder {}".format(dst_dir))
            os.makedirs(dst_dir, exist_ok=True)
            os.makedirs(os.path.join(dst_dir, "geometry"), exist_ok=True)
        except Exception as e:
            LOG.warning(
                "Could not setup directories %s, " \
                "continuing to next scene: %s" % (dst_dir, e))
            q.task_done()
            continue
        LOG.debug("{} directory ready".format(dst_dir))

        # Randomize resolution
        scale = np.random.choice([1, 2, 4, 8])
        width = rparams["width"]*scale
        height = rparams["height"]*scale

        # Maintain the size constant despite the resolution change
        rparams["random_crop_w"] = rparams["width"]
        rparams["random_crop_h"] = rparams["height"]
        rparams["width"] = width
        rparams["height"] = height

        # parameters = {"spp": rparams.spp, "gt_spp": rparams.gt_spp, "width":
        #               width, "height": height, "path_depth":
        #               rparams.path_depth, "random_crop_x": rparams.width,
        #               "random_crop_h": rparams.height, "tile_size":
        #               rparams.tile_size}
        renderer = scenegen.Renderer(**rparams)

        scn = scenegen.Scene(renderer=renderer)

        max_attempts = 20
        attempt = 0
        try:
            gen = np.random.choice(params.gen)
            while not gen.sample(scn, dst_dir):
                attempt += 1
                LOG.warning("Sampling another Scene {}".format(gen))
                if attempt == max_attempts:
                    break

            if attempt == max_attempts:
                LOG.warning(
                    "Could not generate a scene, continuing to next seed")
                q.task_done()
                continue
        except Exception as e:
            LOG.warning(
                "Scene sampling failed at attempt {}: {}, continuing to next"
                " scene".format(attempt, e))
            q.task_done()
            continue

        try:
            scn_file = os.path.join(dst_dir, "scene.pbrt")
            with open(scn_file, 'w') as fid:
                fid.write(scn.pbrt())
        except:
            LOG.error("Failed to save .pbrt file, continuing")
            q.task_done()
            continue

        render_data = {"idx": idx, "gen_params": params, "render_params":
                       rparams, "scene_dir": dst_dir, "verbose":
                       data["verbose"], "clean": data["clean"]}
        LOG.info("Adding scene #%d to the render queue", idx)
        render_queue.put(render_data, block=False)

        q.task_done()
        continue


def render(render_queue):
    while True:
        data = render_queue.get(block=True)
        idx = data["idx"]
        params = data["gen_params"]
        rparams = data["render_params"]
        dst_dir = data["scene_dir"]
        verbose = data["verbose"]
        clean = data["clean"]
        LOG.info("Rendering scene '%s' using '%s'", dst_dir, params.renderer)

        try:
            os.chdir(dst_dir)
            if verbose:
                stderr = None
            else:
                stderr = subprocess.DEVNULL
            ret = subprocess.check_output([params.renderer, "scene.pbrt"],
                                          stderr=stderr)
            LOG.debug("Renderer output %s", ret)
        except Exception as e:
            LOG.warning("Rendering failed for scene %s: %s" % (dst_dir, e))
            render_queue.task_done()
            continue
        os.chdir(params.working_dir)

        # If scene is not rendered, delete folder
        if not _validate_render(dst_dir):
            LOG.warning(
                "Render did not complete properly, deleting %s" % dst_dir)
            shutil.rmtree(dst_dir)
            render_queue.task_done()
            continue

        # Remove pbrt scene file, geometry and folders (everything except the
        # .bins)
        if clean:
            _clean_bin_folder(dst_dir)

        LOG.info("Finished rendering scene #%d", idx)
        render_queue.task_done()


def main(args):
    ttools.set_logger(args.verbose)

    if args.width % args.tile_size != 0 or args.height % args.tile_size != 0:
        LOG.error("Block size should divide width and height.")
        raise ValueError("Block size should divide widt and height.")

    LOG.info("Starting job on worker %d of %d with %d threads" %
             (args.worker_id, args.num_workers, args.threads))

    gen_params = GeneratorParams(args)
    # render_params = RenderingParams(args)
    render_params = dict(spp=args.spp, gt_spp=args.gt_spp, height=args.height,
                         width=args.width, path_depth=args.path_depth,
                         tile_size=args.tile_size)

    scene_queue = JoinableQueue()
    render_queue = JoinableQueue()
    Pool(args.threads, create_scene_file,
                      (scene_queue, render_queue))
    Pool(args.threads, render, (render_queue, ))

    LOG.info("Generating %d random scenes", args.count)

    count = 0  # count the number of scenes generated
    while True:
        # Generate a batch of scene files (to limit memory usage, we do not
        # queue all scenes at once.
        for _ in range(min(args.batch_size, args.count)):
            idx = args.start_index + count*args.num_workers + args.worker_id
            data = {
                "idx": idx,
                "gen_params": gen_params,
                "render_params": render_params,
                "verbose": args.verbose,
                "clean": args.clean,
            }
            if args.count > 0 and count == args.count:
                break
            scene_queue.put(data, block=False)
            count += 1

        LOG.debug("Waiting for scene queue.")
        scene_queue.join()

        LOG.debug("Waiting for render queue.")
        render_queue.join()

        LOG.debug("Finished all queues.")

        # Only render up to `args.count` images
        if args.count > 0 and count == args.count:
            break

    LOG.debug("Shutting down the scene generator")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External binaries need to render the scene and convert the geometry
    parser.add_argument("pbrt_exe", help="path to the `pbrt` executable.")
    parser.add_argument("obj2pbrt_exe", help="path to PBRT's `obj2prt` "
                        "executable.")

    # Data and output folders
    parser.add_argument('assets', help="path to the assets to use.")
    parser.add_argument('output')

    # parser.add_argument('--suncg_root', type=str, default="local_data/suncg")

    # Distributed workers params
    parser.add_argument('--start_index', type=int, default=0,
                        help="index of the first scene to generate.")
    parser.add_argument('--worker_id', type=int, default=0,
                        help="id of the current worker.")
    parser.add_argument('--num_workers', type=int, default=1,
                        help="number of distributed workers in the swarm.")
    parser.add_argument('--threads', type=int,
                        default=max(cpu_count() // 2, 1),
                        help="threads to use for parallelized work.")
    parser.add_argument('--count', type=int, default=-1,
                        help="number of scenes to generate per worker.")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="number of scenes to generate before gathering"
                        " the outputs.")
    parser.add_argument("--verbose", dest="verbose", action="store_true",
                        default=False, help="Use verbose log messages.")

    # Generators configuration
    parser.add_argument('--generators', nargs="+",
                        default=["OutdoorSceneGenerator"],
                        choices=scenegen.generators.__all__,
                        help="scene generator class to use.")

    # Rendering parameters
    parser.add_argument('--spp', type=int, default=32)
    parser.add_argument('--gt_spp', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--path_depth', type=int, default=5)
    parser.add_argument('--tile_size', type=int, default=128)

    parser.add_argument('--no-clean', dest="clean", action="store_false",
                        default=True)

    main(parser.parse_args())
