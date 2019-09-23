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
"""Core scene description classes for the PBRT scene generator."""
import numpy as np

import re
import copy
import uuid
import sys

import ttools

__all__ = ["Scene", "Renderer", "Camera"]


LOG = ttools.get_logger(__name__)


class Scene():
    """PBRT Scene definition.

    y axis is up.

    Args:
        renderer(None or Renderer)
        camera(None or Camera)

    Attributes:
        renderer(Renderer): PBRT renderer config.
        lights(list of Light): lights in the scene.
        textures(list of Texture): textures used in the scene.
        materials(list of Material): materials used in the scene.
        shapes(list of Shape): geometry of the scene.
        camera(Camera): camera for the rendering.
        shapes(list of Shape): scene geometry.
    """

    def __init__(self, renderer=None, camera=None):
        self.renderer = renderer
        self.lights = []
        self.textures = []
        self.materials = []
        self.shapes = []
        self.camera = camera

    def __repr__(self):
        return "Scene with {} shapes, {} materials, {} textures, {}" \
            " lights".format(len(self.shapes), len(self.materials),
                             len(self.textures), len(self.lights))

    def pbrt(self):
        """Encodes scene as valid PBRT scene description string.

        Args:
            renderer_header(bool): if True, writes the renderer parameters.
        """
        out = ""
        if self.renderer is not None:
            out += self.renderer.pbrt()
        if self.camera is not None:
            out += self.camera.pbrt()
        out += "TransformTimes 0.0 1.0\n\n"
        out += "WorldBegin\n"
        for obj in self.lights + self.textures + self.materials + \
                self.shapes:
            out += obj.pbrt()
        out += "WorldEnd\n"
        return out


class Renderer():
    """PBRT renderer definition.

    Args:
        tile_size
        spp
        gt_spp
        path_depth
        pixel_filter
        height
        width
        random_crop_w
        random_crop_h
        integrator
        sampler
    """
    def __init__(self, **kwargs):
        self.tile_size = kwargs.get("tile_size", 128)
        self.spp = kwargs.get("spp", 16)
        self.gt_spp = kwargs.get("gt_spp", None)
        self.path_depth = kwargs.get("path_depth", 5)
        self.pixel_filter = kwargs.get("pixel_filter", "box")
        self.height = kwargs.get("height", 512)
        self.width = kwargs.get("width", 1024)
        self.random_crop_w = kwargs.get("random_crop_w", None)
        self.random_crop_h = kwargs.get("random_crop_h", None)
        self.integrator = kwargs.get("integrator", "pathrendernet")
        self.sampler = kwargs.get("sampler", "lowdiscrepancy")
        self.filename = kwargs.get("filename", None)

        if self.integrator == "pathrendernet":
            if self.path_depth != 5:
                LOG.error("Sample-based rendering only supports path_depth=5")
                raise RuntimeError("Sample-based rendering only supports"
                                   " path_depth=5, got %d" % self.path_depth)
            if self.gt_spp is None:
                LOG.error("Sample-based rendering requires `gt_spp` to be set")
                raise RuntimeError("Sample-based rendering requires `gt_spp`"
                                   " to be set")

    def pbrt(self):
        """Output PBRT scene description string."""

        out = 'SurfaceIntegrator "%s" "integer maxdepth" [%d]\n' % (
            self.integrator, self.path_depth)

        # Sampler configuration
        if self.integrator == "pathrendernet":
            out += 'Renderer "rendernet" "integer tilesize" [{}] ' \
                '"integer recordedsamples" [{}]\n'.format(self.tile_size,
                                                          self.spp)
            out += 'Sampler "%s" "integer pixelsamples" [%d]\n' % (
                self.sampler, self.gt_spp)  # The ref has `gt_spp` samples
        else:
            out += 'Sampler "%s" "integer pixelsamples" [%d]\n' % (
                self.sampler, self.spp)

        out += 'PixelFilter "%s"\n' % self.pixel_filter
        out += self._film()

        return out

    def _film(self):
        film = 'Film "image" "integer xresolution" [%d]' % self.width
        film += ' "integer yresolution" [%d]' % self.height
        if self.filename is not None:
            film += ' "string filename" "%s"\n\n' % self.filename

        # Handle crop requests
        if self.random_crop_w is not None:
            if self.random_crop_h is None:
                LOG.error("When requesting a crop, provide both x and y"
                          " coordinates.")
                raise RuntimeError("Please provide both x/y crop.")

            height = self.height
            width = self.width

            # Make sure the crop is within bounds
            start_x = 0
            if self.random_crop_w < width:
                start_x = np.random.randint(
                    0, width - self.random_crop_w) * 1.0 / width

            start_y = 0
            if self.random_crop_h < height:
                start_y = np.random.randint(
                    0, height - self.random_crop_h) * 1.0 / height

            # PBRT expects crop coordinates in [0, 1]
            float_w = 1.0*self.random_crop_w / width
            float_h = 1.0*self.random_crop_h / height
            film += ' "float cropwindow" [{} {} {} {}]'.format(
                start_x, start_x + float_w, start_y, start_y + float_h)

        film += "\n"
        return film


class Camera():
    """PBRT camera.

    Args:
    """
    def __init__(self, **kwargs):
        self.xform = kwargs.get("xform", None)
        self.position = kwargs.get("position", [0, 0, 1])
        self.target = kwargs.get("target", [0, 0, 0])
        self.up = kwargs.get("up", [0, 1, 0])
        self.type = kwargs.get("type", "perspective")
        self.fov = kwargs.get("fov", 35)
        self.shutteropen = kwargs.get("shutteropen", 0.0)
        self.shutterclose = kwargs.get("shutterclose", 0.0)
        self.lensradius = kwargs.get("lensradius", 0.0)
        self.focaldistance = kwargs.get("focaldistance", 0.0)

    def __repr__(self):
        out = "camera\n"
        out += " .fov = {}\n".format(self.fov)
        out += " .at ({:.1f} {:.1f} {:.1f})\n".format(*self.position)
        out += " .looking at ({:.1f} {:.1f} {:.1f})\n".format(*self.target)
        out += " .shutter ({:.1f} {:.1f})\n".format(self.shutteropen,
                                                    self.shutterclose)
        out += " .focus distance {:.1f}\n".format(self.focaldistance)
        out += " .lens radius {:.10f}\n".format(self.lensradius)
        return out

    def pbrt(self):
        """PBRT string representation.

        Returns:
            s(str): PBRT formated string.
        """
        if self.xform is None:
            s = 'LookAt {} {} {}  {} {} {}  {} {} {}\n'.format(
                *(self.position + self.target + self.up))
        else:
            s = self.xform.pbrt()
        s += ('Camera "{}" "float fov" [{}] "float shutteropen" [{}] '
              '"float shutterclose" [{}] "float lensradius" [{}]'
              ' "float focaldistance" [{}]\n').format(self.type, self.fov,
                                                      self.shutteropen,
                                                      self.shutterclose,
                                                      self.lensradius,
                                                      self.focaldistance)
        return s
