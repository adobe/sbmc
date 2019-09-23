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
"""Texture classes for the PBRT scene generator."""

import ttools


__all__ = ["Imagemap", "Checkerboard"]


LOG = ttools.get_logger(__name__)


class Texture():
    """Base texture class.

    Args:
        id(str): a unique string identifier for this texture.
        type(str): a unique string identifier for this texture.
        uscale(float): scaling of the texture coordinate in u direction.
        vscale(float): scaling of the texture coordinate in v direction.
        udelta(float): shift of the texture coordinate in u direction.
        vdelta(float): shift of the texture coordinate in v direction.
    """
    def __init__(self, id, type, uscale=1.0, vscale=1.0, udelta=0.0,
                 vdelta=0.0):
        if type not in ["spectrum", "float"]:
            LOG.error("texture type should be one of `spectrum`, `float`,"
                      " got %s", type)
            raise RuntimeError("Unknown texture type %s", type)
        self.id = id
        self.type = type
        self.uscale = uscale
        self.vscale = vscale
        self.udelta = udelta
        self.vdelta = vdelta

    def pbrt(self):
        """Outputs PBRTv2 string representation.

        Returns:
            str: pbrt format.
        """
        return 'Texture "{}" "{}" '.format(self.id, self.type)

    def suffix(self):
        """Generic PBRT parameters shared by all textures.

        Returns:
            (str): the pbrt string suffix parameterizing the texture.
        """
        return '"float uscale" [{}] "float vscale" [{}] "float udelta"'\
            ' [{}] "float vdelta" [{}]'.format(self.uscale, self.vscale,
                                               self.udelta, self.vdelta)


class Imagemap(Texture):
    """Texture based on a bitmap image.

    Args:
        id(str): a unique string identifier for this texture.
        type(str): a unique string identifier for this texture.
        filename(str): path to the texture (PBRT2 supports .tga and .bmp only).
        scale(float): scale factor to apply to the bitmap.
        gamma(float): gamma correction to apply to the bitmap.
        trilinear(bool): if True, uses trilinear filtering.
        uscale(float): scaling of the texture coordinate in u direction.
        vscale(float): scaling of the texture coordinate in v direction.
        udelta(float): shift of the texture coordinate in u direction.
        vdelta(float): shift of the texture coordinate in v direction.
    """
    def __init__(self, id, type, filename, scale=1.0, gamma=1.0,
                 trilinear=False, uscale=1.0, vscale=1.0, udelta=1.0,
                 vdelta=1.0):
        super(Imagemap, self).__init__(id, type, uscale=uscale, vscale=vscale,
                                       udelta=udelta, vdelta=vdelta)
        self.filename = filename
        self.gamma = gamma
        self.scale = scale
        self.trilinear = "true" if trilinear else "false"

    def pbrt(self):
        s = super(Imagemap, self).pbrt()
        s += ' "imagemap" "string filename" ["{}"] "float gamma" [{}] ' \
            '"float scale" [{}] "bool trilinear" ["{}"]\n'.format(
                self.filename, self.gamma, self.scale, self.trilinear)
        s += super(Imagemap, self).suffix()
        return s


class Checkerboard(Texture):
    """A simple parametric checkerboard texture.

    Args:
        id(str): a unique string identifier for this texture.
        type(str): a unique string identifier for this texture.
        uscale(float): scaling of the texture coordinate in u direction.
        vscale(float): scaling of the texture coordinate in v direction.
        udelta(float): shift of the texture coordinate in u direction.
        vdelta(float): shift of the texture coordinate in v direction.
        tex1(list of 3 floats or None): first color of the checker.
        tex2(list of 3 floats or None): second color of the checker.
    """
    def __init__(self, id, type, uscale=1.0, vscale=1.0, udelta=1.0,
                 vdelta=1.0, tex1=None, tex2=None):
        super(Checkerboard, self).__init__(id, type, uscale=uscale,
                                           vscale=vscale, udelta=udelta,
                                           vdelta=vdelta)
        if tex1 is not None:
            self.tex1 = tex1
        else:
            self.tex1 = [0.9, 0.9, 0.9]
        if tex2 is not None:
            self.tex2 = tex2
        else:
            self.tex2 = [0.2, 0.2, 0.2]

    def pbrt(self):
        s = super(Checkerboard, self).pbrt()
        s += ' "checkerboard" "rgb tex1" [{} {} {}] "rgb tex2" ' \
            '[{} {} {}]\n'.format(*self.tex1, *self.tex2)
        s += super(Checkerboard, self).suffix()
        return s
