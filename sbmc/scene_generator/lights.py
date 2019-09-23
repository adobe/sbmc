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
"""Light sources to build PBRT scenes."""

__all__ = ["InifiniteLight", "PointLight", "AreaLight"]


class Light():
    """Base class for light sources."""
    def __init__(self):
        self.xform = None

    def pbrt(self):
        """Outputs PBRTv2 string representation.

        Returns:
            str: pbrt format.
        """
        return "LightSource"


class PointLight(Light):
    """A point light source.

    Args:
        spectrum(None or list of float): RGB intensity of the light.
            Defaults to white.
    """
    def __init__(self, spectrum=None):
        super(PointLight, self).__init__()
        if spectrum is None:
            spectrum = [1, 1, 1]
        self.spectrum = spectrum

    def pbrt(self):
        s = "AttributeBegin\n"
        if self.xform is not None:
            s += self.xform.pbrt()
        s += super(PointLight, self).pbrt()
        s += ' "point" "rgb I" [{} {} {}]\n'.format(*self.spectrum)
        s += "AttributeEnd\n"
        return s


class InifiniteLight(Light):
    """Infinitely distant light source. Useful for envmaps.

    If `mapname` is provided, `spectrum` is overidden by the map's color.

    Args:
        spectrum(None or list of float): RGB intensity of the light.
            Defaults to white.
        mapname(None or str): environment map to use as light.
        nsamples(int): number of light samples per pixel.
    """
    def __init__(self, spectrum=None, mapname=None, nsamples=8):
        super(InifiniteLight, self).__init__()
        self.nsamples = nsamples
        self.mapname = None
        self.spectrum = None

        if mapname is None:
            if spectrum is None:
                self.spectrum = [1, 1, 1]
            else:
                self.spectrum = spectrum
        else:
            self.mapname = mapname

    def pbrt(self):
        s = "AttributeBegin\n"
        if self.xform is not None:
            s += self.xform.pbrt()
        s += super(InifiniteLight, self).pbrt()
        s += ' "infinite" '
        if self.mapname is None:
            s += '"rgb L" [{} {} {}]'.format(*self.spectrum)
        else:
            s += '"string mapname" ["{}"]'.format(self.mapname)
        s += ' "integer nsamples" [{}]\n'.format(self.nsamples)
        s += "AttributeEnd\n"

        return s


class AreaLight(Light):
    """Geometric area light.

    Args:
        geom(sbmc.scene_generator.Shape): shape of the light.
        spectrum(None or list of float): RGB intensity of the light.
            Defaults to white.
        nsamples(int): number of light samples per pixel.
    """
    def __init__(self, geom, spectrum=None, nsamples=8):
        super(AreaLight, self).__init__()
        self.nsamples = nsamples
        self.spectrum = None
        self.geom = geom

        if spectrum is None:
            self.spectrum = [1, 1, 1]
        else:
            self.spectrum = spectrum

    def pbrt(self):
        s = "AttributeBegin\n"
        if self.xform is not None:
            s += self.xform.pbrt()
        s += 'AreaLightSource "diffuse" '
        s += '"rgb L" [{} {} {}]'.format(*self.spectrum)
        s += ' "integer nsamples" [{}]\n'.format(self.nsamples)
        s += self.geom.pbrt()
        s += "AttributeEnd\n"
        return s
