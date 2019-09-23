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
"""Material classes for the PBRT scene generator."""

import ttools


__all__ = ["MatteMaterial", "UberMaterial", "MirrorMaterial", "GlassMaterial",
           "MetalMaterial", "PlasticMaterial", "SubstrateMaterial"]


LOG = ttools.get_logger(__name__)


class Material():
    """Base material.

    Args:
        id(str): unique id to identify the material.
        bump_texture(Texture or None): float texture to use as bump map.
    """
    def __init__(self, id=None, bump_texture=None):
        self.id = id
        self.textures = []
        self.bump_texture = bump_texture
        if self.bump_texture is not None:
            if self.bump_texture.type != "float":
                LOG.error("Bump texture should be of type `float`, got %s",
                          self.bump_texture.type)
                raise RuntimeError("Incorrect bump type error")
            self.textures.append(bump_texture)

    def pbrt(self):
        """Outputs PBRTv2 string representation.

        Returns:
            str: pbrt format.
        """
        out = ""
        for tex in self.textures:
            out += tex.pbrt()
        if self.id:
            return out + 'MakeNamedMaterial "{}" "string type"'.format(self.id)
        else:
            return out + "Material"

    def suffix(self):
        """Generic PBRT parameters shared by all materials.

        Returns:
            (str): the pbrt string suffix parameterizing the texture.
        """
        out = " "
        if self.bump_texture is not None:
            out += '"texture bumpmap" "{}"\n'.format(self.bump_texture.id)
        return out


class MatteMaterial(Material):
    """PBRT Matte material.

    Args:
        id(str): unique id to identify the material.
        diffuse(list of 3 floats or None): color of the diffuse component.
        diffuse_texture(sbmc.scene_generator.Texture): texture for the diffuse
            component. Overrides `diffuse` if provided.
        bump_texture(Texture or None): float texture to use as bump map.
    """
    def __init__(self, id=None, diffuse=None, diffuse_texture=None,
                 bump_texture=None):
        super(MatteMaterial, self).__init__(id=id, bump_texture=bump_texture)
        self.diffuse = None
        self.diffuse_texture = None
        if diffuse_texture is None:
            if diffuse is None:
                self.diffuse = [1, 1, 1]
            else:
                self.diffuse = diffuse
        else:
            self.diffuse_texture = diffuse_texture
            self.textures.append(diffuse_texture)

    def pbrt(self):
        out = super(MatteMaterial, self).pbrt()
        out += ' "matte" '
        if self.diffuse_texture is not None:
            out += '"texture Kd" "{}"\n'.format(self.diffuse_texture.id)
        else:
            out += '"rgb Kd" [{} {} {}]\n'.format(*self.diffuse)
        out += super(MatteMaterial, self).suffix()
        return out


class UberMaterial(MatteMaterial):
    """PBRT Uber material.

    Args:
        id(str): unique id to identify the material.
        diffuse(list of 3 floats or None): color of the diffuse component.
        diffuse_texture(sbmc.scene_generator.Texture): texture for the diffuse
            component. Overrides `diffuse` if provided.
        glossy_reflection(float): intensity of the glossy reflection.
        specular_reflection(float): intensity of the specular reflection.
        roughness(float): roughness of the material.
        index(float): index of refraction of the material.
        opacity(float): opacity of the material (0 is transparent, 1 opaque).
        bump_texture(Texture or None): float texture to use as bump map.
    """
    def __init__(self, id=None, diffuse=None, diffuse_texture=None,
                 glossy_reflection=0.25, specular_reflection=0,
                 roughness=0.1, index=1.5, opacity=1.0, bump_texture=None):
        super(UberMaterial, self).__init__(
            id=id, diffuse=diffuse, diffuse_texture=diffuse_texture,
            bump_texture=bump_texture)
        self.glossy_reflection = [glossy_reflection]*3
        self.specular_reflection = [specular_reflection]*3
        self.roughness = roughness
        self.index = index
        self.opacity = [opacity]*3

    def pbrt(self):
        out = super(UberMaterial, self).pbrt()
        out = out.replace('"matte"', '"uber"').strip()
        out += ' "rgb Ks" [{} {} {}]'.format(*self.glossy_reflection)
        out += ' "rgb Kr" [{} {} {}]'.format(*self.specular_reflection)
        out += ' "float roughness" [{}]'.format(self.roughness)
        out += ' "float index" [{}]'.format(self.index)
        out += ' "rgb opacity" [{} {} {}]'.format(*self.opacity)
        out += super(UberMaterial, self).suffix()
        out += '\n'
        return out


class MirrorMaterial(Material):
    """PBRT Mirror material.

    Args:
        id(str): unique id to identify the material.
        specular(list of 3 floats or None): color of the specular component.
        specular_texture(sbmc.scene_generator.Texture): texture for the
            specular component. Overrides `diffuse` if provided.
        bump_texture(Texture or None): float texture to use as bump map.
    """
    def __init__(self, id=None, specular=None, specular_texture=None,
                 bump_texture=None):
        super(MirrorMaterial, self).__init__(id=id, bump_texture=bump_texture)
        self.specular = None
        self.specular_texture = None
        if specular_texture is None:
            if specular is None:
                self.specular = [1, 1, 1]
            else:
                self.specular = specular
        else:
            self.specular_texture = specular_texture
            self.textures.append(specular_texture)

    def pbrt(self):
        out = super(MirrorMaterial, self).pbrt()
        out += ' "mirror" '
        if self.specular_texture is not None:
            out += '"texture Kr" "{}"\n'.format(self.specular_texture.id)
        else:
            out += '"rgb Kr" [{} {} {}]\n'.format(*self.specular)
        out += super(MirrorMaterial, self).suffix()
        return out


class GlassMaterial(Material):
    """PBRT Glass material.

    Args:
        id(str): unique id to identify the material.
        reflection(list of 3 floats or None): color of the reflection
            component.
        reflection_texture(sbmc.scene_generator.Texture): texture for the
            reflection component. Overrides `reflection` if provided.
        transmission(list of 3 floats or None): color of the transmission
            component. transmission_texture(sbmc.scene_generator.Texture):
            texture for the transmission component. Overrides `transmission` if
            provided.
        index(float): index of refraction of the material.
        bump_texture(Texture or None): float texture to use as bump map.
    """
    def __init__(self, id=None, reflection=None, reflection_texture=None,
                 transmission=None, transmission_texture=None,
                 index=1.5, bump_texture=None):
        super(GlassMaterial, self).__init__(id=id, bump_texture=bump_texture)
        self.index = index
        self.reflection = None
        self.reflection_texture = None
        self.transmission = None
        self.transmission_texture = None
        if reflection_texture is None:
            if reflection is None:
                self.reflection = [1, 1, 1]
            else:
                self.reflection = reflection
        else:
            self.reflection_texture = reflection_texture
            self.textures.append(reflection_texture)
        if transmission_texture is None:
            if transmission is None:
                self.transmission = [1, 1, 1]
            else:
                self.transmission = transmission
        else:
            self.transmission_texture = transmission_texture
            self.textures.append(transmission_texture)

    def pbrt(self):
        out = super(GlassMaterial, self).pbrt()
        out += ' "glass" '
        out += ' "float index" [{}] '.format(self.index)
        if self.reflection_texture is not None:
            out += '"texture Kr" "{}"\n'.format(self.reflection_texture.id)
        else:
            out += '"rgb Kr" [{} {} {}]\n'.format(*self.reflection)
        if self.transmission_texture is not None:
            out += '"texture Kt" "{}"\n'.format(self.transmission_texture.id)
        else:
            out += '"rgb Kt" [{} {} {}]\n'.format(*self.transmission)
        out += super(GlassMaterial, self).suffix()
        return out


class MetalMaterial(Material):
    """PBRT Metal material.

    Args:
        id(str): unique id to identify the material.
        eta(list of 3 floats or None): colored index of refraction
        eta_texture(sbmc.scene_generator.Texture): texture for the colored
            index of refraction. Overrides `eta` if provided.
        k(list of 3 floats or None): colored absorption coefficient.
        k_texture(sbmc.scene_generator.Texture): texture for the k
            coefficient. Overrides `k` if provided.
        roughness(float): roughness of the material.
        bump_texture(Texture or None): float texture to use as bump map.
    """
    def __init__(self, id=None, eta=None, eta_texture=None,
                 k=None, k_texture=None,
                 roughness=0.01, bump_texture=None):
        super(MetalMaterial, self).__init__(id=id, bump_texture=bump_texture)
        self.roughness = roughness
        self.eta = None
        self.eta_texture = None
        self.k = None
        self.k_texture = None

        if eta_texture is None:
            if eta is None:
                self.eta = [0.6, 0.5, 0.4]
            else:
                self.eta = eta
        else:
            self.eta_texture = eta_texture
            self.textures.append(eta_texture)

        if k_texture is None:
            if k is None:
                self.k = [2, 2, 3]
            else:
                self.k = k
        else:
            self.k_texture = k_texture
            self.textures.append(k_texture)

    def pbrt(self):
        out = super(MetalMaterial, self).pbrt()
        out += ' "metal" '
        out += ' "float roughness" [{}] '.format(self.roughness)
        if self.eta_texture is not None:
            out += '"texture eta" "{}"\n'.format(self.eta_texture.id)
        else:
            out += '"rgb eta" [{} {} {}]\n'.format(*self.eta)
        if self.k_texture is not None:
            out += '"texture k" "{}"\n'.format(self.k_texture.id)
        else:
            out += '"rgb k" [{} {} {}]\n'.format(*self.k)
        out += super(MetalMaterial, self).suffix()
        return out


class PlasticMaterial(Material):
    """PBRT Plastic material.

    Args:
        id(str): unique id to identify the material.
        diffuse(list of 3 floats or None): color of the diffuse component.
        diffuse_texture(sbmc.scene_generator.Texture): texture for the diffuse
            component. Overrides `diffuse` if provided.
        specular(list of 3 floats or None): color of the specular component.
        specular_texture(sbmc.scene_generator.Texture): texture for the
            specular component. Overrides `specular` if provided.
        roughness(float): roughness of the material.
        bump_texture(Texture or None): float texture to use as bump map.
    """
    def __init__(self, id=None, diffuse=None, diffuse_texture=None,
                 specular=None, specular_texture=None,
                 roughness=0.01, bump_texture=None):
        super(PlasticMaterial, self).__init__(id=id, bump_texture=bump_texture)
        self.roughness = roughness
        self.diffuse = None
        self.diffuse_texture = None
        self.specular = None
        self.specular_texture = None
        if diffuse_texture is None:
            if diffuse is None:
                self.diffuse = [1.0, 1.0, 1.0]
            else:
                self.diffuse = diffuse
        else:
            self.diffuse_texture = diffuse_texture
            self.textures.append(diffuse_texture)
        if specular_texture is None:
            if specular is None:
                self.specular = [1, 1, 1]
            else:
                self.specular = specular
        else:
            self.specular_texture = specular_texture
            self.textures.append(specular_texture)

    def pbrt(self):
        out = super(PlasticMaterial, self).pbrt()
        out += ' "plastic" '
        out += ' "float roughness" [{}] '.format(self.roughness)
        if self.diffuse_texture is not None:
            out += '"texture Kd" "{}"\n'.format(self.diffuse_texture.id)
        else:
            out += '"rgb Kd" [{} {} {}]\n'.format(*self.diffuse)
        if self.specular_texture is not None:
            out += '"texture Ks" "{}"\n'.format(self.specular_texture.id)
        else:
            out += '"rgb Ks" [{} {} {}]\n'.format(*self.specular)
        out += super(PlasticMaterial, self).suffix()
        return out


class SubstrateMaterial(Material):
    """PBRT Substrate material.

    Args:
        id(str): unique id to identify the material.
        diffuse(list of 3 floats or None): color of the diffuse component.
        diffuse_texture(sbmc.scene_generator.Texture): texture for the diffuse
            component. Overrides `diffuse` if provided.
        specular(list of 3 floats or None): color of the specular component.
        specular_texture(sbmc.scene_generator.Texture): texture for the
            specular component. Overrides `specular` if provided.
        uroughness(float): roughness of the material in the u direction.
        vroughness(float): roughness of the material in the v direction.
        bump_texture(Texture or None): float texture to use as bump map.
    """
    def __init__(self, id=None, diffuse=None, diffuse_texture=None,
                 specular=None, specular_texture=None,
                 uroughness=0.1, vroughness=0.1, bump_texture=None):
        super(SubstrateMaterial, self).__init__(
            id=id, bump_texture=bump_texture)
        self.uroughness = uroughness
        self.vroughness = vroughness
        self.diffuse = None
        self.diffuse_texture = None
        self.specular = None
        self.specular_texture = None
        if diffuse_texture is None:
            if diffuse is None:
                self.diffuse = [0.5, 0.5, 0.5]
            else:
                self.diffuse = diffuse
        else:
            self.diffuse_texture = diffuse_texture
            self.textures.append(diffuse_texture)
        if specular_texture is None:
            if specular is None:
                self.specular = [0.5, 0.5, 0.5]
            else:
                self.specular = specular
        else:
            self.specular_texture = specular_texture
            self.textures.append(specular_texture)

    def pbrt(self):
        out = super(SubstrateMaterial, self).pbrt()
        out += ' "substrate" '
        out += ' "float uroughness" [{}] '.format(self.uroughness)
        out += ' "float vroughness" [{}] '.format(self.vroughness)
        if self.diffuse_texture is not None:
            out += '"texture Kd" "{}"\n'.format(self.diffuse_texture.id)
        else:
            out += '"rgb Kd" [{} {} {}]\n'.format(*self.diffuse)
        if self.specular_texture is not None:
            out += '"texture Ks" "{}"\n'.format(self.specular_texture.id)
        else:
            out += '"rgb Ks" [{} {} {}]\n'.format(*self.specular)
        out += super(SubstrateMaterial, self).suffix()
        return out
