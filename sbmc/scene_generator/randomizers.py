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
"""Functions to generate random scene content."""

import copy
import uuid
import numpy as np
from . import (lights, materials, textures)


def random_texture(textures_list, gamma=1.0):
    """Generates a random texture from a list of texture images.
    Args:
        textures_list(list of str): path to the texture images.
        gamma(float): gamma correction factor.
    """
    texname = np.random.choice(textures_list)
    name = str(uuid.uuid4())
    tex = textures.Imagemap(name, "spectrum", texname, gamma=gamma,
                            udelta=np.random.uniform(0, 1),
                            vdelta=np.random.uniform(0, 1),
                            uscale=np.random.uniform(0.1, 20),
                            vscale=np.random.uniform(0.1, 20))
    return tex


def random_envmap(maps, nsamples=8):
    """Creates a random environment map light from a list of .pfm images.

    Args:
        maps(list of str): paths to the hdr images, in .pfm format.
    """
    m = np.random.choice(maps)
    light = lights.InifiniteLight(mapname=m, nsamples=nsamples)
    return light


def random_diffuse_material(id=None, textures_list=None):
    """Random diffuse material.

    Args:
        id(str): unique identifier for the material.
        textures_list(list of str): path to the texture images.
    """
    gamma = 1.0

    if textures_list is not None:
        tex = random_texture(textures_list, gamma=gamma)
        mat = materials.MatteMaterial(
            id=id, diffuse_texture=tex, bump_texture=None)
    else:
        diffuse = list(np.random.uniform(size=(3,)))
        mat = materials.MatteMaterial(id=id, diffuse=diffuse)
    return mat


def random_uber_material(id=None, textures_list=None):
    """Random uber material.

    Args:
        id(str): unique identifier for the material.
        textures_list(list of str): path to the texture images.
    """
    gamma = 1.0

    if textures_list is not None:
        tex = random_texture(textures_list, gamma=gamma)
        glossy = np.random.uniform(0, 1)
        specular = np.random.uniform(0, 1)
        rough = np.random.uniform(0, 1)
        index = np.random.uniform(1, 1.5)
        mat = materials.UberMaterial(id=id, diffuse_texture=tex,
                                     bump_texture=None,
                                     glossy_reflection=glossy,
                                     specular_reflection=specular,
                                     roughness=rough, index=index)
    else:
        diffuse = list(np.random.uniform(size=(3,)))
        mat = materials.UberMaterial(id=id, diffuse=diffuse)
    return mat


def random_mirror_material(id=None, textures_list=None):
    """Random mirror material.

    Args:
        id(str): unique identifier for the material.
        textures_list(list of str): path to the texture images.
    """
    gamma = 1.0

    spec = list(np.random.uniform(size=(3,)))
    mat = materials.MirrorMaterial(id=id, specular=spec)

    return mat


def random_glass_material(id=None, textures_list=None):
    """Random glass material.

    Args:
        id(str): unique identifier for the material.
        textures_list(list of str): path to the texture images.
    """
    gamma = 1.0
    reflection = list(np.random.uniform(size=(3,)))
    transmission = list(np.random.uniform(size=(3,)))
    index = np.random.uniform(1.1, 1.8)
    mat = materials.GlassMaterial(id=id, reflection=reflection,
                                  transmission=transmission, bump_texture=None)
    return mat


def random_metal_material(id=None, textures_list=None):
    """Random metal material.

    Args:
        id(str): unique identifier for the material.
        textures_list(list of str): path to the texture images.
    """
    gamma = 1.0
    eta = list(np.random.uniform(size=(3,)))
    k = list(np.random.uniform(1, 3, size=(3,)))
    rough = np.random.uniform(0, 1)
    mat = materials.MetalMaterial(id=id, eta=eta, k=k,
                                  roughness=rough, bump_texture=None)
    return mat


def random_plastic_material(id=None, textures_list=None):
    """Random plastic material.

    Args:
        id(str): unique identifier for the material.
        textures_list(list of str): path to the texture images.
    """
    gamma = 1.0

    specular = list(np.random.uniform(0, 1, size=(3,)))
    rough = np.random.uniform(0, 1)
    if textures_list is not None:
        tex = random_texture(textures_list, gamma=gamma)
        mat = materials.PlasticMaterial(id=id, diffuse_texture=tex,
                                        bump_texture=None, specular=specular,
                                        roughness=rough)
    else:
        diffuse = list(np.random.uniform(size=(3,)))
        mat = materials.PlasticMaterial(id=id, roughness=rough,
                                        diffuse=diffuse, bump_texture=None)
    return mat


def random_substrate_material(id=None, textures_list=None):
    """Random substrate material.

    Args:
        id(str): unique identifier for the material.
        textures_list(list of str): path to the texture images.
    """
    gamma = 1.0

    specular = list(np.random.uniform(0, 1, size=(3,)))
    rough = np.random.uniform(0, 1)
    rough2 = np.random.uniform(0, 1)
    if textures_list is not None:
        tex = random_texture(textures_list, gamma=gamma)
        mat = materials.SubstrateMaterial(id=id, diffuse_texture=tex,
                                          bump_texture=None, specular=specular,
                                          uroughness=rough, vroughness=rough2)
    else:
        diffuse = np.random.uniform(size=(3,))
        mat = materials.SubstrateMaterial(
            id=id, diffuse=diffuse, uroughness=rough, vroughness=rough2,
            bump_texture=None)
    return mat


def random_material(id=None, textures_list=None):
    """Random material.

    With a bias towards simple diffuse materials.

    Args:
        id(str): unique identifier for the material.
        textures_list(list of str): path to the texture images.
    """
    p = np.random.uniform()
    p_tex = np.random.uniform()

    if id is None:
        id = str(uuid.uuid4())

    if p_tex < 0.01:  # 1% chance uniform
        tex = None
    else:
        tex = textures_list

    if p < 0.05:
        return random_mirror_material(id, tex)
    if p < 0.1:
        return random_glass_material(id, tex)
    if p < 0.2:
        return random_metal_material(id, tex)
    if p < 0.4:
        return random_plastic_material(id, tex)
    if p < 0.5:
        return random_substrate_material(id, tex)
    if p < 0.8:
        return random_uber_material(id, tex)

    return random_diffuse_material(id, tex)


def random_bump(ref_texture, textures_list):
    """Random bump map texture.

    The texture is randomly decorrelated from the reference with 50%
    probability.

    Args:
        ref_texture(sbmc.scene_generator.Texture): reference color texture.
        textures_list(list of str): path to the texture images.
    """
    if textures_list is None:
        return None
    do_bump = np.random.uniform() > 0.5
    if not do_bump:
        return None

    # randomly decorrelate bump texture
    if ref_texture is None or np.random.uniform() > 0.5:
        texname = np.random.choice(textures_list)

        name = str(uuid.uuid4())
        bump_scale = np.random.uniform(0.01, 0.05)
        bump = textures.Imagemap(
            name, "float",
            texname, scale=bump_scale, gamma=1.0, trilinear=True)
    else:
        bump = copy.deepcopy(ref_texture)
        bump.type = "float"
    return bump
