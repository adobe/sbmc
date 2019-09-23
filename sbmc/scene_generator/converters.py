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
"""Helpers for file format conversion."""

import os
import re
import uuid
import subprocess

import ttools

__all__ = ["ObjConverter"]


LOG = ttools.get_logger(__name__)


OBJ_RE = re.compile(r'^# Name\s*"(?P<obj_name>.*)".*$')
"""Regular expression for objects parsing."""


MAT_RE = re.compile(
    r'.*"float roughness"\s\[(?P<roughness>[^\]]*)\]\s.*"float index"'
    r'\s*\[(?P<index>[^\]]*)\]\s.*"rgb opacity"\s*\[(?P<opacity>[^\]]*)\].*')
"""Regular expression for materials parsing."""


class ObjConverter():
    """A helper class used that converts .obj files to the .pbrt string format.

    Args:
        pbrt_converter(str): path to the `obj2pbrt` executable from PBRT.
    """
    def __init__(self, pbrt_converter):
        self.pbrt_converter = pbrt_converter

    def __call__(self, fname, dst):
        """Convert a .obj geometry to the .pbrt format.

        Args:
            fname(str): path to the .obj file.
            dst(str): path to the .pbrt output.
        """
        dirname = os.path.dirname(fname)
        basename = os.path.basename(fname)

        # OBJ companion material if any
        mtl_file = os.path.splitext(basename)[0] + ".mtl"

        _split_material_groups(fname, os.path.join(dst, basename))

        pbrt_file = basename.replace(".obj", ".pbrt")

        cwd = os.getcwd()
        os.chdir(dst)

        if not os.path.exists(mtl_file):
            os.symlink(os.path.join(dirname, mtl_file), mtl_file)

        subprocess.check_output([self.pbrt_converter, basename, pbrt_file])

        objects = _split_pbrt_objects(pbrt_file)

        os.remove(basename)
        os.remove(mtl_file)

        os.chdir(cwd)
        return objects


def _split_pbrt_objects(in_f):
    """Given an input .pbrt object file, splits the objects by material groups.

    This functions creates new .pbrt files identified by a unique material id
    alongside the original .pbrt file. It returns the path to the .pbrt files
    generated, as well as the corresponding materials.

    Args:
        in_f(str): path to the input .pbrt geometry.

    Returns:
        objects(dict of )
    """
    object_idx = 0
    objects = []

    with open(in_f) as fid:
        line = fid.readline()
        while line:  # parse the .pbrt file
            matches = OBJ_RE.match(line)
            if matches:  # We have a new object
                # Ignore header, comments, etc
                while (not line.startswith("Material") or
                       line.startswith("Shape")):
                    line = fid.readline()

                # Parse material definition if there is one
                if line.startswith("Material"):
                    mat_ = OBJMaterial(line)
                else:
                    LOG.error("Found no material.")
                    mat_ = OBJMaterial("")

                # Seek to the shape definition
                while not line.startswith("Shape"):
                    line = fid.readline()

                # Write a new geometry file for each material group
                new_f = os.path.splitext(
                    in_f)[0] + "%s_object%04d.pbrt" % (mat_.id, object_idx)
                with open(new_f, 'w') as new_fid:
                    # Define the material
                    new_fid.write('AttributeBegin\n')
                    new_fid.write('NamedMaterial "%s"\n' % mat_.id)

                    # Write the geometry content
                    while not line.strip() == "AttributeEnd":
                        new_fid.write(line)
                        line = fid.readline()

                    # write the AttributeEnd command
                    new_fid.write(line)

                object_idx += 1

                # Store the path to the geometry file and the material
                # definition
                obj_ = PBRTObject(new_f, mat_)

                # Save the object
                objects.append(obj_)

            line = fid.readline()

    return objects


def _split_material_groups(src, dst):
    """Splits .obj material groups.

    This functions adds matgroup tags so we can split the geometry per material
    group.

    Args:
        src(str): path to source .obj file.
        dst(str): path to target .obj file.
    """
    matre = re.compile(r'^usemtl\s*(?P<mat>.*)\s*$')
    geomre = re.compile(r'^g\s*(?P<geo>.*)\s*$')

    basename = os.path.splitext(os.path.basename(src))[0]

    with open(dst, 'w') as dst_fid:
        last_geom = None  # Beginning of the file: no geometry found yet
        with open(src) as src_fid:
            for line in src_fid.readlines():
                geometry_match = geomre.match(line)

                if geometry_match:
                    # We found a new geometry group, memory it
                    last_geom = geometry_match.group("geo")
                else:
                    material_match = matre.match(line)
                    if material_match:
                        matname = material_match.group("mat")

                        if last_geom is None:
                            # not geometry: global material
                            name = "g %s@%s\n" % (basename, matname)
                        else:
                            # we had a geometry, apply material to id
                            name = "g %s@%s\n" % (last_geom, matname)
                        dst_fid.write(name)
                dst_fid.write(line)


class OBJMaterial():
    """Holds a material parsed from an OBJ description string.

    Args:
        line(str): string to parse.
    """
    def __init__(self, line):
        if line == "":  # defalt material if there is nothing to be parsed
            self.roughness = 0.5
            self.index = 1.0
            self.opacity = 1.0
        else:
            mat = MAT_RE.match(line)
            if mat is None:
                LOG.error("Parser could not understand OBJ material '%s'",
                          line)
                raise RuntimeError(
                    "Parser could not understand OBJ material '%s'", line)
            self.roughness = float(mat.group("roughness")),
            self.index = float(mat.group("index")),
            self.opacity = min([float(c) for c in
                                mat.group("opacity").split()])
        self.id = str(uuid.uuid4()).replace("-", "_")


class PBRTObject():
    """
    """
    def __init__(self, path, material):
        self.path = path
        self.material = material
