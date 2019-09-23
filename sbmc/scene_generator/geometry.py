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
"""Abstraction of geometric primitives to build PBRT scenes."""

import numpy as np

__all__ = ["Sphere", "ExternalGeometry", "Cylinder", "Plane", "TriangleMesh"]


class Shape():
    """Base shape.

    Attributes:
        xform():
        start_xform():
        end_xform():
        material():
    """
    def __init__(self):
        self.xform = None
        self.start_xform = None
        self.end_xform = None
        self.material = None

    def _prefix(self):
        """String to prepend the pbrt output with.

        Returns:
            (str): prefix string.
        """
        s = "AttributeBegin\n"
        if self.material is not None:
            s += 'NamedMaterial "{}"\n'.format(self.material.id)
        if self.xform is not None:
            s += self.xform.pbrt()
        if self.start_xform is not None:
            s += "ActiveTransform StartTime\n"
            s += self.start_xform.pbrt()
            s += "ActiveTransform All\n"
        if self.end_xform is not None:
            s += "ActiveTransform EndTime\n"
            s += self.end_xform.pbrt()
            s += "ActiveTransform All\n"
        return s

    def _suffix(self):
        """String to append the pbrt output with.

        Returns:
            (str): suffix string.
        """
        s = "AttributeEnd\n"
        return s

    def assign_material(self, mat):
        """Assigns a `sbmc.scene_generator.Material` to this object.

        Args:
            mat(sbmc.scene_generator.Material): material to attach.
        """
        self.material = mat

    def pbrt(self):
        """Outputs PBRTv2 string representation.

        Returns:
            str: pbrt format.
        """
        return ""


class Sphere(Shape):
    """Sphere primitive.

    Args:
        radius(float): sphere radius.
    """
    def __init__(self, radius=1):
        super(Sphere, self).__init__()
        self.radius = radius

    def pbrt(self):
        """Outputs PBRTv2 string representation.

        Returns:
            str: pbrt format.
        """
        out = super(Sphere, self)._prefix()
        out += 'Shape "sphere" "float radius" [{}]\n'.format(self.radius)
        out += super(Sphere, self)._suffix()
        return out


class Cylinder(Shape):
    """Cylinder primitive.

    Args:
        radius(float): cylinder radius.
        height(float): cylinder height.
    """
    def __init__(self, radius=1, height=1):
        super(Cylinder, self).__init__()
        self.radius = radius
        self.height = height

    def pbrt(self):
        s = super(Cylinder, self)._prefix()
        s += 'Shape "cylinder" "float radius" [{}]\n'.format(self.radius)
        s += '"float zmin" [{}] "float zmax" [{}]\n'.format(
            -self.height*0.5, 0.5*self.height)
        s += super(Cylinder, self)._suffix()
        return s


class ExternalGeometry(Shape):
    """External .pbrt geometry that can be included in a scene.

    Args:
        path(str): path to the external geometry .pbrt file.
    """
    def __init__(self, path):
        super(ExternalGeometry, self).__init__()
        self.path = path

    def pbrt(self):
        s = super(ExternalGeometry, self)._prefix()
        s += 'Include "{}"\n'.format(self.path)
        s += super(ExternalGeometry, self)._suffix()
        return s


class TriangleMesh(Shape):
    """Generic triangle mesh.

    Args:
        indices(np.array[m, 3], int): vertex indices corresponding to
            triangles.
        points(np.array[n, 3], floats): vertex positions (xyz).
    """
    def __init__(self, indices, points):
        super(TriangleMesh, self).__init__()
        self.indices = indices
        self.points = points

    def pbrt(self):
        s = super(TriangleMesh, self)._prefix()
        idx = " ".join(map(str, list(self.indices.ravel())))
        points = " ".join(map(str, list(self.points.ravel())))
        s += 'Shape "trianglemesh" "integer indices" [{}]\n'.format(idx)
        s += '"point P" [{}]\n'.format(points)
        s += super(TriangleMesh, self)._suffix()
        return s


class Plane(TriangleMesh):
    """A 1-square YZ-plane centered at the origin.
    Args:
        scale(float): scale factor
    """

    def __init__(self, scale=1):
        pts = scale*np.array([
            [0, -0.5, -0.5],
            [0, -0.5, 0.5],
            [0, 0.5, 0.5],
            [0, 0.5, -0.5]
        ])
        idx = np.array([
            [0, 1, 2],
            [2, 3, 0],
        ])
        super(Plane, self).__init__(idx, pts)
