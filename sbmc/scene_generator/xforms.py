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
"""Collection of geometric transforms."""

__all__ = ["translate", "rotate", "scale", "transform", "look_at"]


def translate(obj, t, target="default"):
    """Applies a translation to an object.

    Args:
        obj(sbmc.scene_generator.Shape): object to transform.
        t(list of 3 floats): x,y,z transform parameters.
        target(str): transformation channel to apply the transform to. Should
        be one of:
            - `default` (the default transform)
            - `start` (the transform at the opening of the camera shutter)
            - `end` (the transform at the closing of the camera shutter)

        This allows to animate the object for motion blur computation.
    """
    _push_xform(obj, Translation(t), target=target)


def rotate(obj, axis, angle, target="default"):
    """Applies a rotation to an object.

    Args:
        obj(sbmc.scene_generator.Shape): object to transform.
        axis(list of 3 floats): x,y,z of the rotation axis.
        angle(float): rotation angle in degrees.
        target(str): transformation channel to apply the transform to.

        Should be one of:
            - `default` (the default transform)
            - `start` (the transform at the opening of the camera shutter)
            - `end` (the transform at the closing of the camera shutter)

        This allows to animate the object for motion blur computation.
    """
    _push_xform(obj, Rotation(axis, angle), target=target)


def scale(obj, mag, target="default"):
    """Applies a rotation to an object.

    Args:
        obj(sbmc.scene_generator.Shape): object to transform.
        mag(list of 3 floats): x,y,z scaling factor.
        target(str): transformation channel to apply the transform to.

        Should be one of:
            - `default` (the default transform)
            - `start` (the transform at the opening of the camera shutter)
            - `end` (the transform at the closing of the camera shutter)

        This allows to animate the object for motion blur computation.
    """
    _push_xform(obj, Scale(mag), target=target)


def transform(obj, coeffs, target="default"):
    """Applies a generic 4x4 transform matrix.

    Args:
        obj(sbmc.scene_generator.Shape): object to transform.
        coeffs(list of 16 floats): coefficients of the 4x4 matrix in row-major
            ordering.
        target(str): transformation channel to apply the transform to.

        Should be one of:
            - `default` (the default transform)
            - `start` (the transform at the opening of the camera shutter)
            - `end` (the transform at the closing of the camera shutter)

        This allows to animate the object for motion blur computation.
    """
    _push_xform(obj, Mat44(coeffs), target=target)


def look_at(obj, src, dst, up, target="default"):
    """Applies a look at transform.

    Args:
        obj(sbmc.scene_generator.Shape): object to transform.
        src(list of 3 floats): x,y,z center of the `eye` of the look at
            transform.
        dst(list of 3 floats): x,y,z position of the point looked at.
        up(list of 3 floats): x,y,z coordinate of the up vector.
        target(str): transformation channel to apply the transform to.

        Should be one of:
            - `default` (the default transform)
            - `start` (the transform at the opening of the camera shutter)
            - `end` (the transform at the closing of the camera shutter)

        This allows to animate the object for motion blur computation.
    """
    _push_xform(obj, LookAt(src, dst, up), target=target)


class Transform():
    """Abstract transform made by composition of elementary transforms.

    Attributes:
        xforms(list of transforms): transforms to compose from back to front.
    """
    def __init__(self):
        self.xforms = []

    def push(self, xf):
        """Add a transform to the back of the chain.

        Args:
            xf(transform): transform to append.
        """
        self.xforms.append(xf)

    def pbrt(self):
        """Outputs PBRTv2 string representation.

        Returns:
            str: pbrt format.
        """
        s = ""
        for xf in reversed(self.xforms):
            s += xf.pbrt()
        return s


class Translation():
    """Translation transform.

    Args:
        t(list of 3 floats): x,y,z transform parameters.
    """
    def __init__(self, t):
        self.tx = t[0]
        self.ty = t[1]
        self.tz = t[2]

    def pbrt(self):
        s = "Translate {} {} {}\n".format(self.tx, self.ty, self.tz)
        return s


class LookAt():
    def __init__(self, src, dst, up):
        self.ex = src[0]
        self.ey = src[1]
        self.ez = src[2]
        self.lx = dst[0]
        self.ly = dst[1]
        self.lz = dst[2]
        self.ux = up[0]
        self.uy = up[1]
        self.uz = up[2]

    def pbrt(self):
        s = "LookAt {} {} {}  {} {} {}  {} {} {}\n".format(
            self.ex, self.ey, self.ez,
            self.lx, self.ly, self.lz,
            self.ux, self.uy, self.uz)
        return s


class Scale():
    """Scaling transform.

    Args:
        s(list of 3 floats): x,y,z scaling factor.
    """
    def __init__(self, s):
        self.sx = s[0]
        self.sy = s[1]
        self.sz = s[2]

    def pbrt(self):
        s = "Scale {} {} {}\n".format(self.sx, self.sy, self.sz)
        return s


class Rotation():
    """Rotation transform.

    Args:
        axis(list of 3 floats): x,y,z of the rotation axis.
        angle(float): rotation angle in degrees.
    """
    def __init__(self, axis, angle):
        self.x = axis[0]
        self.y = axis[1]
        self.z = axis[2]
        self.angle = angle

    def pbrt(self):
        s = "Rotate {} {} {} {}\n".format(self.angle, self.x, self.y, self.z)
        return s


class Mat44():
    """Generic 4x4 transform matrix.

    Args:
        coeffs(list of 16 floats): coefficients of the 4x4 matrix in row-major
            ordering.
    """
    def __init__(self, coeffs):
        assert len(coeffs) == 16
        self.coeffs = coeffs

    def pbrt(self):
        s = ("Transform [" + " {}"*16 + "]\n").format(*self.coeffs)
        return s


def __init_xform(obj, target="default"):
    """Initializes the transform stack of an object.
    Args:
        obj(sbmc.scene_generator.Shape): object to transform.
        target(str): transformation channel to apply the transform to.
            Should be one of:
                - `default` (the default transform)
                - `start` (the transform at the opening of the camera shutter)
                - `end` (the transform at the closing of the camera shutter)
            This allows to animate the object for motion blur computation.
    """
    if target == "default":
        if obj.xform is None:
            obj.xform = Transform()
    elif target == "start":
        if obj.start_xform is None:
            obj.start_xform = Transform()
    elif target == "end":
        if obj.end_xform is None:
            obj.end_xform = Transform()
    else:
        raise ValueError("unknown xform target {}".format(target))


def _push_xform(obj, t, target="default"):
    """Adds a transform to an object's transform stack.
    Args:
        obj(sbmc.scene_generator.Shape): object to transform.
        t(transform): the transform to add.
        target(str): transformation channel to apply the transform to.
            Should be one of:
                - `default` (the default transform)
                - `start` (the transform at the opening of the camera shutter)
                - `end` (the transform at the closing of the camera shutter)
            This allows to animate the object for motion blur computation.
    """
    __init_xform(obj, target=target)
    if target == "default":
        obj.xform.push(t)
    elif target == "start":
        obj.start_xform.push(t)
    elif target == "end":
        obj.end_xform.push(t)
    else:
        raise ValueError("unknown xform target %s, should be one of "
                         "`default`, `start`, `end`" % target)
