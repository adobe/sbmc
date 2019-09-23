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
"""A collection of random scene generators."""
import os

from bridson import poisson_disc_samples as pdisc
import numpy as np

import ttools

from .scene import Camera
from .converters import ObjConverter
from . import geometry
from . import randomizers
from . import xforms


__all__ = ["OutdoorSceneGenerator"]


class SceneGenerator():
    """Base class for the random scene generators.

    Args:
        envmaps(list of str): absolute paths to .pfm HDR images to use as
            envmaps.
        textures(list of str): absolute paths to .tga images used to texture
            scene objects.
        models(list of str): absolute paths to .obj files containing the
            geometry.
        pbrt_converter(str): path to the PBRTv2 executable that converts .obj
            to the .pbrt text format.
    """
    def __init__(self, envmaps, textures, models, pbrt_converter):
        self._envmaps = envmaps
        self._textures = textures
        self._current_textures = []
        self._models = models
        self._converter = ObjConverter(pbrt_converter)
        self._randomize_textures()
        self._log = ttools.get_logger(self.__class__.__name__)

    def __str__(self):
        return self.__class__.__name__

    def _randomize_textures(self):
        """Shuffle the available textures. Stores them in
        `self._current_textures`.
        """
        if self._textures:
            self._current_textures = list(
                np.random.choice(self._textures,
                                 size=(min(30, len(self._textures)), ),
                                 replace=False))
        else:
            self._current_textures = []


class OutdoorSceneGenerator(SceneGenerator):
    """Generates random outdoor scene with a envmap and a ground plane."""
    def _sample_camera(self):
        r_cam = np.random.uniform(1.0, 2.5)
        theta_cam = np.random.uniform(0, 2*np.pi)
        z_cam = np.random.uniform(0.01, 0.1)
        cam_fov = np.random.uniform(15, 65)

        cam_up = np.random.uniform(size=(3,))
        cam_pos = np.array([r_cam*np.cos(theta_cam), r_cam*np.sin(theta_cam),
                            z_cam])
        cam_target = np.random.uniform(0, 1, size=3)
        cam_target[2] = np.random.uniform(1., 2.)*z_cam

        cam_params = {"position": list(cam_pos), "target": list(cam_target),
                      "up": list(cam_up), "fov": cam_fov}
        return cam_params

    def _obj_pos(self, cam):
        factor = 5

        # Camera's fulcrum
        cam_direction = np.array(cam["target"][:2]) - \
            np.array(cam["position"][:2])
        cam_direction /= np.linalg.norm(cam_direction)  # normalized
        cam_halfangle = 1.1*cam["fov"]/180*np.pi  # add 10% for geometry bounds
        c, s = np.cos(cam_halfangle), np.sin(cam_halfangle)
        rot = np.matrix([[c, -s], [s, c]])
        u1 = factor*np.linalg.inv(rot).dot(cam_direction)
        u2 = factor*rot.dot(cam_direction)
        xform = np.vstack([u1, u2]).T

        radius = np.random.uniform(0.13, 0.28)
        scaled_radius = radius*factor

        # Place object centers
        xy = pdisc(width=1, height=1, r=radius/factor)
        np.random.shuffle(xy)  # randomize order
        xx = [x_[0] for x_ in xy]
        yy = [x_[1] for x_ in xy]
        xy = np.vstack([xx, yy])
        # transform coordinates to the fulcrum of the camera
        xy = xform.dot(xy)

        # project along camera direction, reject objects that are too close or
        # too far
        proj = np.ravel(cam_direction.dot(xy))
        keep = np.logical_and(proj > 0.1*scaled_radius, proj < factor)
        xy = xy[:, keep]

        # keep max 50 objects
        nmax = 50
        if xy.shape[1] > nmax:
            xy = xy[:, :nmax]

        proj /= proj.max()

        # move origin at camera
        xy[0, :] += cam["position"][0]
        xy[1, :] += cam["position"][1]

        return xy, scaled_radius, proj

    def sample(self, scn, dst_dir, params=None):
        self._log.debug("Sampling new outdoor scene")
        self._randomize_textures()

        # Random camera
        do_dof = np.random.choice([True, False])
        do_mblur = np.random.choice([True, False])
        cam = self._sample_camera()

        if do_mblur:
            cam["shutterclose"] = 1.0

        if do_dof:
            aperture = _random_aperture()
        else:
            aperture = 0.0

        # Sample objects in the fulcrum of the camera
        self._log.debug("Sampling object positions")
        coords, radius, proj = self._obj_pos(cam)
        count = coords.shape[1]

        # Focus on one of the objects
        if count > 0:
            focus_at = np.random.randint(0, count)

        # Randomizes the number of possible object altitudes
        z_layers = np.random.poisson(0.5) + 1

        count_blurred = 0  # Counts the number of objects that have motion blur
        self._log.debug("Adding %d objects.", count)
        for o_idx in range(count):  # Populate the scene

            # If motion blur is activated, maybe blur this object
            this_mblur = do_mblur and np.random.choice([True, False])
            if this_mblur:
                count_blurred += 1

            # Sample a motion vector
            mvec_r = np.random.uniform(0.00, 2)*radius
            mvec_dir = np.random.uniform(size=(3,))
            mvec_dir /= np.linalg.norm(mvec_dir)
            mvec = mvec_dir*mvec_r

            # Fetch a random object from the library
            dst = os.path.join(dst_dir, "geometry")
            mdl = np.random.choice(self._models)
            pbrt_objects = self._converter(mdl, dst)

            # Randomize the scale and position
            scl = radius*np.random.exponential(0.5)*np.ones((3,))
            z_idx = np.random.randint(0, z_layers)
            altitude = np.random.normal(0.1, 0.2)
            position = [coords[0, o_idx], coords[1, o_idx], altitude]

            # Create a ground plane
            plane = geometry.Plane(20)
            xforms.rotate(plane, [0, 1, 0], 90)
            material = randomizers.random_material(
                id="floormat", textures_list=self._current_textures)
            plane.assign_material(material)
            scn.shapes.append(plane)
            scn.materials.append(material)

            # Compute the focus distance and update the camera paramters
            if do_dof and z_idx == 0 and o_idx == focus_at:
                dist = np.linalg.norm(
                    np.array(cam["position"])-np.array(position))
                if dist > 0:
                    cam["focaldistance"] = dist
                    cam["lensradius"] = aperture

            # Obj files may contain multiple pieces, add them all
            for obj in pbrt_objects:
                geom = geometry.ExternalGeometry(os.path.join("geometry",
                                                              obj.path))
                xforms.rotate(geom, np.random.uniform(size=(3,)),
                              np.random.uniform(0, 360))
                xforms.rotate(geom, np.random.uniform(size=(3,)),
                              np.random.uniform(0, 360))
                xforms.scale(geom, scl)
                xforms.translate(geom, position)

                # Get a material for this piece
                material = randomizers.random_material(
                    id=obj.material.id, textures_list=self._current_textures)
                scn.materials.append(material)

                if this_mblur:
                    xforms.translate(geom, mvec, target="end")

                scn.shapes.append(geom)

        self._log.debug("%s objects have motion blur", count_blurred)

        # Add an envmap
        env = randomizers.random_envmap(self._envmaps, nsamples=8)
        xforms.rotate(env, [0, 0, 1], np.random.uniform(0, 360))
        scn.lights.append(env)

        # Attach the camera to the scene
        scn.camera = Camera(**cam)

        self._log.debug("Camera parameters %s. Motion blur? %s DoF? %s",
                        scn.camera, do_mblur, do_dof)
        if do_mblur:
            if (scn.camera.shutteropen != 0.0 or
                    scn.camera.shutterclose != 1.0):
                return False
        if do_dof:
            if (not scn.camera.lensradius > 0.0 or not
                    scn.camera.focaldistance > 0.0):
                return False

        self._log.debug("Generated Outdoor scene")

        return True


def _random_aperture(min_=0.001, max_=0.05):
    """Sample a camera aperture value, uniformly in log domain).
    Args:
        min_(float): smallest aperture.
        max_(float): largest aperture.
    """
    log_aperture = np.random.uniform(np.log(min_), np.log(max_))
    aperture = np.exp(log_aperture)
    return aperture
