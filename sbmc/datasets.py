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
"""Dataset interfaces for Sample-Based Monte Carlo denoising."""
import os
import struct

import lz4.frame
import numpy as np
from torch.utils.data import Dataset, ConcatDataset

import ttools


LOG = ttools.get_logger(__name__)

__all__ = ["TilesDataset", "FullImagesDataset", "MultiSampleCountDataset"]


class TilesDataset(Dataset):
    """A dataset class that fetches samples from tiles stored in .bin files.

    This data class can function in two modes:

      1. Filelist mode: `path` is the path to a .txt files containing a list of
         .bin filenames

      2. Folder mode: `path`is the path to a root folder containing scenes.
         Each scene is a folder containing .bin files that correspond to image
         tiles for that scene. E.g.:

            + root

              + scene0

                + scene0_tile0.bin

                + scene0_tile1.bin

                + ...

              + scene1

                + scene1_tile0.bin

                + scene1_tile1.bin

                + ...

              + ...

    The class expects one tile per .bin file. The bin files are structured as
    follows:

    **metadata**

        (int32) version

        (int32) tile_size

        (int32) image_width

        (int32) image_height

        (int32) sample_count

        (int32) gt_sample_count

        (int32) sample_features

        (int32) pixel_features

        (int32) path_depth

    **global_features**

        (float32) focus_distance

        (float32) aperture_radius

        (float32) fov

        (float32) scene_radius

    **image_data**

        (float32[3]) diffuse

        (float32[3]) specular

        (float32[3]) albedo

        (float32[3]) normal

        (float32) depth

        (float32) visibility

        (float32) hasHit

    **for s in 1...spp**

        **sample s**

            (float32) dx: subpixel x coordinate

            (float32) dy: subpixel y coordinate

            (float32) lens_u: lens sampling coordinates

            (float32) lens_v: lens sampling coordinate

            (float32) t: sample's time coordinate

            (float32[3]) diffuse

            (float32[3]) specular

            (float32[3]) normal_first: at first geometric bounce

            (float32[3]) normal: at first diffuse bounce

            (float32) depth_first: at first geometric bounce

            (float32) depth: at first diffuse bounce

            (float32) visibility

            (float32) hasHit

            (float32[3]) albedo_first: at first geometric bounce

            (float32[3]) albedo: at first diffuse bounce

            (float32[4*TilesDataset.PATH_DEPTH]) p: sampling probabilities at each path vertex.

            (float32[2*TilesDataset.PATH_DEPTH]) ld: theta and phi. Spherical coordinates of the incoming light ray in camera-space coordinates but with origin at path vertex.

            (int16[TilesDataset.PATH_DEPTH]) bt: boolean flags packed as int16. Describe material interaction at path vertex.

    Args:
        path(str): path to the .txt file or folder containing scenes.
        spp(int): number of sample per pixels to load (the .bin files can
            contain more).
        load_coords(bool): if True, loads the sample coordinate features.
        load_gbuffer(bool): if True, loads the depth/normals/albedo features.
        load_p(bool): if True, loads the multi-importance sampling probability
            features.
        load_ld(bool): if True, loads the light direction features.
        load_bt(bool): if True, loads the bounce type features.
        mode(bool): mode to transform the samples to the format expected
            by:
                "kpcn": [Bako2017] 'Kernel-Predicting Convolutional Networks
                for Denoising Monte Carlo Renderings'
                <http://cvc.ucsb.edu/graphics/Papers/SIGGRAPH2017_KPCN/>,
                "sbmc": our sample format, with log-compressed radiance input,
                "raw": no transformation of the sample data.
    """

    FILELIST_MODE = 0
    """Mode to read filenames from a .txt file"""
    FOLDERS_MODE = 1
    """Mode to list filenames from a folder"""

    PATH_DEPTH = 6
    """Number of vertices in the PBRT2 pathtracer (no russian roulette)"""

    N_BT_FEATURES = 5
    """Number of bounce type features per vertex along the path"""

    # Different preprocessing mode for the data
    SBMC_MODE = "sbmc"
    """Mode to preprocess the features to our expected format [Bako2017]"""
    RAW_MODE = "raw"
    """Mode to load the data without preprocessing"""
    KPCN_MODE = "kpcn"
    """Mode to preprocess the features for the format expected by [Bako2017]"""

    def __init__(self, path, spp=None, load_coords=True, load_gbuffer=True,
                 load_p=True, load_ld=True, load_bt=True,
                 mode="sbmc"):

        if mode not in [TilesDataset.SBMC_MODE, TilesDataset.RAW_MODE,
                        TilesDataset.KPCN_MODE]:
            LOG.error("Unknown dataset loading mode %s", mode)
            raise RuntimeError("Unknown dataset loading mode %s" % mode)

        # Flags to select which features to load from disk
        self.mode = mode
        self.load_coords = load_coords
        self.load_gbuffer = load_gbuffer
        self.load_p = load_p
        self.load_ld = load_ld
        self.load_bt = load_bt
        self.count = 0  # number of data tiles
        self.spp = None

        # Override `raw` and `kpcn` modes don't need most features
        if self.mode != TilesDataset.SBMC_MODE:
            self.load_coords = False
            self.load_gbuffer = True
            self.load_p = False
            self.load_ld = False
            self.load_bt = False

        # Metadata
        self.version = None
        self.tile_size = None
        self.image_width = None
        self.image_height = None
        self.sample_count = None
        self.gt_sample_count = None
        self.sample_features = None
        self.pixel_features = None
        self.path_depth = None

        # Used in filelist io_mode
        self.root = None
        self.files = None

        # Used in folder io_mode, in this case we know tiles belong to a
        # specific scene
        self.scenes = None
        self.tiles = None
        self.indices = None

        # Initialize the dataset interface
        self._init_filelist(path)
        self._init_image_channels()
        self._init_feature_labels()
        self._init_metadata(spp)

    def _init_filelist(self, path):
        """Get the list of .bin files containing the samples.

        Args:
          path(str): path to a .txt containing the list of .bin files or
              a folder structure following this pattern:
                + root
                  + scene
                    + samples_tile0.bin
                    + samples_tile1.bin
                    + ...
        """
        if os.path.splitext(path)[-1] == ".txt":  # We have a sample list
            self.io_mode = TilesDataset.FILELIST_MODE
            LOG.debug("Dataset in filelist mode")
            # Paths should be relative to the folder containing the list.txt
            self.root = os.path.dirname(path)
            self.files = []
            with open(path) as fid:
                for l in fid.readlines():
                    self.files.append(os.path.join(self.root, l.strip()))
            self.count = len(self.files)
        elif os.path.isdir(path):  # We have a root folder
            self.io_mode = TilesDataset.FOLDERS_MODE
            LOG.debug("Dataset in folder mode")
            self.root = path
            # The scenes are subdirectories of the root
            scenes = [os.path.join(path, d) for d in
                      sorted(os.listdir(self.root))]
            self.scenes = [s for s in scenes if os.path.isdir(s)]
            self.tiles = {}
            self.indices = {}
            idx = 0
            for s in self.scenes:
                self.tiles[s] = []
                beg_idx = idx
                files = sorted(os.listdir(s))
                for f in files:
                    if os.path.splitext(f)[-1] == ".bin":
                        self.tiles[s].append(os.path.join(s, f))
                        idx += 1
                self.indices[s] = (beg_idx, idx)
            self.count = idx
        else:
            LOG.error("Unknown dataset format at path %s, maybe the folder is"
                      " empty?" % path)
            raise RuntimeError("Incorrect data path.")

        if self.count == 0:
            LOG.error("Dataset is empty, please check the file format / folder"
                      "structure.")
            raise RuntimeError("Empty dataset")

    def _init_image_channels(self):
        """Set the labels for the pixel data."""
        self.image_channels = ["diffuse_r", "diffuse_g", "diffuse_b",
                               "specular_r", "specular_g", "specular_b",
                               "albedo_r", "albedo_g", "albedo_b", "normal_x",
                               "normal_y", "normal_z", "depth", "visibility",
                               "hasHit"]

    def _init_feature_labels(self):
        """Set the labels for the sample data, depending config flags."""
        self.valid_versions = [20181212, 20190401]
        self.glabels = ["aperture_radius", "focus_distance", "fov"]

        # Should be 27 when all features are activated
        self.labels = []

        # Sample coordinates [0, 5[
        if self.load_coords:
            self.labels += ["dx", "dy", "lens_u", "lens_v", "t"]

        # We always use radiance [5, 11[
        self.labels += ["diffuse_r", "diffuse_g", "diffuse_b", "specular_r",
                        "specular_g", "specular_b"]

        # Add depth, normals, albedo, visibility and hit
        if self.load_gbuffer:
            self.labels += [
                "normal_first_x", "normal_first_y", "normal_first_z",
                "normal_x", "normal_y", "normal_z",
                "depth_first",
                "depth",
                "visibility",
                "hasHit",
                "albedo_first_r", "albedo_first_g", "albedo_first_b",
                "albedo_r", "albedo_g", "albedo_b",
            ]

        # 55 features for the importance sampling probabilities
        if self.load_p:
            # 4 values per vertex
            self.labels += ["p"]*TilesDataset.PATH_DEPTH*4

        # Light direction information in screen space spherical coordinates
        if self.load_ld:
            for i in range(TilesDataset.PATH_DEPTH):
                self.labels.append("ld_theta_%d" % i)
                self.labels.append("ld_phi_%d" % i)

        # Type of bounce at each vertex in the path
        if self.load_bt:
            for txt in ["reflection", "transmisson", "diffuse", "glossy",
                        "specular"]:
                for i in range(TilesDataset.PATH_DEPTH):
                    self.labels.append("bt_%s_%d" % (txt, i))

    def _init_metadata(self, spp):
        """Initializes metadata about this dataset from the first tile."""
        if len(self) == 0:
            LOG.error("This dataset contains no sample.")
            raise RuntimeError("Empty dataset")

        fname = self._filename(0)  # fetch first element to initialize meta
        with open(fname, "rb") as fid:
            try:
                self._read_globals_and_meta(fid)
            except Exception as e:
                LOG.error("Could not read %s", fname)
                raise e

        if spp is None:
            self.spp = self.sample_count
        else:
            if spp > self.sample_count:
                LOG.error("Requested %d samples, which is higher that what"
                          "the data has %d", spp, self.sample_count)
                raise RuntimeError("Requested too many samples.")
            self.spp = spp

    def __len__(self):
        return self.count

    def _filename(self, idx):
        """Gets the filename of the tile indexed.

        Args:
            idx(int): tile index in the data list.
        """
        if self.io_mode == TilesDataset.FOLDERS_MODE:
            scene = next(k for k in self.scenes if self.indices[k][1] > idx)
            tile_idx = idx - self.indices[scene][0]
            fname = self.tiles[scene][tile_idx]
        else:
            fname = self.files[idx]
        return fname

    def __getitem__(self, idx):
        """Get preprocessed tile with index idx.

        Args:
            idx(int): tile index.

        Returns:
            see `_preprocess_standard` and `_preprocess_kpcn`

        """
        sample = self._get_raw_data(idx)
        if self.mode == TilesDataset.KPCN_MODE:
            sample = self._preprocess_kpcn(sample)
        elif self.mode == TilesDataset.SBMC_MODE:
            sample = self._preprocess_standard(sample)
        else:
            assert self.mode == TilesDataset.RAW_MODE, "invalid mode provided"

        return sample

    @property
    def num_features(self):
        if self.mode == TilesDataset.KPCN_MODE:
            return 27
        else:
            return len(self.labels)

    @property
    def num_global_features(self):
        if self.mode == TilesDataset.KPCN_MODE:
            return 0
        else:
            return len(self.glabels)

    def _get_raw_data(self, idx):
        """Loads tile with index idx.

        Args:
            idx(int): tile index in the data list.

        Returns:
            sample(dict) with keys:
              "block_x": (int) x coordinate of the tile within fullres image.
              "block_y": (int) y coordinate of the tile within fullres image.
              "image_data": (np.array[15, h, w]) raw pixel-averages features
                  for the grountruth.
              "image_data_var": (np.array[15, h, w]) raw pixel-variance
                  features for the grountruth.
              "spp":  (np.array[1, 1, 1]) number of spp for this input
              "path": (str) path to the .bin file.
              "target_image": (np.array[3, h, w]) ground truth, with many
                  samples.
              "low_spp": (np.array[3, h, w]) low sample count image (average
                  of samples).
              "global_features": (np.array[3, 1, 1]) global features for the
                  scene from which the tile was extracted.
              "features": (np.array[spp, nfeatures, h, w]) samples features.
              "radiance": (np.array[spp, 3, h, w]) radiance samples.
        """
        fname = self._filename(idx)
        with open(fname, "rb") as fid:
            try:
                try:
                    gfeatures = self._read_globals_and_meta(fid)
                except struct.error as e:
                    LOG.error("reading meta for file %s failed", fname)
                    raise e

                try:
                    sample = self._read_data(fid, gfeatures)
                except RuntimeError as e:
                    LOG.error("reading data from file %s failed", fname)
                    raise e
                except struct.error as e:
                    LOG.error("reading data from file %s failed", fname)
                    raise e
                sample["path"] = fname
            except Exception as e:
                LOG.error("could not read %s", fname)
                raise e

        return sample

    def _rcheck(self, field, value):
        """Reads a field, check its value correspond to that of the metadata.

        The first calls sets the metadata for this object.
        """
        if getattr(self, field) is not None:
            if getattr(self, field) != value and field:
                LOG.error("metadata do not match, got %s for field %s, should"
                          " be %s" % (value, field, getattr(self, field)))
                raise ValueError("Metadata do not match.")
        else:
            if field == "version" and value not in self.valid_versions:
                raise ValueError("Version unsupported: got %s, valid are %s" %
                                 (value, self.valid_versions))

            setattr(self, field, value)

        if field == "path_depth" and value != TilesDataset.PATH_DEPTH:
            LOG.error("The path depth of the rendered data shoud be %d",
                      TilesDataset.PATH_DEPTH)
            raise RuntimeError("Incorrect path depth in the data")

    def _read_globals_and_meta(self, fid):
        """Read the metadata and global features from the tile's header.

        Args:
          fid(filepointer): the open file id of the .bin tile.

        Returns:
          gfeatures(dict) with keys:
            'focus_distance':  distance the camera focused at.
            'aperture_radius':  aperture size of the camera.
            'fov':  field of view of the vamera.
            'scene_radius':  radius of the scene's bounding sphere.
        """

        self._rcheck("version", struct.unpack('i', fid.read(4))[0])
        self._rcheck("tile_size", struct.unpack('i', fid.read(4))[0])
        self._rcheck("image_width", struct.unpack('i', fid.read(4))[0])
        self._rcheck("image_height", struct.unpack('i', fid.read(4))[0])
        self._rcheck("sample_count", struct.unpack('i', fid.read(4))[0])
        self._rcheck("gt_sample_count", struct.unpack('i', fid.read(4))[0])
        self._rcheck("sample_features", struct.unpack('i', fid.read(4))[0])
        self._rcheck("pixel_features", struct.unpack('i', fid.read(4))[0])
        self._rcheck("path_depth", struct.unpack('i', fid.read(4))[0])

        # Global scene features, those are scene-dependent
        gfeatures = {}
        gfeatures["focus_distance"] = struct.unpack('f', fid.read(4))[0]
        gfeatures["aperture_radius"] = struct.unpack('f', fid.read(4))[0]
        gfeatures["fov"] = struct.unpack('f', fid.read(4))[0]
        gfeatures["scene_radius"] = struct.unpack('f', fid.read(4))[0]

        # If the focus distance is not set when not using DoF, the value
        # defaults to NaN: this breaks the network. So set it to 0.
        if gfeatures["aperture_radius"] == 0:
            gfeatures["focus_distance"] = 0.0

        if gfeatures["focus_distance"] < 0:
            LOG.error("Focus distance is negative: data is corrupt.")
            raise RuntimeError("Incorrect focus distance feature.")
        if gfeatures["aperture_radius"] < 0:
            LOG.error("Aperture radius is negative: data is corrupt.")
            raise RuntimeError("Incorrect aperture radius feature.")
        if gfeatures["fov"] < 0:
            LOG.error("Field of view is negative: data is corrupt.")
            raise RuntimeError("Incorrect field of view feature.")
        if gfeatures["scene_radius"] < 0:
            LOG.error("Scene radius is negative: data is corrupt.")
            raise RuntimeError("Incorrect scene radius.")

        return gfeatures

    def __repr__(self):
        s = "Dataset v%d\n" % self.version
        s += "  .image size: %dx%d\n" % (self.image_width, self.image_height)
        s += "  .block size: %d\n" % self.tile_size
        s += "  .sample count: %d (of %d)\n" % (self.spp, self.sample_count)
        s += "  .suffix length: %d\n" % self.pixel_features
        s += "  .sample feature size: %d\n" % self.sample_features
        s += "  .path depth: %d\n" % self.path_depth
        if not self.load_p:
            s += "  .NO probabilities\n"
        if not self.load_ld:
            s += "  .NO light direction\n"
        if not self.load_bt:
            s += "  .NO bounce types\n"
        s += "  .total feature count: %d (and %d global)\n" % (
            len(self.labels), len(self.glabels))
        return s

    def _read_compressed(self, fid):
        """The sample and image data is compressed using lz4, decompress it.

        Args:
            fid(filepointer): pointer to the open .bin file.
        """
        nbytes = struct.unpack('i', fid.read(4))[0]
        buf = fid.read(nbytes)
        data = lz4.frame.decompress(buf)
        return data

    def _read_data(self, fid, gfeatures):
        """Reads and decompress data.

        Args:
          fid(filepointer): pointer to the .bin file open.
          gfeatures(dict): global features previously loaded from that tile.

        Returns:
          dict: see "TilesDataset.__getitem__".
        """
        sample_data = {}

        ts = self.tile_size

        sample_data['block_x'] = struct.unpack('i', fid.read(4))[0]
        sample_data['block_y'] = struct.unpack('i', fid.read(4))[0]

        # Add global features
        gf = [gfeatures[k] for k in self.glabels]
        gf = np.reshape(
            np.array(gf), [len(self.glabels), 1, 1]).astype(np.float32)
        sample_data["global_features"] = gf

        # Read pixel data -----------------------------------------------------
        buf = self._read_compressed(fid)
        image_data = np.frombuffer(buf, dtype=np.float32)
        image_data = np.reshape(image_data, [self.pixel_features, ts, ts])

        # Load image data (mean/varaiance)
        nchans = image_data.shape[0] // 2
        sample_data['image_data'] = image_data[:nchans, ...]
        sample_data['image_data_var'] = image_data[nchans:2*nchans, ...]

        # Set the regression target ro be diffuse + specular
        diffuse = sample_data["image_data"][:3]
        specular = sample_data["image_data"][3:6]
        sample_data['target_image'] = diffuse + specular
        # ---------------------------------------------------------------------

        sample_features = self.sample_features
        ld_features = self.path_depth*2  # for the light directions
        p_features = self.path_depth*4
        bt_features = self.path_depth
        total_features = sample_features

        if self.load_ld:
            total_features += ld_features
        if self.load_p:
            total_features += p_features
        if self.load_bt:
            total_features += bt_features*TilesDataset.N_BT_FEATURES

        # We load only the first self.spp samples
        sample_data["spp"] = self.spp*np.ones((1, 1, 1), dtype=np.int32)

        if self.spp > 0:
            # Initialize sample buffer to 0s
            samples = np.zeros((self.spp*total_features*ts*ts), np.float32)
            samples_buf = memoryview(samples)
            offset = 0  # write position in the target samples buffer

            # byte sizes to read
            sample_bytes = sample_features*ts*ts*4  # 4 for float32
            ld_bytes = ld_features*ts*ts*4  # 4 for float32
            p_bytes = p_features*ts*ts*4  # 4 for float32
            int16_bytes = bt_features*ts*ts*2  # 2 for int16

            for s in range(self.spp):
                buf = self._read_compressed(fid)
                buf_offset = 0  # read position in the input buffer

                # Read the default sample features
                count = sample_features*ts*ts
                samples_buf[offset:offset+count] = np.frombuffer(
                    buf[buf_offset:buf_offset+sample_bytes], dtype=np.float32)
                offset += count
                buf_offset += sample_bytes

                # Read probabilities features
                if self.load_p:
                    count = p_features*ts*ts
                    samples_buf[offset:offset+count] = np.frombuffer(
                        buf[buf_offset:buf_offset+p_bytes], dtype=np.float32)
                    offset += count
                buf_offset += p_bytes  # advance read offset

                # Read light_direction features
                if self.load_ld:
                    count = ld_features*ts*ts
                    samples_buf[offset:offset+count] = np.frombuffer(
                        buf[buf_offset:buf_offset+ld_bytes], dtype=np.float32)
                    offset += count
                buf_offset += ld_bytes  # advance read offset

                # Parse reflection types
                if self.load_bt:
                    int_data = np.frombuffer(
                        buf[buf_offset:buf_offset+int16_bytes], dtype=np.int16)
                    int_data = np.reshape(int_data, [1, bt_features, ts, ts])

                    # Decode bounce type
                    is_reflection = np.bitwise_and(int_data, 1).astype(np.bool)
                    is_transmission = np.bitwise_and(
                        int_data, 1 << 1).astype(np.bool)
                    is_diffuse = np.bitwise_and(
                        int_data, 1 << 2).astype(np.bool)
                    is_glossy = np.bitwise_and(
                        int_data, 1 << 3).astype(np.bool)
                    is_specular = np.bitwise_and(
                        int_data, 1 << 4).astype(np.bool)

                    int_data = np.concatenate([is_reflection, is_transmission,
                                               is_diffuse, is_glossy,
                                               is_specular], 1)

                    int_data = int_data.astype(np.float32)

                    count = int_data.size
                    samples_buf[offset:offset+count] = int_data.ravel()

                    offset += count

            # Reshape to final shape
            samples = np.reshape(samples, (self.spp, total_features, ts, ts))

            # Remove coordinates or G-buffer features if requested
            keep_idx = []
            if self.load_coords:
                keep_idx += list(range(0, 5))
            keep_idx += list(range(5, 11))  # always keep radiance
            if self.load_gbuffer:
                keep_idx += list(range(11, 27))
            if total_features > 27:  # keep extra features
                keep_idx += list(range(27, total_features))
            samples = samples[:, keep_idx]

            # Store sample features
            sample_data["features"] = samples

        if self.spp > 0:
            i_diffuse = self.labels.index("diffuse_r")
            i_specular = self.labels.index("specular_r")
            diffuse = samples[:, i_diffuse:i_diffuse + 3, ...]
            specular = samples[:, i_specular:i_specular+3, ...]

            # Our model works on the complete radiance, sum diffuse + specular
            sample_data["radiance"] = diffuse + specular

            # Average sample for the "low sample count" image (mainly for
            # display)
            sample_data['low_spp'] = sample_data["radiance"].mean(0)
        else:
            LOG.warning("No sample requested, setting low_spp to 0")
            sample_data["low_spp"] = np.zeros(
                sample_data["target_image"].shape)

        sample_data["scene_radius"] = gfeatures["scene_radius"]

        return sample_data

    def _preprocess_standard(self, sample):
        """Transforms the raw samples.

        Sums the diffuse and specular contributions and compresses the radiance
        used as input to the kernel-predictor with a log curve.

        Args:
            sample(dict): raw samples (see `_get_raw_data`).
        Returns:
            dict with the same keys but with the input features transformed.
        """

        feats = sample["features"]

        # Locate diffuse and specular components
        idx = self.labels.index("diffuse_r")
        idx2 = self.labels.index("specular_r")

        # Radiance should be positive
        diffuse = np.maximum(feats[:, idx:idx+3], 0)
        specular = np.maximum(feats[:, idx2:idx2+3], 0)

        # Replace diffuse by diffuse+specular
        total = diffuse + specular

        # Tonemap radiance,
        feats[:, idx:idx+3] = np.log(1 + total) / 10.0
        feats[:, idx2:idx2+3] = np.log(1 + specular) / 10.0

        # All other keys are pass through
        ret = {}
        for k in sample.keys():
            ret[k] = sample[k]

        # But we override the features with our transformed input.
        ret["features"] = feats

        return ret

    def _preprocess_kpcn(self, sample):
        """Transform the raw samples into the format expected by [Bako2017].
        """
        src_f = sample["features"]
        tgt = sample["image_data"]

        spp = src_f.shape[0]

        # Normalize depth
        idx = self.labels.index("depth")
        depth = src_f[:, idx:idx+1].mean(0)
        depth_v = src_f[:, idx:idx+1].var(0)
        max_depth = depth.max()
        if max_depth > 0:
            depth /= max_depth
            depth_v /= max_depth*max_depth*spp
        depth = np.clip(depth, 0, 1)

        # Extract albedo and radiance
        idx = self.labels.index("albedo_r")
        idx2 = self.image_channels.index("albedo_r")
        albedo = src_f[:, idx:idx+3].mean(0) + 0.00316
        albedo_v = src_f[:, idx:idx+3].var(0).mean(0, keepdims=True) / spp
        albedo_r = tgt[idx2:idx2+3] + 0.00316
        albedo_sqr = (albedo*albedo).mean(0, keepdims=True)

        idx = self.labels.index("diffuse_r")
        idx2 = self.image_channels.index("diffuse_r")
        diffuse = np.maximum(src_f[:, idx:idx+3].mean(0), 0)
        diffuse_v = src_f[:, idx:idx+3].var(0).mean(0, keepdims=True) / spp
        diffuse_r = np.maximum(tgt[idx2:idx2+3], 0)

        idx = self.labels.index("specular_r")
        idx2 = self.image_channels.index("specular_r")
        specular = np.maximum(src_f[:, idx:idx+3].mean(0), 0)
        specular_v = src_f[:, idx:idx+3].var(0).mean(0, keepdims=True) / spp
        specular_r = np.maximum(tgt[idx2:idx2+3], 0)

        # Transform diffuse and its variance
        diffuse /= albedo
        diffuse_v /= albedo_sqr

        # Transform specular and its variance
        specular = np.log(1 + specular)
        specular_v /= ((1+specular)*(1+specular)).mean(0, keepdims=True) + 1e-5

        # Extract normals
        idx = self.labels.index("normal_x")
        normals = src_f[:, idx:idx+3].mean(0)
        normals_v = src_f[:, idx:idx+3].var(0).mean(0, keepdims=True) / spp

        # Take gradients
        normals_g = self._gradients(normals)
        depth_g = self._gradients(depth)
        albedo_g = self._gradients(albedo)
        specular_g = self._gradients(specular)
        diffuse_g = self._gradients(diffuse)

        # Transformed targets
        specular_r = np.log(1 + np.maximum(specular_r, 0))
        diffuse_r = diffuse_r / albedo_r

        out_sample = {
            "kpcn_diffuse_in": np.concatenate(
                [diffuse, normals_g, normals_v, depth_g, depth_v, albedo_g,
                 albedo_v, diffuse_g, diffuse_v], 0),
            "kpcn_specular_in": np.concatenate(
                [specular, normals_g, normals_v, depth_g, depth_v, albedo_g,
                 albedo_v, specular_g, specular_v], 0),
            "kpcn_diffuse_buffer": diffuse,
            "kpcn_specular_buffer": specular,
            "kpcn_albedo": albedo,
        }

        for k in ["target_image", "low_spp", "spp", "block_x", "block_y"]:
            out_sample[k] = sample[k]

        return out_sample

    def _gradients(self, buf):
        """Compute the xy derivatives of the input buffer.

        This helper is used in the Di

        Args:
            buf(np.array)[c, h, w]: input image-like tensor.

        Returns:
            (np.array)[2*c, h, w]: horizontal and vertical gradients of buf.
        """
        dy = buf[:, 1:] - buf[:, :-1]
        dx = buf[:, :, 1:] - buf[:, :, :-1]
        dx = np.pad(dx, [[0, 0], [0, 0], [1, 0]], mode="constant")
        dy = np.pad(dy, [[0, 0], [1, 0], [0, 0]], mode="constant")
        return np.concatenate([dx, dy], 0)


class FullImagesDataset(Dataset):
    """A dataset class that fetches full-res samples buffers.

    This data interface assembles multiple tiles corresponding to each full
    size image.

    This class needs the path to a folder structured as such:

    + root

      + scene0

        + scene0_tile0.bin

        + scene0_tile1.bin

        + ...

      + scene1

        + scene1_tile0.bin

        + scene1_tile1.bin

        + ...

      + ...

    More details in :class:`TilesDataset`
    """
    def __init__(self, *args, **kwargs):
        self.tiles_dset = TilesDataset(*args, **kwargs)
        if self.tiles_dset.io_mode != TilesDataset.FOLDERS_MODE:
            LOG.error("Full image dataset needs to point to a folder"
                      " containing scenes, got '%s'.", args[0])
            raise RuntimeError("TilesDataset should be in folder mode.")
        self.scenes = self.tiles_dset.scenes

    def __len__(self):
        return len(self.scenes)

    def __repr__(self):
        return self.tiles_dset.__repr__()

    def get_scene_name(self, idx):
        """Returns the name (folder name) of scene 'idx'.

        Args:
            idx(int): scene index in the list.
        """
        scene = self.scenes[idx]
        return scene

    def __getitem__(self, idx):
        # select a scene
        scene = self.scenes[idx]

        # Index to the first tile for this scene
        start_idx, end_idx = self.tiles_dset.indices[scene]

        # Set the metadata and get the first tile to discover the sample keys
        first = self.tiles_dset[start_idx]

        tile_size = self.tiles_dset.tile_size
        width = self.tiles_dset.image_width
        height = self.tiles_dset.image_height

        # Initialize global features and empty tensor for other parameters
        sample = {}
        tensor_keys = []
        for k in first.keys():
            if k in ["global_features", "scene_radius"]:
                sample[k] = first[k]
            elif type(first[k]) == np.ndarray:
                tensor_keys.append(k)
                shape = list(first[k].shape)
                shape[-2] = height
                shape[-1] = width
                sample[k] = np.zeros(shape, dtype=first[k].dtype)

        # Fill in the main buffer with data from each tile
        for tidx in range(start_idx, end_idx):
            tile = self.tiles_dset[tidx]
            bx, by = tile['block_x'], tile['block_y']
            for k in tensor_keys:
                sample[k][..., by:by+tile_size, bx:bx+tile_size] = tile[k]

        return sample

    @property
    def num_features(self):
        return self.tiles_dset.num_features

    @property
    def num_global_features(self):
        return self.tiles_dset.num_global_features

    @property
    def spp(self):
        return self.tiles_dset.spp

    @property
    def sample_count(self):
        return self.tiles_dset.sample_count

    @property
    def gt_sample_count(self):
        return self.tiles_dset.gt_sample_count

    @property
    def load_p(self):
        return self.tiles_dset.load_p

    @property
    def load_ld(self):
        return self.tiles_dset.load_ld

    @property
    def load_bt(self):
        return self.tiles_dset.load_bt

    @property
    def labels(self):
        return self.tiles_dset.labels

    @property
    def glabels(self):
        return self.tiles_dset.glabels

    @property
    def version(self):
        return self.tiles_dset.version

    @property
    def image_channels(self):
        return self.tiles_dset.image_channels


class MultiSampleCountDataset(ConcatDataset):
    """A tile dataset with variable number of samples per pixel.

    Warning: batching will be tricky since the sample dimension is variable,
    use a batch size = 1.

    For more details, see :class:`TilesDataset`.
    """
    def __init__(self, *args, **kwargs):
        spp = kwargs.get("spp", None)
        if spp is None:
            LOG.error("MultiSampleCountDataset requires a number of spps")
            raise RuntimeError("spp not provided.")
        datasets = []
        if spp < 2:
            LOG.error("MultiSampleCountDataset needs at least 2spp")
            raise RuntimeError("spp too low to randomize sample count, should"
                               "be at least 2.")
        # Instantiates multiple datasets with varying sample count
        for _s in range(2, spp+1):
            kwargs["spp"] = _s
            datasets.append(TilesDataset(*args, **kwargs))
        super(MultiSampleCountDataset, self).__init__(datasets)

        self.labels = datasets[0].labels
        self.glabels = datasets[0].glabels
        self.version = datasets[0].version
        self.num_features = datasets[0].num_features
        self.num_global_features = datasets[0].num_global_features
