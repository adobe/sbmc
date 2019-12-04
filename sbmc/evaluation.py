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
"""Helpers to evaluate on the rendering results."""
import os

import numpy as np
import pyexr
import pandas as pd
from skimage.measure import compare_ssim as ssim

import ttools

LOG = ttools.get_logger(__name__)


def compute(ref_folder, output, methods, scenes, pad=21):
    """Compute metrics from .exr images and saves to disk.

    Args:
        ref_folder(str): path to the folder containing reference images.
        output(str): path to the .csv output file.
        methods(list of str): paths to the folders containing the .exr results,
            one for each method to evaluate. Accepts a list of methods or a
            .txt file listing the methods.
        scenes(str, list of str): name of the .exr scenes to include in the
            comparison. Accepts a list of filenames or a .txt file listing the
            filenames.
        pad(int): number of pixels to ignore on each side when computing
            the metrics.
    """
    scores = pd.DataFrame(columns=["method", "scene", "spp", "valid"] +
                          list(METRIC_LABELS.keys()))

    scenes = _parse_list_or_txt(scenes)
    methods = _parse_list_or_txt(methods)

    n_scenes = len(scenes)
    n_methods = len(methods)

    LOG.info("Evaluating %d scenes and %d methods", n_scenes, n_methods)

    filepaths = []

    if not os.path.splitext(output)[-1] == ".csv":
        mess = "Metric computation expects a .csv output path."
        LOG.error(mess)
        raise RuntimeError(mess)

    dirname = os.path.dirname(output)
    os.makedirs(dirname, exist_ok=True)

    for s_idx, scene in enumerate(scenes):
        sname = os.path.splitext(scene)[0]
        filepaths.append([])

        # Load data for the reference
        ref = pyexr.read(os.path.join(ref_folder, scene))[..., :3]
        if ref.sum() == 0:
            raise ValueError(
                "got an all zero image {}/{}".format(ref_folder, scene))
        if pad > 0:  # remove a side portion
            ref = ref[pad:-pad, pad:-pad, :]

        LOG.info("%s", sname)
        for m_idx, m in enumerate(methods):
            mname = os.path.split(m)[-1]
            mname, spp = _get_spp(mname)
            row = {"method": mname, "scene": sname, "spp": spp}

            LOG.info("  %s %d spp", mname, spp)

            # Load data for the current method
            path = os.path.abspath(os.path.join(m, scene))
            filepaths[s_idx].append(path)
            try:
                im = pyexr.read(path)[..., :3]
            except Exception as e:
                LOG.error(e)
                row["valid"] = False
                for k in METRIC_OPS:
                    row[k] = -1
                scores = scores.append(row, ignore_index=True)
                continue

            if pad > 0:  # remove a side portion
                im = im[pad:-pad, pad:-pad, :]
            if im.sum() == 0:
                LOG.warning("got an all zero image {}/{}, "
                            " invalidating the scene".format(m, scene))
                row["valid"] = False
                for k in METRIC_OPS:
                    row[k] = -1
                scores = scores.append(row, ignore_index=True)
                continue

            row["valid"] = True
            for k in METRIC_OPS:
                row[k] = METRIC_OPS[k](im, ref)
            scores = scores.append(row, ignore_index=True)
    scores.to_csv(output)


def _load_csvs(paths):
    """Load and merge one or more .csv files.

    Args:
        paths(list of str): paths to the .csv files.

    Returns:
        (pandas.DataFrame): merged tables
    """

    df = None
    for idx, path in enumerate(paths):
        _df = pd.read_csv(path, index_col=0)
        if idx == 0:
            df = _df
        else:
            df = df.append(_df, ignore_index=True)
    return df


def stats(csv_files, output):
    df = _load_csvs(csv_files)
    df = _prune_invalid_scenes(df)

    n_tot = df.size
    n_invalid = df[df["valid"] == False].size

    # Keep only the valid entries
    df = df[df["valid"] == True]

    spps = df["spp"].unique()
    methods = df["method"].unique()

    LOG.info("Computing statistics for %d methods with %d distinct spp values",
             len(methods), len(spps))

    mean_df = pd.DataFrame()
    std_df = pd.DataFrame()
    for spp in spps:
        current = df[df["spp"] == spp]
        for m in methods:
            mdata = current[current["method"] == m]
            LOG.info("Method %s, %d elements", m, mdata.size)
            row = dict(mdata.mean())
            row["method"] = m
            row.pop("valid")
            mean_df = mean_df.append(row, ignore_index=True)

            row = dict(mdata.std())
            row["method"] = m
            row.pop("valid")
            std_df = std_df.append(row, ignore_index=True)

    LOG.info("Averages:\n%s", mean_df)

    if n_invalid > 0:
        LOG.warning("%d invalid metrics (out of %d)", n_invalid, n_tot)

    # Save the means to disk
    mean_df.to_csv(output)

    return mean_df, std_df


def _get_spp(method_name):
    """Extracts the number of samples-per-pixel from a method's directory name.

    Args:
        method_name(str): name of the method.
    """

    method_name = method_name.strip()
    s = method_name.split("spp_")
    if len(s) == 2:
        spp = int(s[0])
        method_name = s[1]
    else:  # no "spp_" prefix, maybe its "spp"
        s = method_name.split("spp")
        if len(s) != 2 or s[1] != '':
            raise ValueError("unexpected spp format for '%s'" % method_name)
        spp = int(s[0])
        method_name = "input"
    return method_name, spp


def _mse(im, ref):
    """Mean-squared error between images.

    Args:
        im(np.array): image to test.
        ref(np.array): reference for the comparison.

    Returns:
        (float) error value.
    """
    return np.square(im-ref).mean()


def _rmse(im, ref, eps=1e-4):
    """Relative Mean-squared error between images.

    Args:
        im(np.array): image to test.
        ref(np.array): reference for the comparison.
        eps(float): small number to prevent division by 0.

    Returns:
        (float) error value.
    """
    diff = (np.square(im-ref) / (np.square(ref) + eps))
    diff = np.ravel(diff)
    diff = diff[~np.isnan(diff)]
    return diff.mean()


def _l1(im, ref):
    """Absolute error between images.

    Args:
        im(np.array): image to test.
        ref(np.array): reference for the comparison.

    Returns:
        (float) error value.
    """
    return np.abs(im-ref).mean()


def _rl1(im, ref, eps=1e-4):
    """Relative absolute error between images.

    Args:
        im(np.array): image to test.
        ref(np.array): reference for the comparison.
        eps(float): small number to prevent division by 0.

    Returns:
        (float) error value.
    """
    return (np.abs(im-ref) / (np.abs(ref) + eps)).mean()


def _ssim(im, ref):
    """Structural Similarity error (1-SSIM, or DSSIM).

    Args:
        im(np.array): image to test.
        ref(np.array): reference for the comparison.

    Returns:
        (float) error value.
    """
    return 1-ssim(im, ref, multichannel=True)


def _parse_list_or_txt(_input):
    """Parses a list of string from a .txt file, or list of strings.

    Args:
        _input(str or list of str): if a string, interpreted as a path to a
            list.txt file containing the actual filenames, straight list
            otherwise.
    Returns:
        (list of str): list of filenames.
    """
    if len(_input) == 1 and os.path.splitext(_input[0])[-1] == ".txt":
        print("loading from .txt")
        with open(_input[0]) as fid:
            ret = []
            for line in fid.readlines():
                ret.append(line.strip())
    else:
        ret = _input
    return ret


def _prune_invalid_scenes(scores):
    invalid = scores.loc[scores["valid"] == False]
    invalid = invalid["scene"].unique()
    if len(invalid) > 0:
        LOG.warning("%d invalid scenes %s", len(invalid), invalid)
    invalid_idx = scores.index[scores["scene"].isin(invalid)]
    scores = scores.drop(index=invalid_idx)
    return scores


METRIC_LABELS = {"mse": "MSE", "rmse": "rMSE", "ssim": "DSSIM", "l1": r"$L_1$",
                 "relative_l1": r"relative $L_1$"}


METRIC_OPS = {"mse": _mse, "rmse": _rmse, "ssim": _ssim, "l1": _l1,
              "relative_l1": _rl1}
