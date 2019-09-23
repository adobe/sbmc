"""Compute metrics from a set of .exr images."""
import argparse
import os
import numpy as np
import pyexr
import json
import skimage.io as skio
from skimage.measure import compare_ssim as ssim
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd

sb.set()
# sb.set_context(rc={'lines.markeredgewidth': 1})

try:
    import torch as th
    from ttools.modules.losses import PerceptualLoss
    PLOSS = PerceptualLoss(pretrained=True, n_in=3).cuda()
except:
    print("could not load torch")


plt.rcParams.update({'font.size': 8})


def _get_spp(mname):
    mname = mname.strip()
    s = mname.split("spp_")
    if len(s) == 2:
        spp = int(s[0])
        mname = s[1]
    else: # no "spp_" prefix, maybe its "spp"
        s = mname.split("spp")
        if len(s) != 2 or s[1] != '':
            raise ValueError("unexpected spp format for '%s'" % mname)
        spp = int(s[0])
        mname = "input"
    return mname, spp

def _tonemap(im):
    im = np.maximum(im, 0)  # make sure it's positive
    return np.power(np.clip(im / (1+im), 0, 1), 1/2.2)


def _perceptual(im, ref):
    im = _tonemap(im)
    ref = _tonemap(ref)
    im = th.from_numpy(im).permute(2, 0, 1).unsqueeze(0).cuda()
    ref = th.from_numpy(ref).permute(2, 0, 1).unsqueeze(0).cuda()
    loss = PLOSS(im, ref).item()
    return loss


def _mse(im, ref):
    return np.square(im-ref).mean()


def _rmse(im, ref, eps=1e-4):
    diff = (np.square(im-ref) / (np.square(ref) + eps))
    diff = np.ravel(diff)
    diff = diff[~np.isnan(diff)]
    return diff.mean()


def _l1(im, ref):
    return np.abs(im-ref).mean()


def _rl1(im, ref, eps=1e-4):
    return (np.abs(im-ref) / (np.abs(ref) + eps)).mean()


def _ssim(im, ref):
    return 1-ssim(im, ref, multichannel=True)


METRIC_LABELS = {
    "mse": "MSE",
    "rmse":"rMSE",
    "ssim":"DSSIM",
    "l1":r"$L_1$",
    "relative_l1":r"relative $L_1$",
    "perceptual":r"perceptual"
}


METRIC_OPS = {
    "mse": _mse,
    "rmse": _rmse,
    "ssim": _ssim,
    "l1": _l1,
    "relative_l1": _rl1,
    "perceptual": _perceptual,
}

def _parse_list_or_txt(arg):
    if len(arg) == 1 and os.path.splitext(arg[0])[-1] == ".txt":
        print("loading from .txt")
        with open(arg[0]) as fid:
            ret = []
            for l in fid.readlines():
                ret.append(l.strip())
    else:
        ret = arg
    return ret

def compute(args):
    # assert len(args.labels) == len(args.methods)
    pad = args.pad

    scores = pd.DataFrame(columns=["method", "scene", "spp", "valid"] + list(METRIC_LABELS.keys()))

    scenes = _parse_list_or_txt(args.scenes)
    methods = _parse_list_or_txt(args.methods)

    n_scenes = len(scenes)
    n_methods = len(methods)

    print("evaluating %d scenes and %d methods\n" % (n_scenes, n_methods))

    filepaths = []

    assert os.path.splitext(args.output)[-1] == ".csv", "expects .csv output path"
    dirname = os.path.dirname(args.output)
    os.makedirs(dirname, exist_ok=True)

    for s_idx, s in enumerate(scenes):
        sname = os.path.splitext(s)[0]
        filepaths.append([])
        # Load data for the reference
        ref = pyexr.read(os.path.join(args.ref, s))[..., :3]
        if ref.sum() == 0:
            raise ValueError("got an all zero image {}/{}".format(args.ref, s))
        if pad > 0: # remove a side portion
            ref = ref[pad:-pad, pad:-pad, :]

        print(". %s" % sname)
        for m_idx, m in enumerate(methods):
            mname = os.path.split(m)[-1]
            mname, spp = _get_spp(mname)
            row = { "method": mname, "scene": sname, "spp": spp }

            print("  - %s %d spp" % (mname, spp))

            # Load data for the current method
            path = os.path.abspath(os.path.join(m, s))
            filepaths[s_idx].append(path)
            try:
                im = pyexr.read(path)[..., :3]
            except Exception as e:
                print(e)
                row["valid"] = False
                for k in METRIC_OPS:
                    row[k] = -1
                scores = scores.append(row, ignore_index=True)
                continue 

            if pad > 0: # remove a side portion
                im = im[pad:-pad, pad:-pad, :]
            if im.sum() == 0:
                print("got an all zero image {}/{}, invalidating the scene".format(m, s))
                row["valid"] = False
                for k in METRIC_OPS:
                    row[k] = -1
                scores = scores.append(row, ignore_index=True)
                continue

            row["valid"] = True
            for k in METRIC_OPS:
                row[k] = METRIC_OPS[k](im, ref)
            scores = scores.append(row, ignore_index=True)

            # print("\t", m, "\trmse = {:.5f}".format(scores["rmse"][m_idx, s_idx],))

    print(scores)
    scores.to_csv(args.output)


def _load_csvs(paths):
    df = None
    for idx, path in enumerate(paths):
        _df = pd.read_csv(path, index_col=0)
        if idx == 0:
            df = _df
        else:
            df = df.append(_df, ignore_index=True)
    return df


def stats(args):
    df = _load_csvs(args.inputs)

    df = _prune_invalid_scenes(df)

    n_tot = df.size
    n_invalid = df[df["valid"] == False].size

    # Keep only the valid entries
    df = df[df["valid"]==True]

    spps = df["spp"].unique()
    methods = df["method"].unique()

    # Intersect available with query
    if args.spp is not None:
        spps = args.spp
    if args.methods is not None:
        methods = _parse_list_or_txt(args.methods)

    print("%d methods with %d spp values\n" % (len(methods), len(spps)))

    mean_df = pd.DataFrame()
    std_df = pd.DataFrame()
    for spp in spps:
        current = df[df["spp"]==spp]
        for m in methods:
            mdata = current[current["method"]==m] 
            print(mdata.size, m)
            row = dict(mdata.mean())
            row["method"] = m
            row.pop("valid")
            mean_df = mean_df.append(row, ignore_index=True)

            row = dict(mdata.std())
            row["method"] = m
            row.pop("valid")
            std_df = std_df.append(row, ignore_index=True)

    print("Mean:")
    print(mean_df)

    # print(df[df["spp"]==4])

    # print("Std:")
    # print(std_df)

    print("%d invalid metrics (out of %d)" % (n_invalid, n_tot))

    return mean_df, std_df


def latex(args):
    mean, std = stats(args)

    spps = mean["spp"].unique()
    methods = mean["method"].unique()

    # Intersect available with query
    if args.spp is not None:
        spps = args.spp
    if args.methods is not None:
        methods = req_methods = _parse_list_or_txt(args.methods)
        # methods = list(set(methods) & set(req_methods))

    mean = mean[mean["method"].isin(methods)]
    mean = mean[mean["spp"].isin(spps)]

    if args.labels is not None:
        assert len(args.labels) == len(args.methods), "methods and labels should have the same length"
        labels = args.labels
    else:
        labels = methods
    for idx, m in enumerate(methods):
        mean.loc[mean["method"]==m, "label"] = labels[idx]

    print("\nMaking latex table\n")

    s = "\\toprule\n"
    line = " & "
    for l in labels:
        line += " & \\text{%s}" % l
    line += r" \\" + "\n"
    s += line
    for spp in spps:
        s += "\\midrule\n"
        current_spp = mean.loc[mean["spp"]==spp]
        for k in args.metrics:
            if k == "rmse":
                line = "\\text{%dspp}"% spp
            else:
                line = " "
            line += (" & %s" % METRIC_LABELS[k])
            best_idx = current_spp[k].idxmin()
            # best = current_spp.index[best_idx]
            for m in methods:
                datapoint = current_spp.index[current_spp["method"]==m].tolist()
                if len(datapoint) != 1:
                    print("method %s has no valid data" % m)
                    line += " & \\textemdash" 
                    continue
                datapoint = datapoint[0]

                dataval = current_spp.loc[current_spp["method"]==m][k].iloc[0]

                line += " & " 
                if datapoint == best_idx:
                    line += "\\B "
                line += "%0.4f"% dataval
            line += r" \\" + "\n"
            s += line
    s += "\\bottomrule\n"

    print("Writing to", args.output)
    dirname = os.path.dirname(args.output)
    os.makedirs(dirname, exist_ok=True)
    with open(args.output, 'w') as fid:
        fid.write(s)
    # print(s)


def scenelatex(args):
    df = _load_csvs(args.inputs)
    df = _prune_invalid_scenes(df)
    n_tot = df.size
    n_invalid = df[df["valid"] == False].size
    print("%d invalid metrics (out of %d)" % (n_invalid, n_tot))

    # Keep only the valid entries
    df = df[df["valid"]==True]

    spps = df["spp"].unique()
    methods = df["method"].unique()

    # Intersect available with query
    if args.spp is not None:
        spps = args.spp
    if args.methods is not None:
        methods = req_methods = _parse_list_or_txt(args.methods)

    df = df[df["method"].isin(methods)]
    df = df[df["spp"].isin(spps)]

    if args.labels is not None:
        assert len(args.labels) == len(args.methods), "methods and labels should have the same length"
        labels = args.labels
    else:
        labels = methods
    for idx, m in enumerate(methods):
        df.loc[df["method"]==m, "label"] = labels[idx]

    scenes = df["scene"].unique()
    print("\nMaking latex table\n")

    n_per_table = 8
    n_tables = int(np.ceil(len(scenes) / n_per_table))

    n_methods = len(methods)

    # metric = "rmse"
    s = ""
    for metric in METRIC_LABELS:
        scene_idx = 0
        for tid in range(n_tables):
            table = "\\begin{table*}[!t]\n"
            table += "\\centering\n"
            table += "\\caption{Details of metric \"%s\" per scene (%d of %d)}\n" % (METRIC_LABELS[metric], tid+1, n_tables)
            table += "\\begin{tabular}{ll*{%d}{S[table-format=.4]}}\n" % n_methods
            table += "\\toprule\n"

            header = " scene & spp "
            for m in labels:
                header += " & \\text{%s}" % m
            header += "\\\\\n"

            table += header
            table += "\\midrule\n"

            for count in range(n_per_table):
                if scene_idx >= len(scenes):
                    break
                scene = scenes[scene_idx]
                scene_idx += 1
                for idx_s, spp in enumerate(spps):
                    if idx_s == 0:
                        line = "\\text{%s}" % scene.replace("_", "")
                    else:
                        line = " "
                    line += " & %d" % spp

                    datarow = df.loc[(df["spp"]==spp) & (df["scene"]==scene)]
                    best_idx = datarow[metric].idxmin()

                    for m in methods:
                        datapoint = datarow.index[datarow["method"]==m].tolist()
                        if len(datapoint) != 1:
                            print("method %s has no valid data" % m)
                            line += " & \\textemdash" 
                            continue
                        datapoint = datapoint[0]
                        dataval = datarow.loc[datapoint][metric]

                        line += " & " 
                        if datapoint == best_idx:
                            line += "\\B "
                        line += "%0.4f"% dataval
                    line += "\\\\\n"
                    table += line
                table += "\\midrule\n"

            table += "\\bottomrule\n"
            table += "\\end{tabular}\n"
            table += "\\end{table*}\n\n"

            s += table

    print("Writing to", args.output, "with", n_tables, "tables")
    dirname = os.path.dirname(args.output)
    os.makedirs(dirname, exist_ok=True)
    with open(args.output, 'w') as fid:
        fid.write(s)



def _prune_invalid_scenes(scores):
    invalid = scores.loc[scores["valid"]==False]
    # print(invalid)
    invalid = invalid["scene"].unique()
    print("%d invalid scenes %s" % (len(invalid), invalid))
    invalid_idx = scores.index[scores["scene"].isin(invalid)]
    scores = scores.drop(index=invalid_idx)
    return scores

def distrib(args):
    scores = _load_csvs(args.inputs)

    scores = _prune_invalid_scenes(scores)

    spps = scores["spp"].unique()
    scenes = scores["scene"].unique()
    methods = scores["method"].unique()

    # Intersect available with query
    if args.spp is not None:
        spps = args.spp

    if args.methods is not None:
        methods = req_methods = _parse_list_or_txt(args.methods)
        # methods = list(set(methods) & set(req_methods))

    scores = scores[scores["method"].isin(methods)]
    scores = scores[scores["spp"].isin(spps)]

    if args.labels is not None:
        assert len(args.labels) == len(args.methods), "methods and labels should have the same length"
        labels = args.labels
    else:
        labels = methods
    for idx, m in enumerate(args.methods):
        scores.loc[scores["method"]==m, "label"] = labels[idx]

    if args.normalize_by is not None:
        assert args.normalize_by in methods, "normalizer '%s' should be in the list of methods %s" % (args.normalize_by, methods)

    print("%d methods, %d scenes" % (len(methods), len(scenes)))

    root = args.output
    os.makedirs(root, exist_ok=True)

    n_scenes = len(scenes)

    scenes_per_fig = 7
    for spp in spps:
        scores_ = scores.loc[scores["spp"]==spp]
        print("%d spp" % spp)
        for m in METRIC_LABELS:
            if args.normalize_by is not None:
                ref = scores_.loc[scores_["method"]==args.normalize_by]
                refscores = ref[m]
                for s in scenes:
                    val = ref.loc[ref["scene"]==s][m]
                    assert val.size == 1, "expected size 1 dataframe for normalizer got %d" % val.size
                    val = val.iloc[0]
                    scene_idx = scores.index[(scores["scene"]==s) & (scores["spp"] == spp)]
                    scores.loc[scene_idx, m] /= val

            nrows = int(np.ceil(n_scenes / scenes_per_fig))
            fig = plt.figure(figsize=(3*scenes_per_fig, 3*nrows))
            pidx = 0
            for idx, i in enumerate(range(0, n_scenes, scenes_per_fig)):
                plt.subplot(nrows, 1, idx+1)
                upper = min(i + scenes_per_fig, n_scenes)
                subset = scores.loc[scores["scene"].isin(scenes[i:upper])]

                plot = sb.barplot(x="scene", y=m, hue="label", data=subset.loc[subset["spp"]==spp])
                plot.axhline(1, ls='--')
                for p in plot.patches:
                    if p.get_height() == 1:  # annotate only the ref
                        print(pidx, scenes[pidx], refscores.iloc[pidx])
                        plot.annotate("%.5f" % refscores.iloc[pidx], (p.get_x() + p.get_width() / 2., p.get_y() + 0.5*p.get_height()),
                                    ha='center', va='center', rotation=0, xytext=(0, 0), textcoords='offset points')  #vertical bars
                        pidx += 1
                if args.log:
                    plot.get_figure().get_axes()[0].set_yscale('log')

                if args.normalize_by is not None:
                    plot.set_ylim([0, 10])
                    plot.set_ylabel("relative %s" % m)
            out = os.path.join(root, "%dspp" % spp,  "%s.pdf" % m)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            plt.suptitle("%d spp | %s distribution, normalized with respect to %s" % (spp, METRIC_LABELS[m], labels[methods.index(args.normalize_by)]))
            plt.subplots_adjust(top=0.99, right=0.99)
            plt.savefig(out)
            plt.close(fig)


def timeplot(args):
    args.spp = None
    args.scenes = None
    args.methods = None
    mean, std = stats(args)


    others = {
        "disney_samples_ft": 14.6,
        "2016_bitterli_nfor": 21.9,
        "2015_kalantari_lbf": 10.4,
        "2012_rousselle_nlm": 13.3,
        "rmse_pixelgather": 5.8,
        "rmse_pixelscatter": 5.8
    }
    methods = list(others.keys()) + ["rmse_multisteps3", "rmse_gather", "disney_samples_ft", "2011_sen_rpf", 
                                     ]
    mean = mean.loc[mean["method"].isin(methods)]

    spps = [4, 8, 16, 32, 64, 128]
    rendering = [15.4 , 30.6 , 61.3 , 121.4 , 245.1, 491.8]
    ours = [6.0   , 10.1  , 18.9 , 35.9 , 67.0 , 156.5]
    sen = [281.2 , 638.1 , 1603.1 , 4847.8, None, None]
    for idx, spp in enumerate(spps):
        mean.loc[mean["spp"]==spp, "time"] = rendering[idx]
        mean.loc[(mean["method"]=="rmse_multisteps3") & (mean["spp"]==spp), "time"] += ours[idx]
        mean.loc[(mean["method"]=="rmse_gather") & (mean["spp"]==spp), "time"] += ours[idx]
        # mean.loc[(mean["method"]=="rmse_pixelgather") & (mean["spp"]==spp), "time"] += ours[idx]
        # mean.loc[(mean["method"]=="rmse_pixelscatter") & (mean["spp"]==spp), "time"] += ours[idx]
        mean.loc[(mean["method"]=="2011_sen_rpf") & (mean["spp"]==spp), "time"] += sen[idx]
        for o in others:
            mean.loc[(mean["method"]==o) & (mean["spp"]==spp), "time"] += others[o]
        # mean.loc[, "time"] = 10.6
    print(mean)
    # import ipdb; ipdb.set_trace()


    #
    fig = plt.figure()
    ax = sb.lineplot(x="time", y="rmse", hue="method", data=mean, markers=True)
    # ax = sb.lineplot(x="time", y="rmse", hue="method", data=mean, markers="*")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel("time (s)")
    ax.set_xticks([10*(2**k) for k in range(10)])
    ax.set_yticks([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56])

    def _formatter(x, pos, int_fmt=False):
        """The two args are the value and tick position.
        Label ticks with the product of the exponentiation"""
        if int_fmt:
            return '%1i' % (x)
        else:
            return '%.2f' % (x)

    formatter_i = plt.FuncFormatter(lambda x, pos: _formatter(x, pos, True))
    formatter = plt.FuncFormatter(_formatter)
    ax.xaxis.set_major_formatter(formatter_i)
    ax.yaxis.set_major_formatter(formatter)

    plt.title("rMSE vs. running time $(render+denoise)$")


    dirname = os.path.dirname(args.output)

    # os.makedirs(dirname, exist_ok=True)
    print("saving to %s" % args.output)
    plt.savefig(args.output)
    plt.close(fig)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parsers = parser.add_subparsers(dest="command")

    compute_parser = parsers.add_parser("compute")
    compute_parser.add_argument("ref", help="path to the root of the reference .exr files")
    compute_parser.add_argument("output", help="path to store the stats")
    compute_parser.add_argument("--methods", nargs="+", help="list of methods to compare, their folders are expected to sit next to 'ref/'")
    compute_parser.add_argument("--scenes", nargs="+", help="list of scenes to evaluate")
    compute_parser.add_argument("--pad", type=int, default=21, help="how many pixels to remove on each side")
    # compute_parser.add_argument("--labels", nargs="+", help="how to names the methods for display")

    stats_parser = parsers.add_parser("stats")
    stats_parser.add_argument("inputs", nargs="+", help=".csv files to include")
    stats_parser.add_argument("--methods", nargs="+", help="list of methods to display")
    stats_parser.add_argument("--spp", nargs="+", type=int, help="list of spp to display")

    latex_parser = parsers.add_parser("latex")
    latex_parser.add_argument("inputs", nargs="+", help=".csv files to include")
    latex_parser.add_argument("output", help=".tex outputfile")
    latex_parser.add_argument("--methods", nargs="+", help="list of methods to display")
    latex_parser.add_argument("--spp", nargs="+", type=int, help="list of spp to display")
    latex_parser.add_argument("--labels", nargs="+", help="list of method labels")
    latex_parser.add_argument("--metrics", nargs="+", default=["rmse", "ssim"], help="")

    time_parser = parsers.add_parser("timeplot")
    time_parser.add_argument("inputs", nargs="+", help=".csv files to include")
    time_parser.add_argument("output", help=".tex outputfile")

    scenelatex_parser = parsers.add_parser("scenelatex")
    scenelatex_parser.add_argument("inputs", nargs="+", help=".csv files to include")
    scenelatex_parser.add_argument("output", help=".tex outputfile")
    scenelatex_parser.add_argument("--methods", nargs="+", help="list of methods to display")
    scenelatex_parser.add_argument("--spp", nargs="+", type=int, help="list of spp to display")
    scenelatex_parser.add_argument("--labels", nargs="+", help="list of method labels")

    distrib_parser = parsers.add_parser("distrib")
    distrib_parser.add_argument("inputs", nargs="+", help=".csv files to include")
    distrib_parser.add_argument("output", help="path to store the figures")
    distrib_parser.add_argument("--normalize_by", help="ref method to normalize to")
    distrib_parser.add_argument("--log", dest="log", action="store_true", help="ref method to normalize to")
    distrib_parser.add_argument("--methods", nargs="+", help="list of methods to display")
    distrib_parser.add_argument("--labels", nargs="+", help="list of method labels")
    distrib_parser.add_argument("--spp", nargs="+", type=int, help="list of spp to display")
    distrib_parser.set_defaults(log=False)

    args = parser.parse_args()
    if args.command == "compute":
        compute(args)
    elif args.command == "distrib":
        distrib(args)
    elif args.command == "stats":
        stats(args)
    elif args.command == "latex":
        latex(args)
    elif args.command == "scenelatex":
        scenelatex(args)
    elif args.command == "timeplot":
        timeplot(args)
