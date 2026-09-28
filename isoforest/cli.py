"""Command line interface.

    python -m isoforest bench                     compare variants on synthetic data
    python -m isoforest map two_blobs_diagonal    ASCII score map of a 2D dataset
    python -m isoforest detect data.csv --top 10  rank rows of your own CSV
"""

from __future__ import annotations

import argparse
import csv
import sys
import time

import numpy as np

from . import datasets
from .forest import IsolationForest
from .metrics import average_precision, roc_auc

SHADES = " .:-=+*#%@"


def cmd_bench(args):
    rows = []
    for name, gen in datasets.REGISTRY.items():
        aucs = {False: [], True: []}
        aps = {False: [], True: []}
        for seed in range(args.seeds):
            X, y = gen(seed=seed)
            for ext in (False, True):
                model = IsolationForest(n_estimators=args.trees, extended=ext, random_state=seed).fit(X)
                s = model.score_samples(X)
                aucs[ext].append(roc_auc(y, s))
                aps[ext].append(average_precision(y, s))
        rows.append((name, np.mean(aucs[False]), np.mean(aucs[True]), np.mean(aps[False]), np.mean(aps[True])))

    print(f"{'dataset':<22}{'AUC std':>10}{'AUC ext':>10}{'AP std':>10}{'AP ext':>10}")
    print("-" * 62)
    for name, a0, a1, p0, p1 in rows:
        print(f"{name:<22}{a0:>10.3f}{a1:>10.3f}{p0:>10.3f}{p1:>10.3f}")
    print(f"\nmean over {args.seeds} seeds, {args.trees} trees, psi=256")


def score_grid(model, X, size):
    lo, hi = X.min(axis=0) - 1, X.max(axis=0) + 1
    xs = np.linspace(lo[0], hi[0], size * 2)
    ys = np.linspace(hi[1], lo[1], size)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.c_[gx.ravel(), gy.ravel()]
    return model.score_samples(grid).reshape(gy.shape)


def cmd_map(args):
    X, _ = datasets.REGISTRY[args.dataset](seed=args.seed)
    for ext in (False, True):
        model = IsolationForest(n_estimators=args.trees, extended=ext, random_state=args.seed).fit(X)
        S = score_grid(model, X, args.size)
        lo, hi = S.min(), S.max()
        idx = ((S - lo) / (hi - lo + 1e-12) * (len(SHADES) - 1)).round().astype(int)
        title = "extended" if ext else "standard (axis-parallel)"
        print(f"\n{title}: darker = more anomalous, score range [{lo:.3f}, {hi:.3f}]")
        for row in idx:
            print("".join(SHADES[i] for i in row))


def load_csv(path):
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        rows = [r for r in reader if r]
    header = None
    try:
        [float(v) for v in rows[0]]
    except ValueError:
        header, rows = rows[0], rows[1:]
    X = np.array([[float(v) for v in r] for r in rows])
    if header is None:
        header = [f"col{i}" for i in range(X.shape[1])]
    return header, X


def cmd_detect(args):
    header, X = load_csv(args.path)
    t0 = time.perf_counter()
    model = IsolationForest(
        n_estimators=args.trees,
        contamination=args.contamination,
        extended=args.extended,
        random_state=args.seed,
    ).fit(X)
    scores = model.score_samples(X)
    flags = model.predict(X)
    expl = model.explain(X)
    elapsed = time.perf_counter() - t0

    print(model.summary())
    print(f"{len(X)} rows, {flags.sum()} flagged, {elapsed:.2f}s\n")
    order = np.argsort(-scores)[: args.top]
    print(f"{'row':>6}  {'score':>7}  flag  top features")
    for i in order:
        top = np.argsort(-expl[i])[:3]
        why = ", ".join(f"{header[j]} {expl[i, j]:.0%}" for j in top)
        print(f"{i:>6}  {scores[i]:>7.4f}  {'  X ' if flags[i] else '    '}  {why}")


def build_parser():
    p = argparse.ArgumentParser(prog="isoforest", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="standard vs extended on synthetic datasets")
    b.add_argument("--seeds", type=int, default=5)
    b.add_argument("--trees", type=int, default=100)
    b.set_defaults(func=cmd_bench)

    m = sub.add_parser("map", help="ASCII anomaly score map of a 2D dataset")
    m.add_argument("dataset", choices=sorted(datasets.REGISTRY))
    m.add_argument("--size", type=int, default=24)
    m.add_argument("--trees", type=int, default=200)
    m.add_argument("--seed", type=int, default=0)
    m.set_defaults(func=cmd_map)

    d = sub.add_parser("detect", help="score rows of a numeric CSV")
    d.add_argument("path")
    d.add_argument("--top", type=int, default=10)
    d.add_argument("--trees", type=int, default=200)
    d.add_argument("--contamination", type=float, default=0.05)
    d.add_argument("--extended", action="store_true")
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(func=cmd_detect)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
