"""Synthetic benchmark datasets with known ground truth labels.

Each generator returns ``(X, y)`` where ``y == 1`` marks an injected anomaly.
They are chosen to expose different strengths and weaknesses:

* ``blobs``: easy global outliers scattered around Gaussian clusters.
* ``two_blobs_diagonal``: two clusters on a diagonal. Axis-parallel forests
  paint "ghost" normal-looking bands at (x of cluster A, y of cluster B);
  small anomaly clumps are planted near those corners. Run
  ``python -m isoforest map two_blobs_diagonal`` to see the bands.
* ``ring``: a noisy circle with anomalies in the empty centre. Every anomaly
  has perfectly ordinary marginals, so axis-parallel cuts struggle to isolate
  them; this is where the extended forest earns its keep.
* ``local_feature``: high dimensional inliers where each anomaly deviates on a
  single feature only, used to check that explanations point at that feature.
"""

from __future__ import annotations

import numpy as np


def blobs(n=1000, contamination=0.05, d=2, seed=0):
    rng = np.random.default_rng(seed)
    n_out = int(round(n * contamination))
    n_in = n - n_out
    centers = rng.uniform(-5, 5, size=(3, d))
    which = rng.integers(0, 3, size=n_in)
    inliers = centers[which] + rng.standard_normal((n_in, d)) * 0.6
    outliers = rng.uniform(-9, 9, size=(n_out, d))
    return _pack(inliers, outliers, rng)


def two_blobs_diagonal(n=1000, contamination=0.03, seed=0):
    rng = np.random.default_rng(seed)
    n_out = int(round(n * contamination))
    n_in = n - n_out
    half = n_in // 2
    a = rng.standard_normal((half, 2)) * 0.5 + [-3.0, -3.0]
    b = rng.standard_normal((n_in - half, 2)) * 0.5 + [3.0, 3.0]
    ghosts = np.array([[-3.0, 3.0], [3.0, -3.0]])
    outliers = ghosts[rng.integers(0, 2, size=n_out)] + rng.standard_normal((n_out, 2)) * 0.4
    return _pack(np.vstack([a, b]), outliers, rng)


def ring(n=1000, contamination=0.03, radius=4.0, seed=0):
    rng = np.random.default_rng(seed)
    n_out = int(round(n * contamination))
    n_in = n - n_out
    theta = rng.uniform(0, 2 * np.pi, n_in)
    r = radius + rng.standard_normal(n_in) * 0.25
    inliers = np.c_[r * np.cos(theta), r * np.sin(theta)]
    outliers = rng.standard_normal((n_out, 2)) * 0.6
    return _pack(inliers, outliers, rng)


def local_feature(n=1000, contamination=0.02, d=8, seed=0):
    """Returns ``(X, y, feature)`` where ``feature[i]`` is the perturbed
    column for anomaly ``i`` and ``-1`` for inliers."""
    rng = np.random.default_rng(seed)
    n_out = int(round(n * contamination))
    n_in = n - n_out
    inliers = rng.standard_normal((n_in, d))
    outliers = rng.standard_normal((n_out, d))
    cols = rng.integers(0, d, size=n_out)
    outliers[np.arange(n_out), cols] = rng.choice([-1, 1], size=n_out) * rng.uniform(6, 9, n_out)
    X = np.vstack([inliers, outliers])
    y = np.r_[np.zeros(n_in, int), np.ones(n_out, int)]
    feat = np.r_[np.full(n_in, -1), cols]
    perm = rng.permutation(n)
    return X[perm], y[perm], feat[perm]


def _pack(inliers, outliers, rng):
    X = np.vstack([inliers, outliers])
    y = np.r_[np.zeros(len(inliers), int), np.ones(len(outliers), int)]
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


REGISTRY = {
    "blobs": blobs,
    "two_blobs_diagonal": two_blobs_diagonal,
    "ring": ring,
}
