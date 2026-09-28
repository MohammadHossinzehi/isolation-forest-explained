"""Isolation Forest ensemble: subsampling, scoring, thresholding, explanation."""

from __future__ import annotations

import math

import numpy as np

from .tree import IsolationTree, average_path_length


class IsolationForest:
    """Unsupervised anomaly detector built from random isolation trees.

    The intuition: anomalies are few and different, so random partitioning
    separates them from the rest in very few cuts. Averaging the depth at
    which a point gets isolated over many random trees gives a stable score.

    Parameters
    ----------
    n_estimators:
        Number of trees. Scores converge quickly; 100 is the paper default.
    max_samples:
        Subsample size ``psi`` per tree. Small subsamples are a feature, not a
        compromise: they reduce swamping (normals near anomalies looking odd)
        and masking (dense anomaly clusters hiding each other).
    contamination:
        Expected outlier fraction, used only to pick the ``predict`` threshold
        from training scores. ``None`` uses the paper's fixed cut of 0.5.
    extended:
        ``False`` for axis-parallel splits, ``True`` for fully extended
        hyperplane splits, or an int to set the extension level explicitly.
    random_state:
        Seed or ``numpy.random.Generator`` for reproducibility.
    """

    def __init__(
        self,
        n_estimators=100,
        max_samples=256,
        contamination=None,
        extended=False,
        random_state=None,
    ):
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if contamination is not None and not 0.0 < contamination < 0.5:
            raise ValueError("contamination must be in (0, 0.5)")
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.extended = extended
        self.random_state = random_state

    # ------------------------------------------------------------------ fit
    def fit(self, X):
        X = self._check(X, fitting=True)
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)

        psi = n if self.max_samples in (None, "all") else min(int(self.max_samples), n)
        if psi < 2:
            raise ValueError("need at least 2 samples to fit")
        self.psi_ = psi
        max_depth = math.ceil(math.log2(psi))

        if self.extended is False:
            level = None
        elif self.extended is True:
            level = d - 1
        else:
            level = int(self.extended)
            if not 0 <= level <= d - 1:
                raise ValueError(f"extension level must be in [0, {d - 1}]")
        self.extension_level_ = level

        self.trees_ = []
        for _ in range(self.n_estimators):
            sample = rng.choice(n, size=psi, replace=False)
            tree = IsolationTree(max_depth, extension_level=level, rng=rng)
            self.trees_.append(tree.fit(X[sample]))

        self._c_psi = average_path_length(psi)
        train_scores = self.score_samples(X)
        if self.contamination is None:
            self.threshold_ = 0.5
        else:
            self.threshold_ = float(np.quantile(train_scores, 1.0 - self.contamination))
        return self

    # ---------------------------------------------------------------- score
    def expected_path_length(self, X):
        X = self._check(X)
        total = np.zeros(X.shape[0])
        for tree in self.trees_:
            total += tree.path_length(X)
        return total / len(self.trees_)

    def score_samples(self, X):
        """Anomaly score ``s = 2 ** (-E[h(x)] / c(psi))`` in (0, 1].

        Close to 1: isolated almost immediately, very likely an anomaly.
        Around 0.5 or below everywhere: no distinct anomalies in the data.
        """
        return np.power(2.0, -self.expected_path_length(X) / self._c_psi)

    def decision_function(self, X):
        """Signed distance to the threshold; positive means anomalous."""
        return self.score_samples(X) - self.threshold_

    def predict(self, X):
        """1 for anomalies, 0 for inliers."""
        return (self.decision_function(X) > 0).astype(int)

    def fit_predict(self, X):
        return self.fit(X).predict(X)

    # -------------------------------------------------------------- explain
    def explain(self, X):
        """Per feature attribution of each sample's isolation, rows sum to 1.

        See ``IsolationTree.decision_path_weights`` for the rule. The raw
        credit is compared to the average credit on a reference sample so a
        feature only stands out when it isolates *this* point more than it
        isolates typical points.
        """
        X = self._check(X)
        raw = np.zeros((X.shape[0], self.n_features_))
        for tree in self.trees_:
            raw += tree.decision_path_weights(X)
        raw /= len(self.trees_)
        rel = raw / self._baseline_credit()
        return rel / rel.sum(axis=1, keepdims=True)

    def _baseline_credit(self):
        if getattr(self, "_baseline", None) is None:
            ref = self._reference_
            credit = np.zeros(self.n_features_)
            for tree in self.trees_:
                credit += tree.decision_path_weights(ref).mean(axis=0)
            credit /= len(self.trees_)
            credit[credit <= 0] = credit[credit > 0].min() if (credit > 0).any() else 1.0
            self._baseline = credit
        return self._baseline

    # -------------------------------------------------------------- helpers
    def _check(self, X, fitting=False):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values")
        if fitting:
            self.n_features_ = X.shape[1]
            # Keep a small reference sample for explanation baselines.
            rng = np.random.default_rng(0)
            take = min(512, X.shape[0])
            self._reference_ = X[rng.choice(X.shape[0], size=take, replace=False)]
            self._baseline = None
        else:
            if not hasattr(self, "trees_"):
                raise RuntimeError("call fit() first")
            if X.shape[1] != self.n_features_:
                raise ValueError(f"expected {self.n_features_} features, got {X.shape[1]}")
        return X

    def summary(self):
        nodes = [t.n_nodes for t in self.trees_]
        kind = "axis-parallel" if self.extension_level_ is None else f"extended (level {self.extension_level_})"
        return (
            f"IsolationForest: {len(self.trees_)} trees, psi={self.psi_}, "
            f"{kind}, mean nodes/tree={np.mean(nodes):.1f}, threshold={self.threshold_:.4f}"
        )
