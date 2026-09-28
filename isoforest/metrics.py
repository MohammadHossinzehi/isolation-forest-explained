"""Ranking metrics for anomaly scores, implemented without scikit-learn."""

from __future__ import annotations

import numpy as np


def _rank_average_ties(values):
    """1 based ranks, tied values share the mean of their positions."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(values.size)
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def roc_auc(y_true, scores):
    """Area under the ROC curve via the Mann Whitney U statistic.

    Equals the probability that a random anomaly scores higher than a random
    inlier, with ties counted as one half.
    """
    y = np.asarray(y_true).astype(bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc needs both classes present")
    ranks = _rank_average_ties(scores)
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def average_precision(y_true, scores):
    """Area under the precision recall curve (step interpolation).

    More informative than ROC AUC when anomalies are rare, because it ignores
    the huge pool of easy true negatives.
    """
    y = np.asarray(y_true).astype(bool)
    if y.sum() == 0:
        raise ValueError("average_precision needs at least one positive")
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    s, t = scores[order], y[order]
    # Evaluate only at the last index of each distinct score so ties are one step.
    distinct = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]
    tp = np.cumsum(t)[distinct]
    precision = tp / (distinct + 1)
    recall = tp / y.sum()
    prev_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - prev_recall) * precision))


def precision_at_k(y_true, scores, k):
    y = np.asarray(y_true).astype(bool)
    top = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")[:k]
    return float(y[top].mean())
