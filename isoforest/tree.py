"""Isolation trees stored as flat NumPy arrays.

A single class covers both variants from the literature:

* Standard Isolation Forest (Liu, Ting & Zhou, 2008): every split is an
  axis-parallel cut ``x[f] <= t`` with ``t`` drawn uniformly between the
  node's min and max on a randomly chosen feature ``f``.
* Extended Isolation Forest (Hariri, Kind & Brunner, 2018): every split is a
  random hyperplane ``(x - p) . n <= 0`` where ``n`` is a random direction and
  ``p`` is drawn uniformly inside the node's bounding box.

Both are expressed the same way internally: a node stores a normal vector and
an offset, and a sample goes left when ``x . normal <= offset``. An axis
parallel split is just a one-hot normal. That keeps the scoring loop
identical for both variants and fully vectorised.
"""

from __future__ import annotations

import numpy as np

EULER_GAMMA = 0.5772156649015329


def average_path_length(n):
    """Expected path length of an unsuccessful BST search over ``n`` points.

    This is the ``c(n)`` normaliser from the original paper. It is used twice:
    to normalise the final score and to credit leaves that still hold more
    than one sample when the depth limit stopped growth early.
    """
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n)
    big = n > 2
    out[n == 2] = 1.0
    nb = n[big]
    out[big] = 2.0 * (np.log(nb - 1.0) + EULER_GAMMA) - 2.0 * (nb - 1.0) / nb
    return out if out.ndim else float(out)


class IsolationTree:
    """One randomly grown isolation tree.

    Parameters
    ----------
    max_depth:
        Growth stops at this depth. The paper uses ``ceil(log2(psi))`` which
        is roughly the average depth of a random BST on ``psi`` points: deeper
        nodes only separate normal points from each other, which is wasted work.
    extension_level:
        ``None`` for classic axis-parallel splits. An integer ``k`` in
        ``[0, d-1]`` for the extended variant, where ``k + 1`` coordinates of
        each random normal are non zero. ``k = d - 1`` is the fully extended
        forest; ``k = 0`` is axis-parallel again but with the intercept drawn
        from the bounding box instead of per feature.
    """

    MAX_SPLIT_TRIES = 32

    def __init__(self, max_depth, extension_level=None, rng=None):
        self.max_depth = int(max_depth)
        self.extension_level = extension_level
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------ fit
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        self.n_features_ = d

        normals, offsets, left, right, size, depth = [], [], [], [], [], []

        def new_node(sz, dp):
            normals.append(np.zeros(d))
            offsets.append(0.0)
            left.append(-1)
            right.append(-1)
            size.append(sz)
            depth.append(dp)
            return len(size) - 1

        root = new_node(n, 0)
        # Explicit stack instead of recursion: deep trees on large psi would
        # otherwise flirt with Python's recursion limit.
        stack = [(root, np.arange(n))]
        while stack:
            node, idx = stack.pop()
            if depth[node] >= self.max_depth or idx.size <= 1:
                continue
            # A random hyperplane through a random point of the bounding box
            # can miss every sample (think of a thin diagonal cluster). The
            # reference EIF implementation keeps such empty children; we
            # redraw instead, which keeps trees compact and every split
            # useful. Axis-parallel splits essentially never need a retry.
            for _ in range(self.MAX_SPLIT_TRIES):
                split = self._draw_split(X[idx])
                if split is None:  # all points identical: cannot isolate further
                    break
                normal, offset = split
                go_left = X[idx] @ normal <= offset
                li, ri = idx[go_left], idx[~go_left]
                if li.size and ri.size:
                    break
            else:
                split = None
            if split is None:
                continue
            normals[node] = normal
            offsets[node] = offset
            l_id = new_node(li.size, depth[node] + 1)
            r_id = new_node(ri.size, depth[node] + 1)
            left[node], right[node] = l_id, r_id
            stack.append((l_id, li))
            stack.append((r_id, ri))

        self.normals_ = np.vstack(normals)
        self.offsets_ = np.asarray(offsets)
        self.left_ = np.asarray(left, dtype=np.int64)
        self.right_ = np.asarray(right, dtype=np.int64)
        self.size_ = np.asarray(size, dtype=np.int64)
        self.depth_ = np.asarray(depth, dtype=np.int64)
        self.is_leaf_ = self.left_ < 0
        # Path length credited when a sample lands in a leaf: its depth plus the
        # expected extra depth had the leaf been grown out fully.
        self.leaf_value_ = self.depth_ + average_path_length(self.size_)
        return self

    def _draw_split(self, Xn):
        lo, hi = Xn.min(axis=0), Xn.max(axis=0)
        spread = hi - lo
        usable = np.flatnonzero(spread > 0)
        if usable.size == 0:
            return None
        d = Xn.shape[1]
        normal = np.zeros(d)

        if self.extension_level is None:
            f = self.rng.choice(usable)
            normal[f] = 1.0
            return normal, self.rng.uniform(lo[f], hi[f])

        # Extended split: random direction restricted to k + 1 coordinates,
        # intercept point uniform in the node's bounding box.
        k = min(int(self.extension_level), d - 1)
        dims = self.rng.choice(d, size=k + 1, replace=False)
        normal[dims] = self.rng.standard_normal(k + 1)
        p = self.rng.uniform(lo, hi)
        return normal, float(p @ normal)

    # ---------------------------------------------------------------- score
    def apply(self, X):
        """Return the leaf index each row of ``X`` ends up in."""
        X = np.asarray(X, dtype=float)
        node = np.zeros(X.shape[0], dtype=np.int64)
        active = ~self.is_leaf_[node]
        while active.any():
            rows = np.flatnonzero(active)
            cur = node[rows]
            proj = np.einsum("ij,ij->i", X[rows], self.normals_[cur])
            node[rows] = np.where(proj <= self.offsets_[cur], self.left_[cur], self.right_[cur])
            active[rows] = ~self.is_leaf_[node[rows]]
        return node

    def path_length(self, X):
        return self.leaf_value_[self.apply(X)]

    def decision_path_weights(self, X):
        """Per feature credit for how quickly each sample was isolated.

        Walks every sample down the tree. At each split it passes, the split's
        features share ``log(parent_size / child_size)``: the number of nats
        of "isolation" that split bought for this particular sample. A split
        that throws an anomaly into a nearly empty child earns a lot; a split
        that halves a dense cluster earns ``log 2``. The share is divided
        among features in proportion to ``|normal|``, so the same rule works
        for axis-parallel and hyperplane splits. It is a lightweight cousin of
        the DIFFI importance scheme.
        """
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        out = np.zeros((n, self.n_features_))
        abs_n = np.abs(self.normals_)
        norm = abs_n.sum(axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        share = abs_n / norm
        node = np.zeros(n, dtype=np.int64)
        active = ~self.is_leaf_[node]
        while active.any():
            rows = np.flatnonzero(active)
            cur = node[rows]
            proj = np.einsum("ij,ij->i", X[rows], self.normals_[cur])
            nxt = np.where(proj <= self.offsets_[cur], self.left_[cur], self.right_[cur])
            gain = np.log(self.size_[cur] / self.size_[nxt])
            out[rows] += share[cur] * gain[:, None]
            node[rows] = nxt
            active[rows] = ~self.is_leaf_[nxt]
        return out

    @property
    def n_nodes(self):
        return int(self.size_.size)
