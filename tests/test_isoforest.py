import math

import numpy as np
import pytest

from isoforest import IsolationForest, IsolationTree, average_path_length, average_precision, roc_auc
from isoforest import datasets
from isoforest.cli import main


# --------------------------------------------------------------- c(n)
def test_average_path_length_known_values():
    assert average_path_length(0) == 0.0
    assert average_path_length(1) == 0.0
    assert average_path_length(2) == 1.0
    # c(256) from the paper's formula, computed by hand
    h = math.log(255) + 0.5772156649015329
    assert average_path_length(256) == pytest.approx(2 * h - 2 * 255 / 256)


def test_average_path_length_close_to_exact_harmonic():
    # c(n) = 2 H(n-1) - 2(n-1)/n with the exact harmonic number
    n = 1000
    exact = 2 * sum(1 / i for i in range(1, n)) - 2 * (n - 1) / n
    assert average_path_length(n) == pytest.approx(exact, rel=1e-3)


# --------------------------------------------------------------- tree
@pytest.mark.parametrize("level", [None, 0, 1])
def test_tree_partitions_training_points(level):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 2))
    tree = IsolationTree(max_depth=50, extension_level=level, rng=rng).fit(X)
    leaves = tree.apply(X)
    assert tree.is_leaf_[leaves].all()
    # Unlimited depth on distinct points: every point isolated in its own leaf.
    assert np.unique(leaves).size == len(X)
    assert (tree.size_[leaves] == 1).all()
    # Children sizes add up to the parent
    internal = np.flatnonzero(~tree.is_leaf_)
    assert (tree.size_[tree.left_[internal]] + tree.size_[tree.right_[internal]] == tree.size_[internal]).all()


def test_tree_depth_limit_and_leaf_credit():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((256, 3))
    tree = IsolationTree(max_depth=3, rng=rng).fit(X)
    assert tree.depth_.max() <= 3
    big = tree.is_leaf_ & (tree.size_ > 1)
    assert big.any()
    assert np.allclose(tree.leaf_value_[big], tree.depth_[big] + average_path_length(tree.size_[big]))


def test_tree_handles_duplicate_points():
    X = np.ones((50, 2))
    tree = IsolationTree(max_depth=10, rng=np.random.default_rng(0)).fit(X)
    assert tree.n_nodes == 1
    assert tree.path_length(X)[0] == pytest.approx(average_path_length(50))


# --------------------------------------------------------------- forest
def test_obvious_outlier_scores_highest():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.standard_normal((500, 2)), [[8.0, 8.0]]])
    for ext in (False, True):
        s = IsolationForest(random_state=0, extended=ext).fit(X).score_samples(X)
        assert s.argmax() == len(X) - 1
        assert s[-1] > 0.65
        assert np.median(s[:-1]) < 0.5


def test_scores_in_unit_interval_and_deterministic():
    X, _ = datasets.blobs(seed=3)
    a = IsolationForest(random_state=7).fit(X).score_samples(X)
    b = IsolationForest(random_state=7).fit(X).score_samples(X)
    c = IsolationForest(random_state=8).fit(X).score_samples(X)
    assert ((a > 0) & (a <= 1)).all()
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_contamination_sets_flag_rate():
    X, _ = datasets.blobs(n=2000, seed=0)
    model = IsolationForest(contamination=0.1, random_state=0).fit(X)
    assert model.predict(X).mean() == pytest.approx(0.1, abs=0.01)


@pytest.mark.parametrize("name", sorted(datasets.REGISTRY))
def test_detects_synthetic_anomalies(name):
    X, y = datasets.REGISTRY[name](seed=0)
    for ext in (False, True):
        s = IsolationForest(random_state=0, extended=ext).fit(X).score_samples(X)
        assert roc_auc(y, s) > 0.9


def test_extended_beats_standard_on_ring():
    std, ext = [], []
    for seed in range(3):
        X, y = datasets.ring(seed=seed)
        std.append(average_precision(y, IsolationForest(random_state=seed).fit(X).score_samples(X)))
        ext.append(average_precision(y, IsolationForest(random_state=seed, extended=True).fit(X).score_samples(X)))
    assert np.mean(ext) > np.mean(std) + 0.2


def test_explanations_point_at_perturbed_feature():
    X, y, feat = datasets.local_feature(seed=1)
    model = IsolationForest(random_state=0).fit(X)
    e = model.explain(X)
    assert np.allclose(e.sum(axis=1), 1.0)
    idx = np.flatnonzero(y)
    assert (e[idx].argmax(axis=1) == feat[idx]).mean() >= 0.9


def test_input_validation():
    with pytest.raises(ValueError):
        IsolationForest(contamination=0.7)
    with pytest.raises(ValueError):
        IsolationForest().fit([[np.nan, 1.0], [0.0, 1.0]])
    with pytest.raises(RuntimeError):
        IsolationForest().score_samples([[0.0]])
    m = IsolationForest(random_state=0).fit(np.random.default_rng(0).standard_normal((50, 3)))
    with pytest.raises(ValueError):
        m.score_samples(np.zeros((2, 4)))
    with pytest.raises(ValueError):
        IsolationForest(extended=5).fit(np.zeros((10, 2)) + np.arange(10)[:, None])


def test_small_dataset_uses_all_samples():
    X = np.random.default_rng(0).standard_normal((20, 2))
    m = IsolationForest(random_state=0).fit(X)
    assert m.psi_ == 20


# --------------------------------------------------------------- metrics
def test_roc_auc_hand_computed():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert roc_auc([1, 1, 0, 0], [1, 2, 3, 4]) == 0.0


def test_average_precision_hand_computed():
    # ranking: 0.8(+) 0.4(-) 0.35(+) 0.1(-): AP = 0.5*1 + 0.5*(2/3)
    assert average_precision([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.5 + 1 / 3)


def test_metrics_match_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.random(300) + y * 0.3, 2)  # rounding creates ties
    assert roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s))
    assert average_precision(y, s) == pytest.approx(sk.average_precision_score(y, s))


def test_comparable_to_sklearn_isolation_forest():
    ens = pytest.importorskip("sklearn.ensemble")
    X, y = datasets.blobs(seed=2)
    ours = roc_auc(y, IsolationForest(n_estimators=200, random_state=0).fit(X).score_samples(X))
    theirs = roc_auc(y, -ens.IsolationForest(n_estimators=200, random_state=0).fit(X).score_samples(X))
    assert abs(ours - theirs) < 0.02


# --------------------------------------------------------------- cli
def test_cli_detect(tmp_path, capsys):
    rng = np.random.default_rng(0)
    X = np.vstack([rng.standard_normal((200, 3)), [[0.0, 12.0, 0.0]]])
    path = tmp_path / "d.csv"
    np.savetxt(path, X, delimiter=",", header="a,b,c", comments="")
    main(["detect", str(path), "--top", "3", "--trees", "50"])
    out = capsys.readouterr().out
    first = [line for line in out.splitlines() if line.strip().startswith("200")]
    assert first and "b " in first[0]


def test_cli_map_runs(capsys):
    main(["map", "ring", "--size", "6", "--trees", "20"])
    assert "extended" in capsys.readouterr().out
