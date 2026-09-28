# isolation-forest-explained

Isolation Forest and Extended Isolation Forest written from scratch in NumPy, with per-feature explanations for every flagged row, rank metrics implemented by hand, and a CLI that draws the anomaly score landscape right in your terminal.

## Why this exists

Most anomaly detectors ask "how far is this point from normal?", which means modelling what normal looks like. Isolation Forest flips the question: anomalies are few and different, so if you keep slicing the data at random, they get cut off from everyone else in very few slices. Average that depth over a few hundred random trees and you have a score that needs no distance metric, no density estimate, and runs in `O(n log psi)`.

The classic version only cuts along axes, and that leaves visible artifacts: bands of "normal looking" space that exist only because two clusters happen to share an x range or a y range. The Extended Isolation Forest (Hariri et al., 2018) cuts with random hyperplanes instead. This repo implements both behind one switch so you can see the difference, measure it, and explain what each model is reacting to.

## Quick start

```bash
pip install numpy            # the only runtime dependency
python -m isoforest bench    # standard vs extended on three synthetic datasets
python -m isoforest map ring # ASCII heatmap of the score landscape
python -m isoforest detect your_data.csv --top 10 --extended
```

Or install it as a package (adds an `isoforest` command):

```bash
pip install -e ".[test]"
pytest
```

## Using it from Python

```python
from isoforest import IsolationForest, roc_auc

model = IsolationForest(n_estimators=200, contamination=0.05, extended=True, random_state=0).fit(X)
scores = model.score_samples(X)   # in (0, 1], higher = more anomalous
flags = model.predict(X)          # 1 = anomaly, 0 = inlier
why = model.explain(X)            # (n_samples, n_features), rows sum to 1
```

`extended` accepts `False` (axis-parallel), `True` (fully extended), or an integer extension level `k`, where each split direction uses `k + 1` random coordinates.

## What the benchmark shows

`python -m isoforest bench` (5 seeds, 100 trees, psi = 256):

```
dataset                  AUC std   AUC ext    AP std    AP ext
--------------------------------------------------------------
blobs                      0.980     0.981     0.921     0.927
two_blobs_diagonal         0.999     1.000     0.945     1.000
ring                       0.957     0.994     0.254     0.762
```

The ring dataset is the interesting one. Anomalies sit in the empty centre of a circle, so every one of them has a perfectly ordinary x value and a perfectly ordinary y value. Axis-parallel cuts struggle to wall them off, and average precision collapses to 0.25. Diagonal cuts fix it (0.76). ROC AUC hides most of this because it is dominated by the thousands of easy inliers, which is why both metrics are reported.

`python -m isoforest map ring --size 14` makes the reason visible (darker = more anomalous; every other row shown, side by side):

```
standard (axis-parallel)            extended
@@@@####*********+****##%@@@        %%%%%%%%%%%%%%%%%%%%%%%@@@@@
###*++==-:-:....:::-===+*###        ######**+=-:::::::=+**#%#%%%
***+-..::-----------:::-+***        ##**=:.-==++++++++==-::=**##
+++-  .::-=----------:..-+++        **+-.-==++********++==-:=**#
+++=. .::-=---------:...=***        *++-.:--=++**+++++==--::+*##
***+=-:.::-:::::::....:-+***        ###*+=-::-===--:::...:+*#%%%
@@@%#***+++======++*****%@@@        %%%%%%##*+++======++*##%%%@@
```

The standard forest paints a cross shaped band of low scores through the middle and treats the centre as barely unusual. The extended forest produces a round well that actually follows the data.

## Explanations

`model.explain(X)` answers "which feature made this row stand out?" For each tree, a sample walks from root to leaf; each split it passes shrinks its node from `n_parent` to `n_child` points, so that split bought `log(n_parent / n_child)` nats of isolation for this sample. The credit is shared among the split's features in proportion to `|normal|`, which makes the same rule work for axis cuts and hyperplanes. Finally the credit is divided by the average credit on a reference sample, so a feature only stands out when it isolates this row more than it isolates typical rows.

On the `local_feature` dataset (8 dimensional Gaussian, each anomaly pushed out on exactly one random feature) the top ranked feature matches the perturbed one for every anomaly. The CLI prints this next to each flagged row. Here 200 Gaussian rows plus one row pushed to `b = 12`:

```
$ python -m isoforest detect demo.csv --top 3
IsolationForest: 200 trees, psi=201, axis-parallel, mean nodes/tree=101.7, threshold=0.5380
201 rows, 10 flagged, 0.68s

   row    score  flag  top features
   200   0.7231    X   b 90%, c 5%, a 4%
    73   0.6605    X   a 62%, c 26%, b 12%
   101   0.6460    X   a 79%, c 11%, b 10%
```

## Design decisions

* **Trees are flat arrays, not node objects.** Each tree stores `normals`, `offsets`, `left`, `right`, `size`, `depth` as NumPy arrays. Scoring pushes every sample down the tree at once, level by level, with one `einsum` per level. No Python recursion anywhere, so there is no recursion limit to hit and scoring is vectorised.
* **One split representation for both variants.** A node sends `x` left when `x . normal <= offset`. An axis-parallel split is just a one-hot normal. This removed a whole code path and guarantees the two variants differ only in how splits are drawn.
* **Empty hyperplane splits are redrawn.** A random hyperplane through a random point of a node's bounding box can miss every sample, especially for thin diagonal clusters. The reference EIF code keeps such empty children; this implementation redraws (up to 32 times), which keeps trees compact and every split meaningful.
* **Leaves that hit the depth limit are credited with `c(size)`.** Growth stops at `ceil(log2(psi))` as in the paper, and the unexplored subtree is replaced by its expected path length, so scores stay unbiased.
* **Metrics are implemented, not imported.** ROC AUC uses the Mann Whitney U statistic with averaged tie ranks; average precision is the step interpolated PR area. Both are checked against scikit-learn in the test suite (on data with deliberate ties).

## Testing

`pytest` runs 23 tests covering:

* the `c(n)` normaliser against the exact harmonic number formula
* tree invariants: every training point reaches a leaf, children sizes sum to the parent, unlimited depth isolates every distinct point, duplicates do not loop forever
* determinism under a fixed seed, contamination controls the flag rate
* detection quality on every synthetic dataset for both variants, and that extended beats standard on the ring by a clear margin
* explanations pick the perturbed feature for at least 90% of anomalies
* ROC AUC and AP equal scikit-learn's values; our standard forest's AUC is within 0.02 of scikit-learn's `IsolationForest` (skipped automatically if scikit-learn is not installed)
* the CLI end to end on a generated CSV

## Project layout

```
isoforest/
  tree.py       IsolationTree: flat array tree, both split types, path length, attribution
  forest.py     IsolationForest: subsampling, scoring, thresholds, explain()
  metrics.py    roc_auc, average_precision, precision_at_k
  datasets.py   synthetic datasets with ground truth
  cli.py        bench / map / detect commands
tests/
  test_isoforest.py
```

## References

* F. T. Liu, K. M. Ting, Z. H. Zhou. *Isolation Forest.* ICDM 2008.
* S. Hariri, M. Carrasco Kind, R. J. Brunner. *Extended Isolation Forest.* IEEE TKDE 2021 (arXiv 1811.02141).
* M. Carletti, M. Terzi, G. A. Susto. *Interpretable Anomaly Detection with DIFFI.* 2020, the inspiration for the attribution scheme.

## License

MIT
