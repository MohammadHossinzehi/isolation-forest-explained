"""Isolation Forest and Extended Isolation Forest, from scratch in NumPy."""

from .forest import IsolationForest
from .metrics import average_precision, precision_at_k, roc_auc
from .tree import IsolationTree, average_path_length

__all__ = [
    "IsolationForest",
    "IsolationTree",
    "average_path_length",
    "roc_auc",
    "average_precision",
    "precision_at_k",
]
__version__ = "0.1.0"
