"""Dataset abstractions + synthetic + sklearn loaders. Data-only, no math."""
from __future__ import annotations
import numpy as np


class ArrayDataset:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert len(X) == len(y), "X/y length mismatch"
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class DataLoader:
    def __init__(self, ds: ArrayDataset, batch: int = 32, shuffle: bool = True,
                 rng: np.random.Generator | None = None):
        self.ds = ds
        self.batch = batch
        self.shuffle = shuffle
        self.rng = rng or np.random.default_rng()

    def __iter__(self):
        idx = np.arange(len(self.ds))
        if self.shuffle:
            self.rng.shuffle(idx)
        for s in range(0, len(idx), self.batch):
            b = idx[s:s + self.batch]
            yield self.ds.X[b], self.ds.y[b]


def make_xor(n: int = 200, rng: np.random.Generator | None = None):
    r = rng or np.random.default_rng(0)
    X = r.integers(0, 2, size=(n, 2)).astype(float)
    y = (X[:, 0].astype(int) ^ X[:, 1].astype(int)).astype(int)
    return X, y


def make_spirals(n_per: int = 300, noise: float = 0.05,
                 rng: np.random.Generator | None = None):
    """Twin spirals: 2D binary classification, linearly inseparable."""
    r = rng or np.random.default_rng(0)
    k = np.arange(n_per)
    radius = k / n_per
    theta = k / n_per * 3.5 * np.pi
    x0 = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    x1 = np.stack([radius * np.cos(theta + np.pi),
                   radius * np.sin(theta + np.pi)], axis=1)
    X = np.vstack([x0, x1]) + r.normal(0, noise, (2 * n_per, 2))
    y = np.hstack([np.zeros(n_per, dtype=int), np.ones(n_per, dtype=int)])
    idx = r.permutation(2 * n_per)
    return X[idx], y[idx]


def load_digits_flat(test_size: float = 0.2, seed: int = 0):
    """sklearn 8x8 digits: 1797x64 (0-16) -> 0..1, y 0..9. Loader-only sklearn use."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    d = load_digits()
    X = d.data.astype(float) / 16.0
    y = d.target.astype(int)
    return train_test_split(X, y, test_size=test_size, random_state=seed)
