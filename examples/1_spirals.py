"""Twin spirals with decision-boundary plot. Run: uv run python examples/1_spirals.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from neural_net import Dense, ReLU, Sequential
from losses import SoftmaxCrossEntropy
from optimizers import Adam
from data import ArrayDataset, DataLoader, make_spirals
from trainer import accuracy


def main():
    rng = np.random.default_rng(0)
    X, y = make_spirals(n_per=300, noise=0.05, rng=rng)
    loader = DataLoader(ArrayDataset(X, y), batch=64, shuffle=True, rng=rng)

    model = Sequential([Dense(2, 64, rng=rng), ReLU(),
                        Dense(64, 64, rng=rng), ReLU(),
                        Dense(64, 2, rng=rng)])
    loss, opt = SoftmaxCrossEntropy(), Adam(lr=1e-2)

    for ep in range(1, 3001):
        total, nb = 0.0, 0
        for Xb, yb in loader:
            logits = model.forward(Xb)
            total += loss.forward(logits, yb)
            model.backward(loss.backward())
            opt.step(model.params_grads())
            nb += 1
        if ep % 500 == 0 or ep == 1:
            acc = accuracy(model.forward(X), y)
            print(f"ep {ep} loss {total/nb:.4f} acc {acc:.3f}", flush=True)

    # decision boundary
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-.2, X[:, 0].max()+.2, 200),
                         np.linspace(X[:, 1].min()-.2, X[:, 1].max()+.2, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    pred = model.forward(grid).argmax(axis=1).reshape(xx.shape)
    plt.contourf(xx, yy, pred, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=8)
    plt.title(f"spirals acc={accuracy(model.forward(X), y):.3f}")
    plt.savefig("spirals_boundary.png")
    print("saved spirals_boundary.png")


if __name__ == "__main__":
    main()
