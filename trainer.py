"""Training loop. Depends only on Model/Loss/Optimizer/DataLoader abstractions."""
from __future__ import annotations
import numpy as np


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == y).mean())


def evaluate(model, loss, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    logits = model.forward(X)
    return float(loss.forward(logits, y)), accuracy(logits, y)


def fit(model, loss, opt, loader, epochs: int = 1000, log_every: int = 200,
        val: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    hist = {"loss": []}
    for ep in range(1, epochs + 1):
        total, nb = 0.0, 0
        for Xb, yb in loader:
            logits = model.forward(Xb)
            l = loss.forward(logits, yb)
            model.backward(loss.backward())
            opt.step(model.params_grads())
            total += l
            nb += 1
        avg = total / max(nb, 1)
        hist["loss"].append(avg)
        if ep % log_every == 0 or ep == 1:
            msg = f"ep {ep}/{epochs} loss {avg:.4f}"
            if val is not None:
                vl, va = evaluate(model, loss, val[0], val[1])
                msg += f" | val_loss {vl:.4f} val_acc {va:.3f}"
            print(msg, flush=True)
    return hist
