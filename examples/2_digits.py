"""8x8 digits classifier. Run: uv run python examples/2_digits.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from neural_net import Dense, ReLU, Sequential
from losses import SoftmaxCrossEntropy
from optimizers import Adam
from data import ArrayDataset, DataLoader, load_digits_flat
from trainer import accuracy


def main():
    rng = np.random.default_rng(0)
    Xtr, Xva, ytr, yva = load_digits_flat()
    loader = DataLoader(ArrayDataset(Xtr, ytr), batch=64, shuffle=True, rng=rng)

    model = Sequential([Dense(64, 128, rng=rng), ReLU(),
                        Dense(128, 10, rng=rng)])
    loss, opt = SoftmaxCrossEntropy(), Adam(lr=1e-3)

    for ep in range(1, 51):
        for Xb, yb in loader:
            logits = model.forward(Xb)
            loss.forward(logits, yb)
            model.backward(loss.backward())
            opt.step(model.params_grads())
        if ep % 10 == 0 or ep == 1:
            tra = accuracy(model.forward(Xtr), ytr)
            vaa = accuracy(model.forward(Xva), yva)
            print(f"ep {ep} train_acc {tra:.3f} val_acc {vaa:.3f}", flush=True)

    # show 10 predictions
    logits = model.forward(Xva[:10])
    pred = logits.argmax(axis=1)
    for i in range(10):
        print(f"true={yva[i]} pred={pred[i]}")


if __name__ == "__main__":
    main()
