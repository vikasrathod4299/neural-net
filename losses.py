import numpy as np


class SoftmaxCrossEntropy:
    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.probs = exp / exp.sum(axis=1, keepdims=True)
        self.y = y
        N = logits.shape[0]
        return -np.log(self.probs[np.arange(N), y] + 1e-15).mean()

    def backward(self) -> np.ndarray:
        N = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(N), self.y] -= 1.0
        return grad / N
