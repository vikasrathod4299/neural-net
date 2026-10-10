import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def step(self, params_grads: list[tuple[np.ndarray, np.ndarray]]): ...


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self, params_grads):
        for p, g in params_grads:
            p -= self.lr * g


class Adam(Optimizer):
    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params_grads):
        self.t += 1
        for i, (p, g) in enumerate(params_grads):
            if i not in self.m:
                self.m[i] = np.zeros_like(g)
                self.v[i] = np.zeros_like(g)

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g**2)

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
