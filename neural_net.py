import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def backward(self, grad_out: np.ndarray) -> np.ndarray: ...
    def params(self) -> list[np.ndarray]:
        return []

    def grads(self) -> list[np.ndarray]:
        return []


class Dense(Layer):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        init: str = "he",
        rng: np.random.Generator | None = None,
    ):
        if init not in ["he", "xavier", "small"]:
            raise ValueError(f"Invalid init method: {init}. Must be 'he', 'xavier' or 'small'.")
        r = rng or np.random.default_rng()

        if init == "he":  # for relu activation functions
            scale = np.sqrt(2 / in_dim)
        elif init == "xavier":  # for sigoid/tanh activation functions
            scale = np.sqrt(1 / in_dim)
        else:  # for linear activation functions
            scale = 0.01

        self.W: np.ndarray = r.standard_normal((in_dim, out_dim)) * scale
        self.b: np.ndarray = np.zeros(out_dim)
        self.dW: np.ndarray = np.zeros_like(self.W)
        self.db: np.ndarray = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != self.W.shape[0]:
            raise ValueError(f"Input must be a 2D array, got {x.ndim}D array instead.")

        self.x = x

        return x @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        # grad_out: (N, out_dim) = dL/dY
        self.dW = self.x.T @ grad_out
        self.db = grad_out.sum(axis=0)

        return grad_out @ self.W.T  # dL/dX

    def params(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]


class ReLU(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return np.where(self.mask, x, 0.0)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self.mask


class Tanh(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * (1.0 - self.out**2)


class Sigmoid(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
        return self.out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self.out * (1 - self.out)


class Sequential:
    def __init__(self, layers: list[Layer]):
        if not layers:
            raise ValueError("Sequential model must contain at least one layer.")
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad_out = layer.backward(grad_out)
        return grad_out

    def params_grads(self) -> list[tuple[np.ndarray, np.ndarray]]:
        out = []
        for layer in self.layers:
            out.extend(zip(layer.params(), layer.grads()))
        return out
