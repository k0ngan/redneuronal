"""Red neuronal simple implementada solo con NumPy."""

from __future__ import annotations

import numpy as np
from typing import Iterable, List


class NeuralNetwork:
    """Red neuronal feed-forward para clasificaci\u00f3n binaria."""

    def __init__(self, layers: Iterable[int], lr: float = 0.01, random_state: int = 42) -> None:
        np.random.seed(random_state)
        self.lr = lr
        self.layers = list(layers)
        # Pesos y biases inicializados peque\u00f1os
        self.W: List[np.ndarray] = [
            np.random.randn(self.layers[i], self.layers[i + 1]) * 0.1
            for i in range(len(self.layers) - 1)
        ]
        self.b: List[np.ndarray] = [
            np.zeros((1, n)) for n in self.layers[1:]
        ]

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _dsigmoid(a: np.ndarray) -> np.ndarray:
        return a * (1 - a)

    def _forward(self, X: np.ndarray) -> List[np.ndarray]:
        """Propagaci\u00f3n hacia adelante."""
        A = [X]
        for W, b in zip(self.W, self.b):
            Z = A[-1] @ W + b
            A.append(self._sigmoid(Z))
        return A

    def _backward(self, A: List[np.ndarray], y: np.ndarray) -> List[np.ndarray]:
        """Retropropagaci\u00f3n de errores."""
        deltas = [(A[-1] - y) * self._dsigmoid(A[-1])]
        for i in range(len(self.W) - 1, 0, -1):
            delta = (deltas[0] @ self.W[i].T) * self._dsigmoid(A[i])
            deltas.insert(0, delta)
        grads_W = [A[i].T @ deltas[i] / len(y) for i in range(len(self.W))]
        grads_b = [delta.mean(axis=0, keepdims=True) for delta in deltas]
        return grads_W, grads_b

    def _update(self, dW: List[np.ndarray], db: List[np.ndarray]) -> None:
        for i in range(len(self.W)):
            self.W[i] -= self.lr * dW[i]
            self.b[i] -= self.lr * db[i]

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000) -> List[float]:
        """Entrena la red neuronal."""
        losses: List[float] = []
        for _ in range(epochs):
            A = self._forward(X)
            loss = np.mean((A[-1] - y) ** 2)
            dW, db = self._backward(A, y)
            self._update(dW, db)
            losses.append(loss)
        return losses

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        A = self._forward(X)
        return A[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)
