import numpy as np

class NeuralNetwork:
    def __init__(self, layers, lr=0.01):
        self.lr = lr
        self.W = [np.random.randn(layers[i], layers[i+1])*0.1
                  for i in range(len(layers)-1)]
        self.b = [np.zeros((1, n)) for n in layers[1:]]

    @staticmethod
    def sigmoid(z): return 1/(1+np.exp(-z))
    @staticmethod
    def dsigmoid(a): return a*(1-a)

    def forward(self, X):
        A, Z = [X], []
        for W, b in zip(self.W, self.b):
            Z.append(A[-1] @ W + b)
            A.append(self.sigmoid(Z[-1]))
        return A, Z

    def backward(self, A, Z, y):
        dW, db = [], []
        delta = (A[-1]-y) * self.dsigmoid(A[-1])
        for i in reversed(range(len(self.W))):
            dW.insert(0, A[i].T @ delta / len(y))
            db.insert(0, delta.mean(axis=0, keepdims=True))
            if i:  # propagate if not input layer
                delta = (delta @ self.W[i].T) * self.dsigmoid(A[i])
        return dW, db

    def step(self, dW, db):
        for i in range(len(self.W)):
            self.W[i] -= self.lr * dW[i]
            self.b[i] -= self.lr * db[i]

    def fit(self, X, y, epochs=1000):
        history = []
        for _ in range(epochs):
            A, Z = self.forward(X)
            loss = np.mean((A[-1]-y)**2)
            dW, db = self.backward(A, Z, y)
            self.step(dW, db)
            history.append(loss)
        return history

    def predict(self, X):
        A, _ = self.forward(X)
        return (A[-1] > 0.5).astype(int)
