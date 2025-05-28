import numpy as np

class NeuralNetwork:
    def __init__(self, learning_rate=0.01):
        self.learningRate = learning_rate
        self.b1 = np.zeros((1, 32))
        self.w1 = np.random.randn(7500, 32) * np.sqrt(2.0 / 7500)  # He initialization
        self.b2 = np.zeros((1, 3))
        self.w2 = np.random.randn(32, 3) * np.sqrt(1.0 / 32)  # Xavier initialization

    def ReLU(self, v):
        return np.maximum(0, v)

    def ReLU_derivative(self, v):
        return (v > 0).astype(float)

    def softmax(self, v):
        e_v = np.exp(v - np.max(v, axis=1, keepdims=True))
        return e_v / e_v.sum(axis=1, keepdims=True)

    def forwardPropagation(self, X):
        self.Z1 = np.dot(X, self.w1) + self.b1
        self.A1 = self.ReLU(self.Z1)
        self.Z2 = np.dot(self.A1, self.w2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def computeLoss(self, Y_hat, Y):
        m = Y.shape[0]
        loss = -np.sum(Y * np.log(Y_hat + 1e-9)) / m
        return loss

    def backwardPropagation(self, X, Y):
        m = X.shape[0]
        dZ2 = self.A2 - Y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.w2.T)
        dZ1 = dA1 * self.ReLU_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.w1 -= self.learningRate * dW1
        self.b1 -= self.learningRate * db1
        self.w2 -= self.learningRate * dW2
        self.b2 -= self.learningRate * db2

    def train(self, X, Y, epochs=100, print_every=10):
        losses = []
        for epoch in range(epochs):
            Y_hat = self.forwardPropagation(X)
            loss = self.computeLoss(Y_hat, Y)
            losses.append(loss)
            self.backwardPropagation(X, Y)
            if epoch % print_every == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
        return self.forwardPropagation(X), losses


