from re import I
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegression(object):
    def __init__(self, lr = 1e-4, num_epochs = 10000, batch_size = 48, init_weights = None):
        self.init_weights = init_weights
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.weights = None
        self.losses = []
        self.scores = []
        if self.init_weights is not None:
            print("loading initial weight")
            self.weights = np.copy(self.init_weights)

    def train(self, X, Y):
        n_examples = X.shape[0]
        if self.init_weights is None:
            print("no initial weight")
            self.weights = np.zeros((1, X.shape[1]))
        for _ in range(self.num_epochs):
            indices = np.arange(n_examples)
            np.random.shuffle(indices)
            X, Y = X[indices], Y[indices]
            for i in range(n_examples // self.batch_size + (n_examples % self.batch_size != 0)):
                X_batch = X[i * self.batch_size: min(n_examples, (i + 1) * self.batch_size)]
                Y_batch = Y[i * self.batch_size: min(n_examples, (i + 1) * self.batch_size)]
                m = len(X_batch)
                W_gradient = np.zeros((1, X.shape[1]))
                for x, y in zip(X_batch, Y_batch):
                    W_gradient[0] += 2/m * (x @ np.transpose(self.weights).reshape(-1) - y) * x
                self.weights -= self.lr * W_gradient
            self.losses.append(self.MSE(X, Y))
            self.scores.append(self.score(X, Y))

    def predict(self, X):
        predictions = X @ np.transpose(self.weights).reshape(-1)
        return predictions

    def MSE(self, X, y):
        n_examples = len(y)
        loss = 0
        predictions = self.predict(X)
        for i in range(n_examples):
            loss += (y[i]-predictions[i])**2/n_examples
        return loss
    
    def score(self, X, y):
        predictions = self.predict(X)
        score = 1.0 - np.sum((y - predictions)** 2)/np.sum((y - y.mean()) ** 2)
        return score

    def plot_performance(self):
        plt.plot(self.losses)
        plt.plot(self.scores)
        plt.show()
    
    def output_gradient(self):
        if self.init_weights is None:
            return self.weights
        else:
            return self.weights - self.init_weights