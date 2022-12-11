import numpy as np
import pandas as pd
from model import LinearRegression

def get_train_data(index):
    X_train = pd.read_csv('./data/train_X_' + str(index) + '.csv',header=None)
    y_train = pd.read_csv('./data/train_y_' + str(index) + '.csv',header=None)
    X_train = np.append(X_train, np.ones((len(X_train), 1)), axis=1)
    y_train = np.array([i[0] for i in y_train.values])
    return X_train, y_train

def get_test_data():
    X_test = pd.read_csv('./data/test_X.csv',header=None)
    y_test = pd.read_csv('./data/test_y.csv',header=None)
    X_test = np.append(X_test, np.ones((len(X_test), 1)), axis=1)
    y_test = np.array([i[0] for i in y_test.values])
    return X_test, y_test

def train_model(id, lr = 1e-4, num_epochs = 10000, batch_size = 48, init_weights = None):
    X_train, y_train = get_train_data(id)
    LR = LinearRegression(lr, num_epochs, batch_size, init_weights)
    LR.train(X_train, y_train)
    return LR.output_gradient()

if __name__ == '__main__':
    train_model(id = 0)