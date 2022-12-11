import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import LinearRegression

def get_train_data(fp, index):
    X_train = pd.read_csv('./' + fp + '/train_X_' + str(index) + '.csv',header=None)
    y_train = pd.read_csv('./' + fp + '/train_y_' + str(index) + '.csv',header=None)
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
    X_test, y_test= get_test_data()
    LR = LinearRegression(lr, num_epochs, batch_size, init_weights)
    LR.train(X_train, y_train)
    print(f"training R^2: {LR.score(X_train, y_train)}")
    print(f"testing R^2: {LR.score(X_test, y_test)}")
    # LR.plot_performance()
    return LR.output_gradient()


def plot(fp="", mses=None, rsquares=None):

    iters = list(range(1, mses.shape[0]+1))

    fig = plt.figure(dpi=300, figsize=(5,5))
    if rsquares is not None:  
        plt.plot(iters, rsquares)
        plt.scatter(iters, rsquares, s=11)

    # plt.gca().set(xlim=rang, ylim=rang)
    plt.title(f"Performance vs Training Iteration (at iter{iters[-1]})", fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    fig.tight_layout()
    plt.savefig(f"{fp}/iter{iters[-1]}_rsquare.png")
    plt.close(fig)


    fig = plt.figure(dpi=300, figsize=(5,5))
    if mses is not None: 
        plt.plot(iters, mses)
        plt.scatter(iters, mses, s=11)

    # plt.gca().set(xlim=rang, ylim=rang)
    plt.title(f"Performance vs Training Iteration (at iter{iters[-1]})", fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    fig.tight_layout()
    plt.savefig(f"{fp}/iter{iters[-1]}_mse.png")
    plt.close(fig)


if __name__ == '__main__':
    train_model(fp = "data1", id = 0)