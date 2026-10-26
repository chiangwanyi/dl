import json
import os
import time

import autograd.numpy as np
from autograd import grad

from mnist_loader import load_mnist

np.random.seed(42)

def train_network(X, Y, n_hidden, lr, epochs, seed=1, tag="", optimizer="gd"):
    rng = np.random.default_rng(seed)

    # P = [W1(784,32), b1(1,32), W2(32,10), b2(1,10)]
    P = [rng.standard_normal((X.shape[1], n_hidden)) * 0.5,
             np.zeros((1, n_hidden)),
             rng.standard_normal((n_hidden, Y.shape[1])) * 0.5,
             np.zeros((1, Y.shape[1]))]

    def forward(W1, b1, W2, b2):
        H = np.tanh(X @ W1 + b1)
        return H @ W2 + b2

    def loss_fn(W1, b1, W2, b2):
        return np.mean((forward(W1, b1, W2, b2) - Y) ** 2)

    grads = grad(loss_fn, argnum=[0, 1, 2, 3])

    m = [np.zeros_like(p) for p in P]
    v = [np.zeros_like(p) for p in P]

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        gs = grads(*P)
        for i in range(4):
            # 带有梯度记忆和自适应步幅的梯度下降算法
            if optimizer == "adam":
                m[i] = 0.9 * m[i] + 0.1 * gs[i]
                v[i] = 0.999 * v[i] + 0.001 * gs[i] ** 2
                P[i] = P[i] - lr * (m[i] / (1 - 0.9 ** epoch)) / \
                    (np.sqrt(v[i] / (1 - 0.999 ** epoch)) + 1e-8)
            else:
                P[i] = P[i] - lr * gs[i]

        if epoch % 1000 == 0 or epoch == epochs:
            print(f"    {tag} epoch {epoch:5d}/{epochs}  loss {loss_fn(*P):.6f}"
                  f"  {time.time() - t0:.1f}s")

    return P

def main():
    X_train, y_train, X_test, y_test = load_mnist()

    print("========== MNIST 数据加载成功 ==========")
    print("训练图片 X_train：", X_train.shape)
    print("训练标签 y_train：", y_train.shape)
    print("测试图片 X_test：", X_test.shape)
    print("测试标签 y_test：", y_test.shape)

    # X_train(60000, 28, 28) -> (60000, 784)
    # y_train(60000,) -> (60000, 10)

    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_train = np.zeros((y_train.shape[0], 10))
    print("调整训练集形状：", X_train.shape, Y_train.shape)
    print("\n第一张图片的标签：", y_train[0])
    print("第一张图片的像素矩阵：")
    print(X_train[0])
    # 打印第1张图片的像素矩阵
    print("第一张图片的标签：", Y_train[0])


if __name__ == "__main__":
    main()    