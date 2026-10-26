import time

import autograd.numpy as np
from autograd import grad

from mnist_loader import load_mnist

np.random.seed(42)


def softmax(z):
    """把 logits 转成概率，减去每行最大值是为了防止 exp 溢出。"""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_network(X, Y, n_hidden, lr, epochs, batch_size=128,
                  seed=1, tag="", optimizer="adam", loss="ce"):
    """训练一个 784 -> n_hidden -> 10 的两层 MLP。

    batch_size: 每步用多少张图计算梯度（全量传 60000 就是原始行为）
    loss:       "ce" 交叉熵（分类首选）/ "mse" 均方误差（能跑但很慢）
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    # P = [W1(784,32), b1(1,32), W2(32,10), b2(1,10)]
    P = [rng.standard_normal((X.shape[1], n_hidden)) * 0.2,
         np.zeros((1, n_hidden)),
         rng.standard_normal((n_hidden, Y.shape[1])) * 0.2,
         np.zeros((1, Y.shape[1]))]

    def forward(W1, b1, W2, b2, xb):
        return np.tanh(xb @ W1 + b1) @ W2 + b2

    def loss_fn(W1, b1, W2, b2, xb, yb):
        z = forward(W1, b1, W2, b2, xb)
        if loss == "ce":
            p = softmax(z)
            return -np.mean(np.sum(yb * np.log(p + 1e-12), axis=1))
        return np.mean((z - yb) ** 2)

    grads = grad(loss_fn, argnum=[0, 1, 2, 3])

    m = [np.zeros_like(p) for p in P]
    v = [np.zeros_like(p) for p in P]
    labels = np.argmax(Y, axis=1)

    step = 0          # 优化器的真实步数，小批量下一轮有几百步
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        idx = rng.permutation(n)          # 每轮打乱，避免固定顺序
        total_loss = 0.0

        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            xb, yb = X[b], Y[b]

            gs = grads(*P, xb, yb)
            step += 1

            for i in range(4):
                if optimizer == "adam":
                    m[i] = 0.9 * m[i] + 0.1 * gs[i]
                    v[i] = 0.999 * v[i] + 0.001 * gs[i] ** 2
                    # 偏差修正要用 step，不能用 epoch
                    bc1 = 1 - 0.9 ** step
                    bc2 = 1 - 0.999 ** step
                    P[i] = P[i] - lr * (m[i] / bc1) / (np.sqrt(v[i] / bc2) + 1e-8)
                else:
                    P[i] = P[i] - lr * gs[i]

            total_loss += loss_fn(*P, xb, yb) * len(b)

        acc = np.mean(np.argmax(forward(*P, X), axis=1) == labels)
        passed = time.time() - t0
        eta = passed / epoch * (epochs - epoch)
        print(f"    {tag} epoch {epoch:3d}/{epochs}  loss {total_loss / n:.4f}"
              f"  train_acc {acc:.4f}  {passed:5.1f}s  (剩余约 {eta:.0f}s)")

    return P


def predict(X, P):
    W1, b1, W2, b2 = P
    return np.tanh(X @ W1 + b1) @ W2 + b2


def one_hot(y, num_classes=10):
    """把数字标签 (N,) 转成独热编码 (N, num_classes)。

    y = [5, 0, 9]  ->  [[0,0,0,0,0,1,0,0,0,0],
                        [1,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,1]]
    """
    Y = np.zeros((y.shape[0], num_classes))
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


def main():
    X_train, y_train, X_test, y_test = load_mnist()

    print("========== MNIST 数据加载成功 ==========")
    print("训练图片 X_train：", X_train.shape)
    print("训练标签 y_train：", y_train.shape)
    print("测试图片 X_test：", X_test.shape)
    print("测试标签 y_test：", y_test.shape)

    # X_train(60000, 28, 28) -> (60000, 784)，并缩放到 0~1
    # y_train(60000,) -> (60000, 10) 独热编码
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    Y_train = one_hot(y_train, 10)
    Y_test = one_hot(y_test, 10)

    print("\n调整训练集形状：", X_train.shape, Y_train.shape)
    print("调整测试集形状：", X_test.shape, Y_test.shape)

    mlp = train_network(X_train, Y_train, n_hidden=256, lr=0.001, epochs=10,
                        batch_size=128, seed=42, tag="MLP",
                        optimizer="adam", loss="ce")

    # ================== 测试 ==================
    logits = predict(X_test, mlp)
    probs = softmax(logits)
    pred = np.argmax(logits, axis=1)
    acc = np.mean(pred == y_test)
    print(f"\n测试集准确率：{acc:.4f}（{np.sum(pred == y_test)}/{len(y_test)}）")

    # 只看前 10 张
    print("\n前 10 张的预测：")
    for i in range(10):
        print(f"  第 {i} 张  真实 {y_test[i]}  预测 {pred[i]}"
              f"  置信度 {probs[i, pred[i]]:.3f}"
              f"  {'✓' if pred[i] == y_test[i] else '✗'}")


if __name__ == "__main__":
    main()
