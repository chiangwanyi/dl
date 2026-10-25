"""手写反向传播的两层/多层感知机。刻意不依赖 torch：
这个模型只有几千个参数，numpy 足够，而且导出的权重格式和浏览器里的
前向实现可以严格对应，省掉一层"框架差异"带来的隐患。
"""

import numpy as np


def silu(z):
    return z / (1.0 + np.exp(-z))


def silu_grad(z):
    s = 1.0 / (1.0 + np.exp(-z))
    return s * (1.0 + z * (1.0 - s))


class MLP:
    def __init__(self, sizes, rng, seed_init=1.0):
        self.sizes = list(sizes)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            W = rng.standard_normal((fan_in, fan_out)) * (np.sqrt(2.0 / fan_in) * seed_init)
            self.W.append(W.astype(np.float32))
            self.b.append(np.zeros(fan_out, dtype=np.float32))

    # ---------------------------------------------------------------- forward
    def forward(self, X, keep=False):
        acts = [X]
        zs = []
        A = X
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Z = A @ W + b
            zs.append(Z)
            A = Z if i == last else silu(Z)
            acts.append(A)
        if keep:
            self._cache = (acts, zs)
        return A

    # --------------------------------------------------------------- backward
    def backward(self, dA):
        acts, zs = self._cache
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        for i in range(len(self.W) - 1, -1, -1):
            if i != len(self.W) - 1:
                dA = dA * silu_grad(zs[i])
            gW[i] = acts[i].T @ dA
            gb[i] = dA.sum(axis=0)
            dA = dA @ self.W[i].T
        return gW, gb

    def n_params(self):
        return sum(w.size for w in self.W) + sum(b.size for b in self.b)


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=1e-5):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for i, (p, g) in enumerate(zip(params, grads)):
            if self.wd:
                g = g + self.wd * p
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            p -= (self.lr / c1) * self.m[i] / (np.sqrt(self.v[i] / c2) + self.eps)
