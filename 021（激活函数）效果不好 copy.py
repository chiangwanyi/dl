import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt

np.random.seed(42)

# ================== 1. 生成训练数据 ==================

# 生成 100 个 0 到 5 秒之间的时间点 (1,100)
t = np.linspace(0, 5, 100)

x = 20 * t
y = 30 * t - 0.5 * 9.8 * t ** 2

noise_x = np.random.randn(100) * 0.5
noise_y = np.random.randn(100) * 0.4

x = x + noise_x
y = y + noise_y

# 构造输入矩阵 X (100,1)
X_raw = t.reshape(-1, 1)
# 构造答案矩阵 Y (100,2)
Y_raw = np.column_stack((x, y))

# 标准化输入
X_mean = np.mean(X_raw, axis=0)
X_std = np.std(X_raw, axis=0)
X = (X_raw - X_mean) / X_std

# 标准化输出
Y_mean = np.mean(Y_raw, axis=0)
Y_std = np.std(Y_raw, axis=0)
Y = (Y_raw - Y_mean) / Y_std

# ================== 2. 初始化网络参数 ==================

# 输入层 -> 隐藏层：1 个输入，16 个神经元
W1 = np.random.randn(1, 16) * 0.5
b1 = np.zeros((1, 16))

# 隐藏层 -> 输出层：16 个输入，2 个输出
W2 = np.random.randn(16, 2) * 0.5
b2 = np.zeros((1, 2))

# ================== 3. 定义前向传播 ==================

def forward(X, W1, b1, W2, b2):
    H = np.tanh(X @ W1 + b1)
    Y_pred = H @ W2 + b2
    return Y_pred

# ================== 4. 定义损失函数 ==================

def loss_fn(W1, b1, W2, b2):
    Y_pred = forward(X, W1, b1, W2, b2)
    return np.mean((Y_pred - Y) ** 2)

# ================== 5. 计算梯度 ==================

grad_W1 = grad(loss_fn, 0)
grad_b1 = grad(loss_fn, 1)
grad_W2 = grad(loss_fn, 2)
grad_b2 = grad(loss_fn, 3)

# ================== 6. 梯度下降训练 ==================

learning_rate = 0.01
epochs = 10000

for epoch in range(epochs):
    dW1 = grad_W1(W1, b1, W2, b2)
    db1 = grad_b1(W1, b1, W2, b2)
    dW2 = grad_W2(W1, b1, W2, b2)
    db2 = grad_b2(W1, b1, W2, b2)

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss_fn(W1, b1, W2, b2):.4f}")

# ================== 7. 预测与绘图 ==================

# 得到标准化空间中的预测值
Y_pred_scaled = forward(X, W1, b1, W2, b2)

# 还原成真实位置
Y_pred = Y_pred_scaled * Y_std + Y_mean

plt.figure(figsize=(10, 5))

plt.scatter(t, x, s=15, label="Actual x")
plt.scatter(t, y, s=15, label="Actual y")

plt.plot(t, Y_pred[:, 0], label="Predicted x")
plt.plot(t, Y_pred[:, 1], label="Predicted y")

plt.xlabel("Time (t)")
plt.ylabel("Position")
plt.legend()
plt.grid(True)
plt.show()        