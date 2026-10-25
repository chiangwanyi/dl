import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad


# ============================================================
# 1. 生成训练数据
# ============================================================

np.random.seed(42)

# 输入数据：10 个样本，每个样本有 1 个特征
X = np.random.rand(10, 1) * 10

# 真实参数：1 个输入特征，2 个输出
W_true = np.array([[2.0, -1.0]])
b_true = np.array([[3.0, 5.0]])

# 添加随机噪声，模拟真实数据
noise = np.random.randn(10, 2) * 2.0

# 根据真实线性关系生成目标值
y = X @ W_true + b_true + noise

print("========== 训练数据 ==========")
print("X 的形状：", X.shape)
print("y 的形状：", y.shape)
print("真实权重 W：\n", W_true)
print("真实偏置 b：\n", b_true)


# ============================================================
# 2. 初始化模型参数
# ============================================================

# 随机初始化权重和偏置
W = np.random.randn(1, 2)
b = np.random.randn(1, 2)

print("\n========== 初始参数 ==========")
print("初始 W：\n", W)
print("初始 b：\n", b)


# ============================================================
# 3. 定义前向传播与损失函数
# ============================================================

def predict(X, W, b):
    """根据输入数据和模型参数，计算预测值。"""
    return X @ W + b


def loss_fn(W, b):
    """计算预测值与真实值之间的均方误差（MSE）。"""
    y_pred = predict(X, W, b)
    error = y_pred - y
    return np.mean(error ** 2)


initial_loss = loss_fn(W, b)

print("\n========== 初始损失 ==========")
print(f"Initial Loss: {initial_loss:.6f}")


# ============================================================
# 4. 使用 Autograd 自动计算梯度
# ============================================================

# argnum=0：对第一个参数 W 求梯度
grad_W_fn = grad(loss_fn, argnum=0)

# argnum=1：对第二个参数 b 求梯度
grad_b_fn = grad(loss_fn, argnum=1)

dW = grad_W_fn(W, b)
db = grad_b_fn(W, b)

print("\n========== 初始梯度 ==========")
print("dW：\n", dW)
print("db：\n", db)


# ============================================================
# 5. 使用有限差分验证梯度
# ============================================================

epsilon = 1e-5

# 仅扰动 W[0, 0]，其他参数保持不变
W_plus = W.copy()
W_minus = W.copy()

W_plus[0, 0] += epsilon
W_minus[0, 0] -= epsilon

# 中心差分近似偏导数
numerical_gradient = (
    loss_fn(W_plus, b) - loss_fn(W_minus, b)
) / (2.0 * epsilon)

# Autograd 计算的对应偏导数
autograd_gradient = dW[0, 0]

print("\n========== 梯度验证 ==========")
print(f"有限差分梯度：{numerical_gradient:.8f}")
print(f"Autograd 梯度：{autograd_gradient:.8f}")
print(f"绝对误差：{abs(numerical_gradient - autograd_gradient):.10e}")


# ============================================================
# 6. 使用梯度下降训练模型
# ============================================================

learning_rate = 0.01
epochs = 100

loss_history = []

for epoch in range(epochs):

    # 6.1 记录当前参数对应的损失
    current_loss = loss_fn(W, b)
    loss_history.append(current_loss)

    # 6.2 计算当前参数的梯度
    dW = grad_W_fn(W, b)
    db = grad_b_fn(W, b)

    # 6.3 沿着梯度的反方向更新参数
    W = W - learning_rate * dW
    b = b - learning_rate * db

    # 6.4 每隔 10 轮输出一次训练信息
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch + 1:3d}/{epochs}], "
            f"Loss: {current_loss:.6f}"
        )


# ============================================================
# 7. 查看训练结果
# ============================================================

final_loss = loss_fn(W, b)

print("\n========== 最终训练结果 ==========")

print("真实权重 W：\n", W_true)
print("训练权重 W：\n", W)

print("\n真实偏置 b：\n", b_true)
print("训练偏置 b：\n", b)

print(f"\n训练前 Loss：{initial_loss:.6f}")
print(f"训练后 Loss：{final_loss:.6f}")


# ============================================================
# 8. 绘制损失曲线
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, epochs + 1),
    loss_history,
    label="Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Loss During Training")

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()