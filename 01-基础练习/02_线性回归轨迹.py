
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt

np.random.seed(42)

# 第一步：生成 100 个 0 到 5 秒之间的时间点
t = np.linspace(0, 5, 100)

# 第二步：根据真实公式生成水平位置
# 水平初速度 vx = 20 m/s， 竖直初速度 vy = 30 m/s，重力加速度 g = 9.8 m/s²
x = 20 * t

# 第三步：根据真实公式生成竖直位置
y = 30 * t - 0.5 * 9.8 * t ** 2

# 第四步：给位置添加少量随机噪声
noise_x = np.random.randn(100) * 0.5
noise_y = np.random.randn(100) * 0.4

x = x + noise_x
y = y + noise_y

# 第五步：构造输入矩阵 X
# 每一行对应一个样本，每一列对应一个特征 t、t²
X = np.column_stack((t, t ** 2))

# 第六步：构造答案矩阵 Y
# 每一行对应一个样本，每一列对应 x、y
Y = np.column_stack((x, y))

print("X 的形状：", X.shape)
print("Y 的形状：", Y.shape)
print("X 前 5 行：\n", X[:5])
print("Y 前 5 行：\n", Y[:5])

#============================================================

# 第七步：初始化模型参数
W = np.random.randn(2, 2)  # 权重矩阵，2 个输入特征，2 个输出
b = np.zeros((1, 2))  # 偏置向量，2 个输出
print("W 的形状：", W.shape)
print("b 的形状：", b.shape)

# 第九步：定义损失函数
def loss_fn(W, b):
    Y_pred = X @ W + b
    loss = np.mean((Y_pred - Y) ** 2)
    return loss

initial_loss = loss_fn(W, b)

# argnum=0：对第一个参数 W 求梯度
grad_W_fn = grad(loss_fn, argnum=0)

# argnum=1：对第二个参数 b 求梯度
grad_b_fn = grad(loss_fn, argnum=1)

learning_rate = 0.01
epochs = 5000

loss_history = []

for epoch in range(epochs):
    # 记录当前参数对应的损失
    current_loss = loss_fn(W, b)
    loss_history.append(current_loss)

    dW = grad_W_fn(W, b)
    db = grad_b_fn(W, b)

    # 更新参数
    W = W - learning_rate * dW
    b = b - learning_rate * db

    # 每隔 1000 轮输出一次训练信息
    if (epoch + 1) % 1000 == 0:
        print(
            f"Epoch [{epoch + 1:3d}/{epochs}], "
            f"Loss: {current_loss:.6f}"
        )

final_loss = loss_fn(W, b)

print("\n========== 最终训练结果 ==========")

print("训练权重 W：\n", W)

print("训练偏置 b：\n", b)

print(f"\n训练前 Loss：{initial_loss:.6f}")
print(f"训练后 Loss：{final_loss:.6f}")        

# 第十步：计算训练完成后的预测结果
Y_pred = X @ W + b

# 绘制导弹轨迹
plt.figure(figsize=(10, 5))

plt.scatter(
    Y[:, 0], Y[:, 1],
    label="Observed trajectory",
    s=15
)

plt.plot(
    Y_pred[:, 0], Y_pred[:, 1],
    label="Predicted trajectory",
    linewidth=2
)

plt.xlabel("Horizontal position (m)")
plt.ylabel("Vertical position (m)")
plt.title("Missile Trajectory Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 绘制损失曲线
plt.figure(figsize=(10, 5))

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.grid(True)
plt.tight_layout()
plt.show()