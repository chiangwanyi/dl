
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
noise_x = np.random.randn(100) * 0.1
noise_y = np.random.randn(100) * 0.1

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
