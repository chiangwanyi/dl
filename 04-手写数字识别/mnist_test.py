
import gzip
import struct
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ================== 1. 数据目录 ==================

DATA_DIR = Path(__file__).parent / "data"


# ================== 2. 读取 MNIST 图片 ==================

def load_images(filename):
    """读取 MNIST 图片，返回形状为 (样本数, 28, 28) 的数组。"""

    path = DATA_DIR / filename

    with gzip.open(path, "rb") as f:
        # 图片文件头：魔数、图片数量、行数、列数
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))

        if magic != 2051:
            raise ValueError(f"图片文件格式错误：{filename}")

        # 读取所有像素，并转换为 NumPy 数组
        data = f.read()
        images = np.frombuffer(data, dtype=np.uint8)

        expected_size = num_images * rows * cols
        if images.size != expected_size:
            raise ValueError(
                f"{filename} 数据不完整："
                f"预期 {expected_size} 个像素，实际 {images.size} 个"
            )

        return images.reshape(num_images, rows, cols)


# ================== 3. 读取 MNIST 标签 ==================

def load_labels(filename):
    """读取 MNIST 标签，返回形状为 (样本数,) 的数组。"""

    path = DATA_DIR / filename

    with gzip.open(path, "rb") as f:
        # 标签文件头：魔数、标签数量
        magic, num_labels = struct.unpack(">II", f.read(8))

        if magic != 2049:
            raise ValueError(f"标签文件格式错误：{filename}")

        data = f.read()
        labels = np.frombuffer(data, dtype=np.uint8)

        if labels.size != num_labels:
            raise ValueError(
                f"{filename} 数据不完整："
                f"预期 {num_labels} 个标签，实际 {labels.size} 个"
            )

        return labels


# ================== 4. 加载训练集和测试集 ==================

X_train = load_images("train-images-idx3-ubyte.gz")
y_train = load_labels("train-labels-idx1-ubyte.gz")

X_test = load_images("t10k-images-idx3-ubyte.gz")
y_test = load_labels("t10k-labels-idx1-ubyte.gz")


# ================== 5. 查看数据 ==================

print("========== MNIST 数据加载成功 ==========")
print("训练图片 X_train：", X_train.shape)
print("训练标签 y_train：", y_train.shape)
print("测试图片 X_test：", X_test.shape)
print("测试标签 y_test：", y_test.shape)

print("\n第一张图片的标签：", y_train[0])
print("第一张图片的像素矩阵：")
print(X_train[0])


# ================== 6. 显示第一张图片 ==================

plt.imshow(X_train[0], cmap="gray")
plt.title(f"MNIST Label: {y_train[0]}")
plt.axis("off")
plt.show()
