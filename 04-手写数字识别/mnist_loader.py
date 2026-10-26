"""MNIST 数据集加载模块。

其他脚本可以直接复用：

    from mnist_loader import load_mnist

    X_train, y_train, X_test, y_test = load_mnist()
    # X_train: uint8, (60000, 28, 28)
    # y_train: uint8, (60000,)

如果模型需要展平的输入，加 flatten=True：

    X_train, y_train, X_test, y_test = load_mnist(flatten=True)
    # X_train: uint8, (60000, 784)

数据文件默认放在本文件同级的 data/ 目录下，也支持传入自定义目录。
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np

# 默认数据目录：与本文件同级
DATA_DIR = Path(__file__).resolve().parent / "data"

# MNIST 官方文件名的简写映射
FILENAMES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

IMAGES_MAGIC = 2051
LABELS_MAGIC = 2049


def load_images(filename: str, data_dir: str | Path | None = None) -> np.ndarray:
    """读取 MNIST 图片文件，返回形状为 (样本数, 行, 列) 的 uint8 数组。"""

    path = Path(data_dir or DATA_DIR) / filename

    with gzip.open(path, "rb") as f:
        # 图片文件头：魔数、图片数量、行数、列数
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))

        if magic != IMAGES_MAGIC:
            raise ValueError(f"图片文件格式错误：{filename}（魔数 {magic}）")

        data = f.read()
        images = np.frombuffer(data, dtype=np.uint8)

        expected_size = num_images * rows * cols
        if images.size != expected_size:
            raise ValueError(
                f"{filename} 数据不完整："
                f"预期 {expected_size} 个像素，实际 {images.size} 个"
            )

        return images.reshape(num_images, rows, cols)


def load_labels(filename: str, data_dir: str | Path | None = None) -> np.ndarray:
    """读取 MNIST 标签文件，返回形状为 (样本数,) 的 uint8 数组。"""

    path = Path(data_dir or DATA_DIR) / filename

    with gzip.open(path, "rb") as f:
        # 标签文件头：魔数、标签数量
        magic, num_labels = struct.unpack(">II", f.read(8))

        if magic != LABELS_MAGIC:
            raise ValueError(f"标签文件格式错误：{filename}（魔数 {magic}）")

        data = f.read()
        labels = np.frombuffer(data, dtype=np.uint8)

        if labels.size != num_labels:
            raise ValueError(
                f"{filename} 数据不完整："
                f"预期 {num_labels} 个标签，实际 {labels.size} 个"
            )

        return labels


def load_mnist(
    data_dir: str | Path | None = None,
    flatten: bool = False,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """一次性加载训练集和测试集。

    返回 (X_train, y_train, X_test, y_test)。

    参数
    ----
    data_dir : 数据所在目录，默认是本文件同级的 data/
    flatten  : True 时把每张图片展平成 784 维，形状 (样本数, 784)
    normalize: True 时把像素缩放到 [0, 1] 的 float32
    """

    root = Path(data_dir or DATA_DIR)

    X_train = load_images(FILENAMES["train_images"], root)
    y_train = load_labels(FILENAMES["train_labels"], root)
    X_test = load_images(FILENAMES["test_images"], root)
    y_test = load_labels(FILENAMES["test_labels"], root)

    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()
    print("========== MNIST 数据加载成功 ==========")
    print("训练图片 X_train：", X_train.shape, X_train.dtype)
    print("训练标签 y_train：", y_train.shape, y_train.dtype)
    print("测试图片 X_test：", X_test.shape, X_test.dtype)
    print("测试标签 y_test：", y_test.shape, y_test.dtype)
    print("\n第一张图片的标签：", y_train[0])
