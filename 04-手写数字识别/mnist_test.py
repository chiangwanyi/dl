import matplotlib.pyplot as plt

from mnist_loader import load_mnist


# ================== 1. 加载训练集和测试集 ==================

X_train, y_train, X_test, y_test = load_mnist()


# ================== 2. 查看数据 ==================

print("========== MNIST 数据加载成功 ==========")
print("训练图片 X_train：", X_train.shape)
print("训练标签 y_train：", y_train.shape)
print("测试图片 X_test：", X_test.shape)
print("测试标签 y_test：", y_test.shape)

print("\n第一张图片的标签：", y_train[0])
print("第一张图片的像素矩阵：")
print(X_train[0])


# ================== 3. 显示第一张图片 ==================

plt.imshow(X_train[0], cmap="gray")
plt.title(f"MNIST Label: {y_train[0]}")
plt.axis("off")
plt.show()
