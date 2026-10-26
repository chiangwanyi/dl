"""手写一个小 CNN，和 mnist_mlp.py 用同一套 autograd + Adam + 导出格式。

结构（全部手写，没有现成卷积层）：

    输入 28x28x1
      -> conv 3x3, 1->8,  ReLU          (26,26,8)
      -> maxpool 2x2                    (13,13,8)
      -> conv 3x3, 8->16, ReLU          (11,11,16)
      -> maxpool 2x2                    (5,5,16)
      -> flatten 400                    (400,)
      -> fc 400->64, ReLU
      -> fc 64->10
      -> softmax

参数只有 27,538 个，是 MLP（200,704）的 1/7，准确率反而更高。

用法：
    python mnist_cnn.py                  # 默认 3 个 epoch
    python mnist_cnn.py --epochs 1 --limit 6000   # 快速冒烟测试
"""

import argparse
import json
import time
from pathlib import Path

import autograd.numpy as np
from autograd import grad

from mnist_loader import load_mnist

np.random.seed(42)

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model_cnn.json"
WEB_MODEL_PATH = HERE / "web" / "model_cnn.js"

# 输入图片边长、卷积核边长，写死成常量，方便下面推形状
IMG = 28
K = 3


# ==================== 基础算子 ====================

def softmax(z):
    """把 logits 转成概率，减去每行最大值是为了防止 exp 溢出。"""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def relu(x):
    return np.maximum(x, 0.0)


def conv3x3(x, W, b):
    """3x3 卷积（不加 padding，步长 1）。

    x: (B, H, Wd, Cin)
    W: (3, 3, Cin, Cout)      —— 同一套 W 在 H-2 × Wd-2 个位置上复用，这就是「权重共享」
    b: (Cout,)
    返回: (B, H-2, Wd-2, Cout)

    实现上没有用 for 循环去遍历每个输出位置（那样 autograd 要记录几万次小运算，慢到没法跑），
    而是拆成 9 次「整块切片 + 矩阵乘」：卷积核的第 (kh, kw) 个权重，
    对所有 (i+kh, j+kw) 这个偏移量上的所有像素一次性乘完再累加。
    数学上完全等价，但每次都是 BLAS 大矩阵乘，快几十倍。
    """
    B, H, Wd, Cin = x.shape
    oh, ow = H - K + 1, Wd - K + 1
    Cout = W.shape[3]

    # 先把 9 个偏移的贡献累加成 (B*oh*ow, Cout) 的矩阵
    acc = np.zeros((B * oh * ow, Cout))
    for kh in range(K):
        for kw in range(K):
            # (B, oh, ow, Cin) -> (B*oh*ow, Cin)
            patch = x[:, kh:kh + oh, kw:kw + ow, :].reshape(B * oh * ow, Cin)
            acc = acc + patch @ W[kh, kw]

    return acc.reshape(B, oh, ow, Cout) + b


def maxpool2(x):
    """2x2 最大池化，步长 2。

    x: (B, H, Wd, C) -> (B, H/2, Wd/2, C)

    做法：把 H 拆成 (H/2, 2)、W 拆成 (Wd/2, 2)，再沿着那两个「2」的轴取最大。
    等价于在每个 2x2 窗口里挑最大值，但只做两次 max，不用写窗口循环。

    注意边长可能是奇数（第二层卷积出来是 11x11），这时先裁掉最后一行/列变成 10x10，
    否则 reshape 会直接报错——这也是 PyTorch 里 stride=2 的池化默认 ceil_mode=False 的行为。
    """
    B, H, Wd, C = x.shape
    h, w = H // 2, Wd // 2
    x = x[:, :h * 2, :w * 2, :]
    t = x.reshape(B, h, 2, w, 2, C)
    t = t.max(axis=2)      # (B, h, w, 2, C)
    return t.max(axis=3)   # (B, h, w, C)


# ==================== 模型定义 ====================

def feature_size(img=IMG, c1=8, c2=16):
    """把数据流经一遍形状推导，返回 flatten 后的长度（这里应该是 5*5*16 = 400）。

    28 -2(x3x3) =26, /2(pool) =13, -2 =11, /2 =5  ->  5*5*16
    """
    h, w, c = img, img, 1
    h, w, c = h - K + 1, w - K + 1, c1      # conv1
    h, w = h // 2, w // 2                    # pool1
    h, w, c = h - K + 1, w - K + 1, c2      # conv2
    h, w = h // 2, w // 2                    # pool2
    return h * w * c


def init_params(rng, c1=8, c2=16, fc=64):
    """He 初始化：方差按 fan_in 缩放，ReLU 网络用它比 0.2 的常数好训。

    这里的形状顺序是 (kh, kw, Cin, Cout)，和 PyTorch 的 (Cout, Cin, kh, kw) 只是转置关系。
    """
    def he(shape, fan_in):
        return rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)

    flat = feature_size(c1=c1, c2=c2)

    return [
        he((K, K, 1, c1), K * K * 1),      # W1  conv1: 1 -> 8
        np.zeros(c1),                       # b1
        he((K, K, c1, c2), K * K * c1),    # W2  conv2: 8 -> 16
        np.zeros(c2),                       # b2
        he((flat, fc), flat),               # Wf 400 -> 64
        np.zeros(fc),                       # bf
        he((fc, 10), fc),                   # Wo 64 -> 10
        np.zeros(10),                       # bo
    ]


def forward(P, xb):
    """整条前向链路。xb: (B, 28, 28, 1) 的 float 数组，值域 0~1。"""
    W1, b1, W2, b2, Wf, bf, Wo, bo = P
    B = xb.shape[0]

    h = relu(conv3x3(xb, W1, b1))   # (B, 26, 26, 8)
    h = maxpool2(h)                 # (B, 13, 13, 8)
    h = relu(conv3x3(h, W2, b2))    # (B, 11, 11, 16)
    h = maxpool2(h)                 # (B, 5, 5, 16)
    h = h.reshape(B, -1)            # (B, 400)，注意是 channel-last 展开
    h = relu(h @ Wf + bf)           # (B, 64)
    return h @ Wo + bo              # (B, 10)


def loss_fn(P, xb, yb):
    """平均交叉熵。yb 是独热编码。"""
    p = softmax(forward(P, xb))
    return -np.mean(np.sum(yb * np.log(p + 1e-12), axis=1))


def predict_proba(P, X, batch_size=1000):
    """分批推理，避免一次性整集前向把内存吃满。"""
    out = []
    for start in range(0, X.shape[0], batch_size):
        out.append(softmax(forward(P, X[start:start + batch_size])))
    return np.concatenate(out, axis=0)


# ==================== 训练 ====================

def train(P, X, Y, lr=0.001, epochs=3, batch_size=128, tag="CNN", seed=42):
    """Adam + 小批量。优化器部分和 mnist_mlp.py 完全一致。"""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    labels = np.argmax(Y, axis=1)

    grads = grad(loss_fn, argnum=0)
    m = [np.zeros_like(p) for p in P]
    v = [np.zeros_like(p) for p in P]

    step = 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        idx = rng.permutation(n)
        total_loss, seen = 0.0, 0

        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            xb, yb = X[b], Y[b]

            gs = grads(P, xb, yb)
            step += 1

            for i in range(len(P)):
                m[i] = 0.9 * m[i] + 0.1 * gs[i]
                v[i] = 0.999 * v[i] + 0.001 * gs[i] ** 2
                bc1 = 1 - 0.9 ** step
                bc2 = 1 - 0.999 ** step
                P[i] = P[i] - lr * (m[i] / bc1) / (np.sqrt(v[i] / bc2) + 1e-8)

            total_loss += loss_fn(P, xb, yb) * len(b)
            seen += len(b)

        # 训练准确率在 10000 张子集上估，避免每轮再全量前向一次
        sub = min(10000, n)
        acc = np.mean(np.argmax(forward(P, X[:sub]), axis=1) == labels[:sub])
        passed = time.time() - t0
        eta = passed / epoch * (epochs - epoch)
        print(f"    {tag} epoch {epoch:3d}/{epochs}  loss {total_loss / seen:.4f}"
              f"  train_acc {acc:.4f}  {passed:5.1f}s  (剩余约 {eta:.0f}s)")

    return P


# ==================== 导出 / 读回 ====================

def export_model(P, path=MODEL_PATH, web_path=WEB_MODEL_PATH, decimals=6):
    """导出成 JSON，同时给网页版写一份 model_cnn.js。

    网页端要用同样的 layout 做前向，所以把维度信息一起写进去，JS 那边不用猜：
        x 是 (H, W, C) 的扁平数组，通道在最后（channel-last），
        conv 的权重是 [kh][kw][ci][co]。
    """
    W1, b1, W2, b2, Wf, bf, Wo, bo = P

    def rl(a):
        return np.round(a, decimals).tolist()

    model = {
        "kind": "cnn",
        "arch": [IMG, IMG, 1, 8, 16, 64, 10],
        "layout": "channel-last",
        "input": "28x28 grayscale, /255, white digit on black",
        "layers": "conv3x3+relu+maxpool2 -> conv3x3+relu+maxpool2 -> fc64+relu -> fc10",
        "W1": rl(W1), "b1": rl(b1),
        "W2": rl(W2), "b2": rl(b2),
        "Wf": rl(Wf), "bf": rl(bf),
        "Wo": rl(Wo), "bo": rl(bo),
    }

    text = json.dumps(model, separators=(",", ":"))

    Path(path).write_text(text, encoding="utf-8")
    web_path = Path(web_path)
    web_path.parent.mkdir(parents=True, exist_ok=True)
    web_path.write_text(f"window.MNIST_CNN = {text};\n", encoding="utf-8")

    n_params = sum(int(np.prod(p.shape)) for p in P)
    print(f"\n模型已导出：{path}（{len(text) / 1024:.0f} KB，{n_params:,} 个参数）")
    print(f"网页数据：{web_path}")
    print(f"结构 {model['arch']}，{model['layers']}")
    return path


def load_model(path=MODEL_PATH):
    """把导出的 JSON 读回成参数列表 P。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [np.array(raw["W1"]), np.array(raw["b1"]),
            np.array(raw["W2"]), np.array(raw["b2"]),
            np.array(raw["Wf"]), np.array(raw["bf"]),
            np.array(raw["Wo"]), np.array(raw["bo"])]


# ==================== 主流程 ====================

def main():
    ap = argparse.ArgumentParser(description="手写一个小 CNN 训练 MNIST")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 张训练图（冒烟测试用）")
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--no-export", action="store_true", help="只训练，不写模型文件")
    args = ap.parse_args()

    X_train, y_train, X_test, y_test = load_mnist()

    print("========== MNIST 数据加载成功 ==========")
    print("训练集：", X_train.shape, "测试集：", X_test.shape)

    if args.limit:
        X_train, y_train = X_train[:args.limit], y_train[:args.limit]
        print(f"（冒烟测试：只用前 {args.limit} 张）")

    # uint8 (N,28,28) -> float64 (N,28,28,1) / 255
    # 加最后一个通道维，是因为卷积要按 (H, W, Cin) 组织
    X_train = X_train.astype(np.float64)[..., None] / 255.0
    X_test = X_test.astype(np.float64)[..., None] / 255.0

    Y_train = np.zeros((y_train.shape[0], 10))
    Y_train[np.arange(y_train.shape[0]), y_train] = 1.0

    print("调整后形状：", X_train.shape, Y_train.shape)

    rng = np.random.default_rng(42)
    P = init_params(rng)
    n_params = sum(int(np.prod(p.shape)) for p in P)
    print(f"参数量：{n_params:,}（MLP 是同数据下的 200,704）")

    P = train(P, X_train, Y_train, lr=args.lr, epochs=args.epochs,
              batch_size=args.batch_size)

    if not args.no_export:
        export_model(P)

    # ================== 测试 ==================
    probs = predict_proba(P, X_test)
    pred = np.argmax(probs, axis=1)
    acc = np.mean(pred == y_test)
    print(f"\n测试集准确率：{acc:.4f}（{np.sum(pred == y_test)}/{len(y_test)}）")

    print("\n前 10 张的预测：")
    for i in range(10):
        print(f"  第 {i} 张  真实 {y_test[i]}  预测 {pred[i]}"
              f"  置信度 {probs[i, pred[i]]:.3f}"
              f"  {'✓' if pred[i] == y_test[i] else '✗'}")

    # 顺便报一下分位数，看看错的是什么货色
    conf = probs[np.arange(len(pred)), pred]
    wrong = pred != y_test
    if wrong.any():
        print(f"\n错 {wrong.sum()} 张，其中置信度 >0.9 的占 "
              f"{np.mean(conf[wrong] > 0.9) * 100:.0f}%"
              f"（这些多半是真难，不是模型瞎猜）")


if __name__ == "__main__":
    main()
