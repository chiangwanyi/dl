"""训练过程快照导出：给 web/train.html 提供「同一张图在不同训练阶段跑一遍」的数据。

做的事情：
    1. 完全复用 mnist_cnn.py 的算子与前向（conv3x3 / maxpool2 / forward / loss_fn），
       保证网页里看到的东西就是 mnist_cnn.py 训练出来的东西；
    2. 在训练途中按固定步数打快照（checkpoint），记录当时的全部权重 + loss + 准确率；
    3. 挑几张测试集图片一起导出，网页端就能逐轮次看这些图每层的形状和像素变化。

输出：
    web/trace_cnn.js      window.CNN_TRACE = {...}

用法：
    python trace_cnn.py                    # 默认 3 个 epoch，约 8 个快照
    python trace_cnn.py --epochs 1 --limit 6000   # 冒烟测试
"""

import argparse
import json
import time
from pathlib import Path

import autograd.numpy as np
from autograd import grad

from mnist_cnn import (IMG, K, conv3x3, forward, init_params, loss_fn,
                       maxpool2, predict_proba, relu, softmax)
from mnist_loader import load_mnist

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE / "web" / "trace_cnn.js"

W_KEYS = ["W1", "b1", "W2", "b2", "Wf", "bf", "Wo", "bo"]


def snapshot(P, decimals=5):
    """把 8 个参数数组压成一份可 JSON 化的快照（保留 5 位小数，肉眼无差别但体积减半）。"""
    return {k: np.round(p, decimals).ravel().tolist() for k, p in zip(W_KEYS, P)}


def inspect(P, x):
    """把单张图 x(28,28,1) 逐层跑一遍，返回每层的激活值（网页端会独立重算一遍）。

    只给校验用：导出时用 numpy 算一份参考值，网页初始化时对一遍，
    不一致就说明 JS 那边的索引写错了，不会默默画出一堆假图。
    """
    W1, b1, W2, b2, Wf, bf, Wo, bo = P
    out = {"input": x}
    out["conv1"] = relu(conv3x3(x, W1, b1))
    out["pool1"] = maxpool2(out["conv1"])
    out["conv2"] = relu(conv3x3(out["pool1"], W2, b2))
    out["pool2"] = maxpool2(out["conv2"])
    flat = out["pool2"].reshape(1, -1)
    out["flat"] = flat
    out["fc1"] = relu(flat @ Wf + bf)
    out["fc2"] = out["fc1"] @ Wo + bo
    out["soft"] = softmax(out["fc2"])
    return out


def train_traced(P, X, Y, X_test, y_test, lr=0.001, epochs=3, batch_size=128,
                 seed=42, n_ckpt=8, probe_n=4000):
    """和 mnist_cnn.train 逐字一致的训练循环，只多了「到点拍照」这一件事。

    拍照时机：把总步数均分成 n_ckpt 段，每段结束时存一份权重。
    第 0 份是初始化状态（一步都没走），方便对比「没训练」和「训练后」。
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    labels = np.argmax(Y, axis=1)
    grads = grad(loss_fn, argnum=0)
    m = [np.zeros_like(p) for p in P]
    v = [np.zeros_like(p) for p in P]

    total_steps = epochs * ((n + batch_size - 1) // batch_size)
    ckpt_every = max(1, total_steps // n_ckpt)

    def evaluate(P):
        """在测试子集上算准确率，顺便留几个能看出「学没学会」的诊断量。"""
        sub = min(probe_n, X_test.shape[0])
        probs = predict_proba(P, X_test[:sub])
        pred = np.argmax(probs, axis=1)
        acc = float(np.mean(pred == y_test[:sub]))
        conf = float(np.mean(probs[np.arange(sub), pred]))
        return acc, conf

    checkpoints, curve = [], {"step": [], "loss": []}
    live = []   # 和 checkpoints 一一对应的「未取整」参数，导完快照后用来算参考值

    acc, conf = evaluate(P)
    checkpoints.append({"epoch": 0, "step": 0, "loss": None, "stepLoss": None,
                        "testAcc": acc, "meanConf": conf, "P": snapshot(P)})
    live.append([np.array(p, copy=True) for p in P])
    print(f"  快照 0/… step 0（初始化）  test_acc {acc:.4f}")

    step, t0 = 0, time.time()
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

            batch_loss = float(loss_fn(P, xb, yb))
            total_loss += batch_loss * len(b)
            seen += len(b)
            curve["step"].append(step)
            curve["loss"].append(round(batch_loss, 5))

            if step % ckpt_every == 0 or step == total_steps:
                acc, conf = evaluate(P)
                checkpoints.append({
                    "epoch": epoch, "step": step,
                    "loss": float(total_loss / seen), "stepLoss": batch_loss,
                    "testAcc": acc, "meanConf": conf, "P": snapshot(P)})
                live.append([np.array(p, copy=True) for p in P])
                print(f"  快照 {len(checkpoints) - 1} step {step:5d}/{total_steps}"
                      f"  loss {total_loss / seen:.4f}  test_acc {acc:.4f}"
                      f"  {time.time() - t0:5.1f}s")

        print(f"    epoch {epoch} 完成  loss {total_loss / seen:.4f}")

    return checkpoints, curve, live


def pick_probes(X_test, y_test, per_digit=4):
    """每个数字挑 per_digit 张，尽量挑笔画清晰的（墨迹量接近该类中位数）。

    注意 X_test 是 uint8，直接取原值即可 —— 曾经在这里写过 `img * 255`，
    uint8 乘法溢出把整张图变成错位的乱码，而网页照旧能画出来，很难发现。
    """
    out = []
    for d in range(10):
        cand = np.where(y_test == d)[0]
        ink = X_test[cand].reshape(len(cand), -1).sum(axis=1)
        order = cand[np.argsort(np.abs(ink - np.median(ink)))]
        for i in order[:per_digit]:
            out.append({"x": X_test[i].ravel().astype(int).tolist(), "label": int(d)})
    return out


def main():
    ap = argparse.ArgumentParser(description="导出 CNN 训练过程快照给网页")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--ckpt", type=int, default=8, help="快照数量（含初始化那份）")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    X_train, y_train, X_test, y_test = load_mnist()
    if args.limit:
        X_train, y_train = X_train[:args.limit], y_train[:args.limit]

    X_train = X_train.astype(np.float64)[..., None] / 255.0
    X_test_f = X_test.astype(np.float64)[..., None] / 255.0

    Y_train = np.zeros((y_train.shape[0], 10))
    Y_train[np.arange(y_train.shape[0]), y_train] = 1.0

    rng = np.random.default_rng(42)
    P = init_params(rng)

    print("开始训练并记录快照…")
    checkpoints, curve, live = train_traced(
        P, X_train, Y_train, X_test_f, y_test,
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
        n_ckpt=args.ckpt)

    images = pick_probes(X_test, y_test)

    # 参考值：每个快照在这 40 张探针图上的 softmax 输出，网页端会独立重算来对账
    probes = np.stack([np.array(im["x"], dtype=np.float64).reshape(IMG, IMG, 1) / 255.0
                       for im in images])
    for ck, Pl in zip(checkpoints, live):
        probs = softmax(forward(Pl, probes))
        ck["probeProbs"] = np.round(probs, 5).tolist()
        # 第 0 张探针图各层激活均值，用来校验中间层（不只看最后输出）
        act = inspect(Pl, probes[0:1])
        ck["probeMeans"] = {k: round(float(np.mean(act[k])), 5)
                            for k in ("conv1", "pool1", "conv2", "pool2",
                                      "flat", "fc1", "fc2")}

    # 护栏：训练好的模型在自己挑的探针图上应该几乎全对。
    # 如果这个数字很低，多半是探针图导出错了（比如像素被写坏），而不是模型不行。
    probe_acc = float(np.mean(np.argmax(np.array(checkpoints[-1]["probeProbs"]), axis=1)
                              == np.array([im["label"] for im in images])))
    print(f"\n探针图准确率（最后一个快照）：{probe_acc:.1%}")
    if probe_acc < 0.85:
        raise SystemExit("探针图准确率异常，导出数据可能有问题，先别拿去画图。")

    trace = {
        "kind": "cnn-trace",
        "arch": {"img": IMG, "k": K, "c1": 8, "c2": 16, "fc": 64, "classes": 10},
        "layout": "channel-last",
        "hyper": {"epochs": args.epochs, "lr": args.lr,
                  "batchSize": args.batch_size, "limit": args.limit,
                  "trainSize": int(X_train.shape[0])},
        "params": int(sum(int(np.prod(p.shape)) for p in P)),
        "images": images,
        "checkpoints": checkpoints,
        "curve": curve,
    }

    text = json.dumps(trace, separators=(",", ":"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"window.CNN_TRACE = {text};\n", encoding="utf-8")
    print(f"\n已写入 {out}（{len(text) / 1024 / 1024:.2f} MB）")
    print(f"快照 {len(checkpoints)} 个，曲线 {len(curve['step'])} 个点，"
          f"探针图 {len(trace['images'])} 张")


if __name__ == "__main__":
    main()
