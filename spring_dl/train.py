"""训练 + 评估 + 导出权重。

用法：
    python train.py            # 生成数据并训练（约 4 分钟）
    python train.py --reuse    # 复用 data_cache.npz
"""

import json
import os
import sys
import time

import numpy as np

from dataset import (DTAU, FEATURE_NAMES, N_STEPS, TAU_END, ZETA_RANGE, GAMMA_RANGE,
                     MIN_RADIUS, generate, make_features, make_training_pairs)
from model import MLP, Adam

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.dirname(HERE)          # 权重写到工作区根目录，和 HTML 放一起
CACHE = os.path.join(HERE, "data_cache.npz")

HIDDEN = [96, 96]
N_TRAJ = 1500
STEPS = 30000
BATCH = 1024
LR = 3e-3
EVAL_EVERY = 2500
SEED = 1
VPT_LEVELS = (0.10, 0.20)                # 偏差阈值，单位 L0


def load_data(reuse=False):
    if reuse and os.path.exists(CACHE):
        z = np.load(CACHE)
        print("复用缓存数据")
        return z["states"], z["zeta"], z["gamma"]
    t0 = time.time()
    states, zeta, gamma = generate(N_TRAJ, seed=SEED)
    np.savez_compressed(CACHE, states=states, zeta=zeta, gamma=gamma)
    print(f"  用时 {time.time() - t0:.1f}s，shape={states.shape}")
    return states, zeta, gamma


# ---------------------------------------------------------------- 预测器

class Predictor:
    def __init__(self, net, x_mean, x_std, y_mean, y_std):
        self.net = net
        self.xm, self.xs = x_mean, x_std
        self.ym, self.ys = y_mean, y_std

    def accel_feat(self, F):
        return self.net.forward((F - self.xm) / self.xs) * self.ys + self.ym

    def accel(self, Rx, Ry, Vx, Vy, zeta, gamma):
        B = np.shape(Rx)[0] if np.ndim(Rx) else 1
        X6 = np.empty((B, 6), dtype=np.float32)
        X6[:, 0], X6[:, 1] = Rx, Ry
        X6[:, 2], X6[:, 3] = Vx, Vy
        X6[:, 4] = np.log(zeta)
        X6[:, 5] = np.log(np.maximum(gamma, 1e-4))
        Y = self.accel_feat(make_features(X6))
        return Y[:, 0], Y[:, 1]


def rollout(predictor, s0, zeta, gamma, n_steps, dtau=DTAU):
    """用学到的向量场做 RK4 自回归，s0: (B,4)。返回 (n_steps+1, B, 4)。"""
    B = s0.shape[0]
    out = np.empty((n_steps + 1, B, 4), dtype=np.float32)
    out[0] = s0
    Rx, Ry, Vx, Vy = (s0[:, 0].copy(), s0[:, 1].copy(), s0[:, 2].copy(), s0[:, 3].copy())

    def f(rx, ry, vx, vy):
        return predictor.accel(rx, ry, vx, vy, zeta, gamma)

    for i in range(n_steps):
        a1x, a1y = f(Rx, Ry, Vx, Vy)
        a2x, a2y = f(Rx + Vx * dtau / 2, Ry + Vy * dtau / 2,
                     Vx + a1x * dtau / 2, Vy + a1y * dtau / 2)
        a3x, a3y = f(Rx + (Vx + a1x * dtau / 2) * dtau / 2,
                     Ry + (Vy + a1y * dtau / 2) * dtau / 2,
                     Vx + a2x * dtau / 2, Vy + a2y * dtau / 2)
        a4x, a4y = f(Rx + (Vx + a2x * dtau / 2) * dtau,
                     Ry + (Vy + a2y * dtau / 2) * dtau,
                     Vx + a3x * dtau, Vy + a3y * dtau)
        Rx = Rx + dtau / 6 * (Vx + 2 * (Vx + a1x * dtau / 2) + 2 * (Vx + a2x * dtau / 2) + (Vx + a3x * dtau))
        Ry = Ry + dtau / 6 * (Vy + 2 * (Vy + a1y * dtau / 2) + 2 * (Vy + a2y * dtau / 2) + (Vy + a3y * dtau))
        Vx = Vx + dtau / 6 * (a1x + 2 * a2x + 2 * a3x + a4x)
        Vy = Vy + dtau / 6 * (a1y + 2 * a2y + 2 * a3y + a4y)
        out[i + 1, :, 0], out[i + 1, :, 1] = Rx, Ry
        out[i + 1, :, 2], out[i + 1, :, 3] = Vx, Vy
    return out


# ---------------------------------------------------------------- 训练

def train():
    states, zeta, gamma = load_data(reuse="--reuse" in sys.argv)
    n = states.shape[1]

    idx = np.random.default_rng(7).permutation(n)
    n_tr, n_va = int(n * 0.8), int(n * 0.9)
    tr, va, te = idx[:n_tr], idx[n_tr:n_va], idx[n_va:]
    print(f"  轨迹划分 train/val/test = {len(tr)}/{len(va)}/{len(te)}")

    Xtr6, Ytr = make_training_pairs(states[:, tr], zeta[tr], gamma[tr], stride=3)
    Xva6, Yva = make_training_pairs(states[:, va], zeta[va], gamma[va])
    print(f"  训练样本 {Xtr6.shape[0]:,}，验证样本 {Xva6.shape[0]:,}")

    Xtr, Xva = make_features(Xtr6), make_features(Xva6)
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8
    ym, ys = Ytr.mean(0), Ytr.std(0) + 1e-8
    print(f"  输出量纲 √ 均值 {ym.round(4)}  标准差 {ys.round(4)}")

    Ntr = (Xtr - xm) / xs
    Ttr = (Ytr - ym) / ys
    Nva = (Xva - xm) / xs
    Tva = (Yva - ym) / ys

    rng = np.random.default_rng(SEED)
    net = MLP([Xtr.shape[1], *HIDDEN, 2], rng)
    print(f"  参数量 {net.n_params():,}，结构 {Xtr.shape[1]}-{'-'.join(map(str, HIDDEN))}-2")

    params = net.W + net.b
    opt = Adam(params, lr=LR, wd=1e-6)
    best, best_state = 1e9, None
    t0 = time.time()

    for step in range(STEPS + 1):
        opt.lr = LR * (0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * step / STEPS)))
        b = rng.integers(0, Ntr.shape[0], BATCH)
        pred = net.forward(Ntr[b], keep=True)
        d = (pred - Ttr[b]) * (2.0 / pred.size)
        gW, gb = net.backward(d)
        opt.step(params, gW + gb)

        if step % EVAL_EVERY == 0:
            vl = float(np.mean((net.forward(Nva) - Tva) ** 2))
            tl = float(np.mean((net.forward(Ntr[:20000]) - Ttr[:20000]) ** 2))
            print(f"  step {step:6d}/{STEPS}  train {tl:.2e}  val {vl:.2e}  "
                  f"lr {opt.lr:.1e}  {time.time() - t0:.0f}s")
            if vl < best:
                best = vl
                best_state = ([W.copy() for W in net.W], [b.copy() for b in net.b])

    net.W, net.b = best_state
    print(f"  最优验证 MSE {best:.2e}")
    return net, (xm, xs, ym, ys), (states, zeta, gamma, te)


# ---------------------------------------------------------------- 评估

def evaluate(net, norm, data):
    states, zeta, gamma, te = data
    pred = Predictor(net, *norm)

    s0 = states[0, te].astype(np.float32)
    zt, gt = zeta[te].astype(np.float32), gamma[te].astype(np.float32)
    gt_traj = states[:, te]

    t0 = time.time()
    pr = rollout(pred, s0, zt, gt, N_STEPS)
    print(f"  rollout {len(te)} 条 × {N_STEPS} 步，用时 {time.time() - t0:.1f}s")

    err = np.linalg.norm(pr[..., :2] - gt_traj[..., :2], axis=-1)      # (T, B)，单位 L0
    taus = np.arange(N_STEPS + 1) * DTAU

    def period(sig):
        d = sig - sig.mean()
        cross = np.where((d[:-1] < 0) & (d[1:] >= 0))[0]
        if len(cross) < 3:
            return np.nan
        return 2 * DTAU * (cross[-1] - cross[0]) / (len(cross) - 1)

    per_t = np.array([period(gt_traj[:, i, 1]) for i in range(len(te))])
    per_p = np.array([period(pr[:, i, 1]) for i in range(len(te))])
    ok = np.isfinite(per_t) & np.isfinite(per_p) & (per_t > 0)
    freq_err = np.median(np.abs(per_p[ok] - per_t[ok]) / per_t[ok]) if ok.any() else float("nan")

    Xte6, Yte = make_training_pairs(states[:, te], zeta[te], gamma[te], stride=5)
    one_step = float(np.sqrt(np.mean((pred.accel_feat(make_features(Xte6)) - Yte) ** 2)))
    res = {"one_step_rmse": one_step}

    print("\n  多步自回归误差（单位 L0）：")
    for tau in (1, 2, 4, 8, 16, 32, 40):
        i = min(int(tau / DTAU), N_STEPS)
        print(f"    tau={tau:2d}  ({tau / 6.283:.1f} 周期)  mean {err[i].mean():.4f}  "
              f"p90 {np.percentile(err[i], 90):.4f}")

    res["fde"] = float(err[-1].mean())
    res["ade"] = float(err.mean())
    res["period_err_median"] = float(freq_err)

    for lv in VPT_LEVELS:
        vpt = np.argmax(err > lv, axis=0)
        never = np.all(err <= lv, axis=0)
        vpt = np.where(never, N_STEPS, vpt) * DTAU
        key = f"vpt_{int(lv * 100)}"
        res[key] = float(np.median(vpt))
        print(f"\n  VPT@{lv:.2f}L0 中位数 {np.median(vpt):.1f} tau "
              f"≈ {np.median(vpt) / 6.283:.1f} 个周期   (10% 分位 {np.percentile(vpt, 10):.1f})")

    print(f"\n  单步加速度 RMSE  {res['one_step_rmse']:.3e}")
    print(f"  ADE（全时域）    {res['ade']:.4f} L0")
    print(f"  FDE（tau={TAU_END:g}）     {res['fde']:.4f} L0")
    print(f"  周期相对误差     {res['period_err_median'] * 100:.2f}%")
    return res


# ---------------------------------------------------------------- 导出

def export(net, norm, metrics):
    xm, xs, ym, ys = norm
    layers = []
    for W, b in zip(net.W, net.b):
        layers.append({
            "in": int(W.shape[0]), "out": int(W.shape[1]),
            "W": [[float(f"{v:.7g}") for v in row] for row in W.astype(np.float64)],
            "b": [float(f"{v:.7g}") for v in b.astype(np.float64)],
        })
    doc = {
        "format": "spring-mlp-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "description": "弹簧振子无量纲向量场；输入特征见 features，输出无量纲加速度",
        "features": FEATURE_NAMES,
        "outputs": ["aRx", "aRy"],
        "arch": {"hidden": HIDDEN, "activation": "silu"},
        "integrator": {"type": "rk4", "dtau": DTAU},
        "norm": {"xMean": [float(v) for v in xm], "xStd": [float(v) for v in xs],
                 "yMean": [float(v) for v in ym], "yStd": [float(v) for v in ys]},
        "layers": layers,
        "validity": {"minRadius": MIN_RADIUS, "maxRadius": 6.0,
                     "zeta": list(ZETA_RANGE), "gamma": [0.0, GAMMA_RANGE[1]],
                     "note": "|R| 或参数超出上述范围属于外推，预测不可信"},
        "metrics": metrics,
    }
    pj = os.path.join(OUT_DIR, "spring_model.json")
    pjs = os.path.join(OUT_DIR, "spring_model.js")
    with open(pj, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
    with open(pjs, "w", encoding="utf-8") as f:
        f.write("/* 由 spring_dl/train.py 自动生成，勿手改。格式见 README-轨迹预测.md */\n")
        f.write("window.SPRING_MODEL = ")
        json.dump(doc, f, ensure_ascii=False)
        f.write(";\n")
    print(f"\n  已导出 {pj} ({os.path.getsize(pj) / 1024:.0f} KB)")
    print(f"  已导出 {pjs} ({os.path.getsize(pjs) / 1024:.0f} KB)")


if __name__ == "__main__":
    t = time.time()
    net, norm, data = train()
    metrics = evaluate(net, norm, data)
    export(net, norm, metrics)
    print(f"\n总用时 {time.time() - t:.0f}s")
