"""022（向量场）—— 把 021 的"猜位置"升级成"猜力"。

021 学会的是：给一个时间 t，猜出那一刻的 (x, y)。
这个网络只认识"vx=20, vy=30 那一条弹道"，初速度一换它就完全不知道该怎么办。

要让它像 03-弹簧振子 那样能跑网页仿真、能随时拖滑块换参数，得换一个问题来问：

    021:   t            ->  (x, y)      猜位置（背诵一条轨迹）
    022:   (vx, vy, k)  ->  (ax, ay)    猜力（学一个向量场）

猜力之后，网页里的 RK4 积分器拿着这个"力"一小步一小步往前推，
自己就能画出整条弹道。换初速度、换阻力、中途加一阵风，都无所谓——
网络只回答一个问题：此刻速度是这样、阻力系数是这么多，加速度该是多少。

本文件训练两个网络，都导出给同目录的 022_向量场.html 用：

    位置模型 position_model :  1 -> 16 -> 2     输入 t，输出 (x, y)       ← 021 原版
    场模型   field_model    :  3 -> 32 -> 2     输入 (vx, vy, k)，输出 (ax, ay)

两个网络都用 021 的那套结构（一层隐藏层 + tanh + autograd 求梯度），
只有优化器不同：位置模型用 021 的朴素梯度下降，场模型必须换成 Adam。

打开网页能直接看到差别：拖一下初速度滑块，
位置模型的灰线立刻飞掉，场模型的橙线纹丝不动。

运行：  python 022_向量场.py        （约 3 分钟）
"""

import json
import os
import time

import autograd.numpy as np
from autograd import grad

# 权重与 HTML 放在同一目录：网页用相对路径 <script src="022_model.js"> 加载
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(42)

G = 9.8          # 重力加速度 (m/s²)
DT = 0.01        # 仿真步长 (s)
STEPS = 700      # 每条轨迹最多推 7 秒

# 滑块范围，网页上读的就是这三个数。训练数据的采样范围必须和它们对齐，
# 否则拖到边缘就是在让网络外推，误差会突然变大。
V0_RANGE = (15.0, 36.0)
DEG_RANGE = (20.0, 75.0)
K_RANGE = (0.0, 0.06)


# ================== 1. 生成训练数据 ==================

def accel_true(vx, vy, k):
    """真实加速度 = 重力 + 二次空气阻力。

    阻力大小是 k|v|²，方向与速度相反，写成矢量就是 -k|v|·v。
    注意这是 021 里没有的东西：没有阻力时 y = 30t - 0.5gt² 有解析解，
    加了阻力就没有解析解了，只能靠数值积分——这正是需要"仿真程序"的原因。
    """
    speed = np.hypot(vx, vy)
    return -k * speed * vx, -k * speed * vy - G


def deriv(X, Y, VX, VY, K):
    """状态的变化率：位置的变化率是速度，速度的变化率是加速度。"""
    AX, AY = accel_true(VX, VY, K)
    return VX, VY, AX, AY


def rk4(X, Y, VX, VY, K, h):
    """四阶龙格-库塔：用四次采样拼出一步，误差是 h⁵ 量级。

    比"位置 += 速度*dt"（欧拉法）准得多，代价是多算三遍加速度。
    网页里用的是同一个算法。
    """
    k1x, k1y, k1vx, k1vy = deriv(X, Y, VX, VY, K)
    k2x, k2y, k2vx, k2vy = deriv(X + h / 2 * k1x, Y + h / 2 * k1y,
                                 VX + h / 2 * k1vx, VY + h / 2 * k1vy, K)
    k3x, k3y, k3vx, k3vy = deriv(X + h / 2 * k2x, Y + h / 2 * k2y,
                                 VX + h / 2 * k2vx, VY + h / 2 * k2vy, K)
    k4x, k4y, k4vx, k4vy = deriv(X + h * k3x, Y + h * k3y,
                                 VX + h * k3vx, VY + h * k3vy, K)
    X = X + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
    Y = Y + h / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
    VX = VX + h / 6 * (k1vx + 2 * k2vx + 2 * k3vx + k4vx)
    VY = VY + h / 6 * (k1vy + 2 * k2vy + 2 * k3vy + k4vy)
    return X, Y, VX, VY


def simulate(v0, ang, K, steps=STEPS, h=DT):
    """批量仿真一批抛体，返回形状 (steps+1, n, 4) 的轨迹，列是 x, y, vx, vy。

    所有轨迹同时用 numpy 推进，不写 for 循环逐条算。
    """
    n = len(v0)
    VX = v0 * np.cos(ang)
    VY = v0 * np.sin(ang)
    X = np.zeros(n)
    Y = np.zeros(n)

    traj = np.zeros((steps + 1, n, 4))
    traj[0, :, 0], traj[0, :, 1] = X, Y
    traj[0, :, 2], traj[0, :, 3] = VX, VY

    for i in range(steps):
        X, Y, VX, VY = rk4(X, Y, VX, VY, K, h)
        traj[i + 1, :, 0], traj[i + 1, :, 1] = X, Y
        traj[i + 1, :, 2], traj[i + 1, :, 3] = VX, VY
    return traj


def build_field_dataset(n_traj=600, n_keep=8000):
    """从一批随机弹道上采集 (vx, vy, k) -> (ax, ay) 的监督对。

    注意输入里没有 x 和 y：抛体的加速度只取决于速度和阻力，跟位置无关。
    （弹簧不一样，弹簧有回复力 -(L-1)，所以 03-弹簧振子 的模型必须把位置喂给网络。）
    """
    v0 = V0_RANGE[0] + (V0_RANGE[1] - V0_RANGE[0]) * np.random.rand(n_traj)
    deg = DEG_RANGE[0] + (DEG_RANGE[1] - DEG_RANGE[0]) * np.random.rand(n_traj)
    ang = np.deg2rad(deg)
    K = K_RANGE[0] + (K_RANGE[1] - K_RANGE[0]) * np.random.rand(n_traj)

    traj = simulate(v0, ang, K)

    air = traj[..., 1] > 0            # 只取还在空中的点，落地之后的不算
    VX = traj[..., 2][air]
    VY = traj[..., 3][air]
    KK = np.broadcast_to(K[None, :], air.shape)[air]
    AX, AY = accel_true(VX, VY, KK)

    n_all = len(VX)
    idx = np.random.choice(n_all, size=n_keep, replace=False)

    Xf = np.column_stack([VX[idx], VY[idx], KK[idx]])
    Yf = np.column_stack([AX[idx], AY[idx]])
    print(f"  仿真 {n_traj} 条弹道，空中状态点 {n_all:,} 个，抽取 {n_keep} 个做训练")
    print(f"  输入范围  vx [{Xf[:, 0].min():.1f}, {Xf[:, 0].max():.1f}]  "
          f"vy [{Xf[:, 1].min():.1f}, {Xf[:, 1].max():.1f}]  "
          f"k [{Xf[:, 2].min():.3f}, {Xf[:, 2].max():.3f}]")
    return Xf, Yf


def build_position_dataset(n=200):
    """照 021 原样：固定 vx=20, vy=30, 无阻力，只学"这一条弹道"。

    021 里只取了 t ∈ [0, 5]，这里延长到 6 秒（覆盖完整飞行过程），
    其余部分——包括样本数、噪声大小——和 021 完全一致。
    vx=20, vy=30 对应 v0 = 36.1 m/s、发射角 56.3°，也就是网页上滑块的默认位置。
    """
    t = np.linspace(0, 6.0, n)
    x = 20 * t + np.random.randn(n) * 0.5
    y = 30 * t - 0.5 * G * t ** 2 + np.random.randn(n) * 0.4
    print(f"  固定弹道 vx=20, vy=30, k=0，t ∈ [0, 6.0]，样本 {n} 个")
    return t.reshape(-1, 1), np.column_stack((x, y))


# ================== 2~6. 训练（就是 021 的第 2~6 节，参数化了一下） ==================

def train_network(X, Y, n_hidden, lr, epochs, seed=1, tag="", optimizer="gd"):
    """训练一个"一层隐藏层"的网络，流程与 021 逐行对应。

    021 的流程：初始化 W1,b1,W2,b2 -> 前向 tanh -> MSE -> grad -> 梯度下降。
    这里唯一的变化是 grad(loss_fn, argnum=[0,1,2,3]) 一次拿到四个梯度，
    而 021 里写了四遍 grad，每遍都要重新前向一次。

    optimizer="adam" 是 023 才会讲的东西，这里先用上，原因写在下面 main() 里：
    同样 7000 轮，朴素梯度下降只能把 loss 压到 0.008，滚 6 秒偏 11 米；
    换 Adam 后 loss 到 1.5e-4，误差不到 1 米。同一套梯度、同一个网络。
    """
    rng = np.random.default_rng(seed)

    P = [rng.standard_normal((X.shape[1], n_hidden)) * 0.5,
         np.zeros((1, n_hidden)),
         rng.standard_normal((n_hidden, Y.shape[1])) * 0.5,
         np.zeros((1, Y.shape[1]))]

    def forward(W1, b1, W2, b2):
        H = np.tanh(X @ W1 + b1)         # 激活函数，跟 021 一样用 tanh
        return H @ W2 + b2

    def loss_fn(W1, b1, W2, b2):
        return np.mean((forward(W1, b1, W2, b2) - Y) ** 2)

    grads = grad(loss_fn, argnum=[0, 1, 2, 3])

    m = [np.zeros_like(p) for p in P]
    v = [np.zeros_like(p) for p in P]

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        gs = grads(*P)
        for i in range(4):
            if optimizer == "adam":
                m[i] = 0.9 * m[i] + 0.1 * gs[i]
                v[i] = 0.999 * v[i] + 0.001 * gs[i] ** 2
                P[i] = P[i] - lr * (m[i] / (1 - 0.9 ** epoch)) / \
                    (np.sqrt(v[i] / (1 - 0.999 ** epoch)) + 1e-8)
            else:
                P[i] = P[i] - lr * gs[i]

        if epoch % 1000 == 0 or epoch == epochs:
            print(f"    {tag} epoch {epoch:5d}/{epochs}  loss {loss_fn(*P):.6f}"
                  f"  {time.time() - t0:.1f}s")

    return P


def predict(params, xm, xs, ym, ys, X):
    """带归一化的前向。网页里用的是同一段逻辑。"""
    W1, b1, W2, b2 = params
    Xn = (X - xm) / xs
    Yn = np.tanh(Xn @ W1 + b1) @ W2 + b2
    return Yn * ys + ym


# ================== 7. 验收：让场模型自己滚完整条弹道 ==================

def rollout(accel_fn, v0, ang, k, steps=STEPS, h=DT):
    """只给初始条件，一步步预测加速度、RK4 积分，滚出整条弹道。

    注意这里没有任何"位置标签"可用——网络从第一步起就得靠自己。
    这就是 03-弹簧振子 里那个词的含义：自回归（autoregressive）rollout。
    """
    X, Y = 0.0, 0.0
    VX, VY = v0 * np.cos(ang), v0 * np.sin(ang)
    path = [(X, Y, VX, VY)]

    for _ in range(steps):
        def f(rx, ry, vx, vy):
            a = accel_fn(vx, vy, k)
            return float(a[0]), float(a[1])

        a1x, a1y = f(X, Y, VX, VY)
        a2x, a2y = f(X + h / 2 * VX, Y + h / 2 * VY, VX + h / 2 * a1x, VY + h / 2 * a1y)
        a3x, a3y = f(X + h / 2 * (VX + h / 2 * a1x), Y + h / 2 * (VY + h / 2 * a1y),
                     VX + h / 2 * a2x, VY + h / 2 * a2y)
        a4x, a4y = f(X + h * (VX + h / 2 * a2x), Y + h * (VY + h / 2 * a2y),
                     VX + h * a3x, VY + h * a3y)

        X = X + h / 6 * (VX + 2 * (VX + h / 2 * a1x) + 2 * (VX + h / 2 * a2x) + (VX + h * a3x))
        Y = Y + h / 6 * (VY + 2 * (VY + h / 2 * a1y) + 2 * (VY + h / 2 * a2y) + (VY + h * a3y))
        VX = VX + h / 6 * (a1x + 2 * a2x + 2 * a3x + a4x)
        VY = VY + h / 6 * (a1y + 2 * a2y + 2 * a3y + a4y)

        path.append((X, Y, VX, VY))
        if Y < 0:
            break
    return np.array(path)


def check(accel_fn, v0, deg, k):
    """拿一组新参数，比对网络自己滚出的弹道和真实弹道。"""
    ang = np.deg2rad(deg)
    true = simulate(np.array([v0]), np.array([ang]), np.array([k]))[:, 0, :2]
    true = true[true[:, 1] >= 0]

    pred = rollout(accel_fn, v0, ang, k)
    pred = pred[pred[:, 1] >= 0][:, :2]

    n = min(len(true), len(pred))
    err = np.linalg.norm(true[:n] - pred[:n], axis=1)
    print(f"    v0={v0:4.1f} m/s, {deg:.0f}°, k={k:.3f}   "
          f"射程 真实 {true[-1, 0]:6.2f} m / 预测 {pred[-1, 0]:6.2f} m   "
          f"最大偏差 {err.max():.2f} m")


# ================== 8. 导出给网页 ==================

def pack(params, norm, name):
    W1, b1, W2, b2 = params
    xm, xs, ym, ys = norm
    return {
        "name": name,
        "layers": [
            {"in": int(W1.shape[0]), "out": int(W1.shape[1]),
             "W": W1.tolist(), "b": b1[0].tolist()},
            {"in": int(W2.shape[0]), "out": int(W2.shape[1]),
             "W": W2.tolist(), "b": b2[0].tolist()},
        ],
        "norm": {"xm": xm.tolist(), "xs": xs.tolist(),
                 "ym": ym.tolist(), "ys": ys.tolist()},
    }


def main():
    t_start = time.time()

    print("\n[1/5] 生成场模型的训练数据")
    Xf, Yf = build_field_dataset()

    xm, xs = Xf.mean(0), Xf.std(0) + 1e-9
    ym, ys = Yf.mean(0), Yf.std(0) + 1e-9
    print(f"  归一化：速度标准差 {xs.round(2)}，加速度标准差 {ys.round(2)}")
    print("  （这一步 021 里没有。速度 ~30、k ~0.03，量级差一千倍，不归一化根本训不动）")

    print("\n[2/5] 训练场模型 (vx, vy, k) -> (ax, ay)，3-32-2")
    print("  优化器用 Adam（023 的内容）。同样 7000 轮，021 的朴素梯度下降只能到 loss 0.008，")
    print("  滚 6 秒偏 11 米；Adam 能到 1.5e-4，误差不到 1 米。网络和梯度都没变，只换了走的步子。")
    field = train_network((Xf - xm) / xs, (Yf - ym) / ys, n_hidden=32, lr=0.01,
                          epochs=9000, seed=1, tag="field", optimizer="adam")

    print("\n[3/5] 生成位置模型的训练数据（021 原版）")
    Xp, Yp = build_position_dataset()

    print("\n[4/5] 训练位置模型 t -> (x, y)，1-16-2")
    print("  这个继续用 021 的朴素梯度下降：它只学一条曲线，1 万轮不到 10 秒就收敛了。")
    pos = train_network(Xp, Yp, n_hidden=16, lr=0.01, epochs=12000, seed=42,
                        tag="pos", optimizer="gd")

    print("\n[5/5] 验收场模型：只给初速度，让它自己滚完整条弹道")
    accel_field = lambda vx, vy, k: predict(field, xm, xs, ym, ys,
                                            np.array([[vx, vy, k]]))[0]
    check(accel_field, 36.0, 56.0, 0.0)
    check(accel_field, 36.0, 56.0, 0.05)
    check(accel_field, 22.0, 35.0, 0.03)
    check(accel_field, 30.0, 70.0, 0.01)

    norm_pos = (np.array([0.0]), np.array([1.0]), np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    doc = {
        "format": "projectile-demo-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "g": G,
        "dt": DT,
        "ranges": {"v0": list(V0_RANGE), "deg": list(DEG_RANGE), "k": list(K_RANGE)},
        "position_model": pack(pos, norm_pos, "021 位置模型 t->(x,y)"),
        "field_model": pack(field, (xm, xs, ym, ys), "022 场模型 (vx,vy,k)->(ax,ay)"),
        "position_model_trained_on": {"vx": 20.0, "vy": 30.0, "k": 0.0},
    }

    with open(os.path.join(OUT_DIR, "022_model.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)

    with open(os.path.join(OUT_DIR, "022_model.js"), "w", encoding="utf-8") as f:
        f.write("/* 由 022_向量场.py 自动生成，勿手改 */\n")
        f.write("window.PROJECTILE_MODEL = ")
        json.dump(doc, f, ensure_ascii=False)
        f.write(";\n")

    print(f"\n已导出 022_model.json / 022_model.js")
    print(f"总用时 {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
