"""训练数据生成：在无量纲空间做拉丁超立方采样，批量积分。

关键点：无量纲化之后，m 和 L0 的取值对动力学没有任何影响
（它们只决定长度尺度和时间尺度的换算），所以整个数据集只需要在
(zeta, gamma) 参数空间 + 无量纲状态空间里采样，维度从 7 降到 5。
"""

import numpy as np

from sim import deriv_nondim, rk4_nondim

TAU_END = 40.0          # 每条轨迹的无量纲时长，约 6.4 个振荡周期
DTAU = 0.04             # 无量纲步长
N_STEPS = int(round(TAU_END / DTAU))
MIN_RADIUS = 0.18       # |R| 小于此值的轨迹丢弃：弹簧被压到接近实心长度，数值上病态
R_MAX = 6.0             # |R| 超过此值说明积分已经发散（见下方 energy 检查）

ZETA_RANGE = (0.02, 4.0)
GAMMA_RANGE = (0.05, 4.0)


def energy(Rx, Ry, Vx, Vy, gamma):
    """无量纲总机械能。阻尼存在时它必须单调不增，用来筛掉数值发散的轨迹。"""
    L = np.hypot(Rx, Ry)
    return 0.5 * (Vx * Vx + Vy * Vy) + 0.5 * (L - 1.0) ** 2 - gamma * Ry


def lhs(n, dims, rng):
    """拉丁超立方采样，返回 (n, dims) 的 [0,1) 矩阵。"""
    out = np.empty((n, dims))
    for d in range(dims):
        strata = (rng.permutation(n) + rng.random(n)) / n
        out[:, d] = strata[rng.permutation(n)]
    return out


def log_uniform(u, lo, hi):
    return np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))


def sample_params(n, rng):
    """(zeta, gamma) 采样。

    gamma == 0 单独占 15%：无重力分支和 gamma>0 是两个不同的定性行为，
    按对数均匀采会漏掉这个边界。
    """
    u = lhs(n, 2, rng)
    zeta = log_uniform(u[:, 0], *ZETA_RANGE)
    gamma = log_uniform(u[:, 1], *GAMMA_RANGE)
    zero_mask = rng.random(n) < 0.15
    gamma[zero_mask] = 0.0
    return zeta, gamma


def sample_states(n, rng, gamma):
    """无量纲初值。围绕平衡半径 1+gamma 附近取，速度分两种模式。"""
    eq = 1.0 + gamma
    radius = eq * (0.55 + 0.85 * rng.random(n))          # 0.55~1.40 倍平衡半径
    theta = rng.random(n) * 2 * np.pi
    Rx = radius * np.cos(theta)
    Ry = radius * np.sin(theta)

    # 一半给切向速度（接近椭圆轨道），一半给任意方向
    speed = 0.08 + 1.05 * rng.random(n)
    tangential = rng.random(n) < 0.5
    sign = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    dir_theta = np.where(tangential, theta + sign * np.pi / 2, rng.random(n) * 2 * np.pi)
    Vx = speed * np.cos(dir_theta)
    Vy = speed * np.sin(dir_theta)
    return Rx, Ry, Vx, Vy


def generate(n_traj=2000, seed=0, verbose=True):
    """拒绝采样：一直生成到凑够 n_traj 条数值健康的轨迹。

    为什么要拒绝：|R| 穿过 0 附近时 d2R/dtau2 里有 1/|R| 的奇异性，
    定步长 RK4 会在这一步注入巨大能量，轨迹直接飞出（|R| 能到 400+）。
    这种样本一旦混进数据集，会把输入/输出的归一化统计量整个污染掉。
    """
    rng = np.random.default_rng(seed)
    good_states, good_zeta, good_gamma = [], [], []
    n_have = 0
    n_try = 0
    while n_have < n_traj:
        batch = max(64, int((n_traj - n_have) * 1.35))
        n_try += batch
        zeta, gamma = sample_params(batch, rng)
        Rx, Ry, Vx, Vy = sample_states(batch, rng, gamma)

        states = np.empty((N_STEPS + 1, batch, 4))
        states[0, :, 0], states[0, :, 1] = Rx, Ry
        states[0, :, 2], states[0, :, 3] = Vx, Vy

        min_r = np.hypot(Rx, Ry).copy()
        max_r = min_r.copy()
        e_drift = np.zeros(batch)
        e0 = energy(Rx, Ry, Vx, Vy, gamma)
        for i in range(N_STEPS):
            Rx, Ry, Vx, Vy = rk4_nondim(Rx, Ry, Vx, Vy, DTAU, zeta, gamma)
            states[i + 1, :, 0], states[i + 1, :, 1] = Rx, Ry
            states[i + 1, :, 2], states[i + 1, :, 3] = Vx, Vy
            r = np.hypot(Rx, Ry)
            np.minimum(min_r, r, out=min_r)
            np.maximum(max_r, r, out=max_r)
            np.maximum(e_drift, energy(Rx, Ry, Vx, Vy, gamma) - e0, out=e_drift)

        keep = (min_r >= MIN_RADIUS) & (max_r <= R_MAX) & (e_drift < 1e-3) \
               & np.isfinite(states).all(axis=(0, 2))
        good_states.append(states[:, keep, :].astype(np.float32))
        good_zeta.append(zeta[keep].astype(np.float32))
        good_gamma.append(gamma[keep].astype(np.float32))
        n_have += int(keep.sum())

    states = np.concatenate(good_states, axis=1)[:, :n_traj, :]
    zeta = np.concatenate(good_zeta)[:n_traj]
    gamma = np.concatenate(good_gamma)[:n_traj]
    if verbose:
        print(f"  采样 {n_try} 条，保留 {n_traj} 条（拒绝率 {100 * (1 - n_traj / n_try):.0f}%）")
    return states, zeta, gamma


FEATURE_NAMES = ["Rx", "Ry", "Vx", "Vy", "log_zeta", "log_gamma", "L", "inv_L"]


def make_features(X6):
    """在 6 维原始输入上补两列半径特征。

    动力学里显含 |R|（弹力大小是 -(L-1)），把它和 1/|R| 直接给网络，
    相当于把"先学一个距离度量再学径向依赖"这两步合成一步，
    实测能把收敛速度和最终误差都改善一个量级。
    """
    L = np.maximum(np.hypot(X6[:, 0], X6[:, 1]), 1e-6)
    return np.concatenate([X6, L[:, None], (1.0 / L)[:, None]], axis=1).astype(np.float32)


def make_training_pairs(states, zeta, gamma, stride=1, balance=True, keep_rest=0.15):
    """把轨迹展平成 (X, y) 监督对。

    标签直接用解析向量场，不需要差分，也不需要展开计算图：
    我们完全知道真实的加速度，这是仿真数据相对实测数据的最大优势。

    balance=True 会丢掉"已经停稳"的样本。原因：阻尼大的轨迹跑几个周期后
    就贴在平衡点上不动了，这些点 a≈0 占了样本的一半以上，却几乎不含信息；
    留在训练集里会把输出归一化尺度压小，让"一律输出 0"变成低 loss 的解。
    """
    s = states[::stride]
    T, N, _ = s.shape
    Rx, Ry, Vx, Vy = (s[..., 0], s[..., 1], s[..., 2], s[..., 3])
    aRx, aRy = deriv_nondim(Rx, Ry, Vx, Vy, zeta[None, :], gamma[None, :])
    X = np.stack([Rx, Ry, Vx, Vy,
                  np.broadcast_to(np.log(zeta)[None, :], Rx.shape),
                  np.broadcast_to(np.log(np.maximum(gamma, 1e-4))[None, :], Rx.shape)],
                 axis=-1).reshape(-1, 6)
    Y = np.stack([aRx, aRy], axis=-1).reshape(-1, 2)

    if balance:
        eq = 1.0 + gamma[None, :]
        active = (np.hypot(Vx, Vy) > 0.05) | (np.abs(np.hypot(Rx, Ry) - eq) > 0.05)
        active = active.reshape(-1)
        rest_idx = np.flatnonzero(~active)
        rng = np.random.default_rng(0)
        pick = rng.permutation(rest_idx)[:int(len(rest_idx) * keep_rest)]
        keep = np.concatenate([np.flatnonzero(active), pick])
        keep.sort()
        X, Y = X[keep], Y[keep]

    return X.astype(np.float32), Y.astype(np.float32)
