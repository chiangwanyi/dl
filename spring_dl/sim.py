"""与 `弹簧小球-vue3-物理增强版.html` 一致的弹簧振子仿真器。

本版本刻意去掉了两个分支：
  - 空气阻力（cfg.air 恒为 False）
  - 地面碰撞（cfg.floor 恒为 False）

只保留弹簧弹力 + 线性粘滞阻尼 + 重力，因此动力学处处光滑，
可以放心交给神经网络去学。

两种坐标系统都在这里：
  - `*_px` 系列：像素单位，与 HTML 内部实现逐位对齐，用于交叉校验；
  - `*_si` 系列：SI 单位（米、秒），用于生成训练数据。
"""

import numpy as np

SCALE = 0.01     # 1 px = 0.01 m，与 HTML 的 SCALE 一致
H_STEP = 1.0 / 240.0


# --------------------------------------------------------------------------
# 像素单位：逐位复刻 HTML 里的 deriv / rk4
# --------------------------------------------------------------------------

def deriv_px(state, ax0, ay0, k, m, c, g, L0):
    """state = (x, y, vx, vy)，全部像素单位；返回 (vx, vy, ax, ay)。"""
    x, y, vx, vy = state
    dx, dy = x - ax0, y - ay0
    length = max(np.hypot(dx, dy), 1e-9)
    ex, ey = dx / length, dy / length
    vxm, vym = vx * SCALE, vy * SCALE
    fs = -k * (length * SCALE - L0)          # 弹力标量，N
    fx = fs * ex - c * vxm
    fy = fs * ey - c * vym + m * g
    return vx, vy, fx / m / SCALE, fy / m / SCALE


def rk4_px(state, h, ax0, ay0, k, m, c, g, L0):
    k1 = deriv_px(state, ax0, ay0, k, m, c, g, L0)
    s2 = tuple(state[i] + k1[i] * h / 2 for i in range(4))
    k2 = deriv_px(s2, ax0, ay0, k, m, c, g, L0)
    s3 = tuple(state[i] + k2[i] * h / 2 for i in range(4))
    k3 = deriv_px(s3, ax0, ay0, k, m, c, g, L0)
    s4 = tuple(state[i] + k3[i] * h for i in range(4))
    k4 = deriv_px(s4, ax0, ay0, k, m, c, g, L0)
    return tuple(state[i] + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
                 for i in range(4))


# --------------------------------------------------------------------------
# SI 单位：批量生成用
# --------------------------------------------------------------------------

def deriv_si(rx, ry, vx, vy, k, m, c, g, L0):
    """全部 SI 单位，支持 numpy 广播。返回 (ax, ay)。"""
    L = np.maximum(np.hypot(rx, ry), 1e-9)
    ex, ey = rx / L, ry / L
    fs = -k * (L - L0)
    ax = (fs * ex - c * vx) / m
    ay = (fs * ey - c * vy + m * g) / m
    return ax, ay


def rk4_si(rx, ry, vx, vy, h, k, m, c, g, L0):
    a1x, a1y = deriv_si(rx, ry, vx, vy, k, m, c, g, L0)
    a2x, a2y = deriv_si(rx + vx * h / 2, ry + vy * h / 2,
                        vx + a1x * h / 2, vy + a1y * h / 2, k, m, c, g, L0)
    a3x, a3y = deriv_si(rx + (vx + a1x * h / 2) * h / 2,
                        ry + (vy + a1y * h / 2) * h / 2,
                        vx + a2x * h / 2, vy + a2y * h / 2, k, m, c, g, L0)
    a4x, a4y = deriv_si(rx + (vx + a2x * h / 2) * h,
                        ry + (vy + a2y * h / 2) * h,
                        vx + a3x * h, vy + a3y * h, k, m, c, g, L0)
    nrx = rx + h / 6 * (vx + 2 * (vx + a1x * h / 2) + 2 * (vx + a2x * h / 2) + (vx + a3x * h))
    nry = ry + h / 6 * (vy + 2 * (vy + a1y * h / 2) + 2 * (vy + a2y * h / 2) + (vy + a3y * h))
    nvx = vx + h / 6 * (a1x + 2 * a2x + 2 * a3x + a4x)
    nvy = vy + h / 6 * (a1y + 2 * a2y + 2 * a3y + a4y)
    return nrx, nry, nvx, nvy


# --------------------------------------------------------------------------
# 无量纲化：整套动力学只剩两个参数
#   R = r / L0,  V = v / (L0*w0),  tau = w0*t,  w0 = sqrt(k/m)
#   zeta = c / (2*sqrt(k*m)),      gamma = m*g / (k*L0)
#   d2R/dtau2 = -(1 - 1/|R|) * R_hat  -  2*zeta*V  +  gamma*y_hat
# --------------------------------------------------------------------------

def deriv_nondim(Rx, Ry, Vx, Vy, zeta, gamma):
    """无量纲向量场，支持 numpy 广播。返回 (aRx, aRy)。

    d2R/dtau2 = -(|R| - 1) * R_hat  -  2*zeta*V  +  gamma*y_hat
             = -R + R_hat          -  2*zeta*V  +  gamma*y_hat
    第二个等号说明：这是一个线性振子加上一个"大小恒定、指向悬点"的力。
    注意别写成 -(1 - 1/|R|)，那等于把弹力整体除以了 |R|。
    """
    L = np.maximum(np.hypot(Rx, Ry), 1e-9)
    ex, ey = Rx / L, Ry / L
    axial = -(L - 1.0)
    aRx = axial * ex - 2.0 * zeta * Vx
    aRy = axial * ey - 2.0 * zeta * Vy + gamma
    return aRx, aRy


def rk4_nondim(Rx, Ry, Vx, Vy, h, zeta, gamma):
    a1x, a1y = deriv_nondim(Rx, Ry, Vx, Vy, zeta, gamma)
    a2x, a2y = deriv_nondim(Rx + Vx * h / 2, Ry + Vy * h / 2,
                            Vx + a1x * h / 2, Vy + a1y * h / 2, zeta, gamma)
    a3x, a3y = deriv_nondim(Rx + (Vx + a1x * h / 2) * h / 2,
                            Ry + (Vy + a1y * h / 2) * h / 2,
                            Vx + a2x * h / 2, Vy + a2y * h / 2, zeta, gamma)
    a4x, a4y = deriv_nondim(Rx + (Vx + a2x * h / 2) * h,
                            Ry + (Vy + a2y * h / 2) * h,
                            Vx + a3x * h, Vy + a3y * h, zeta, gamma)
    nRx = Rx + h / 6 * (Vx + 2 * (Vx + a1x * h / 2) + 2 * (Vx + a2x * h / 2) + (Vx + a3x * h))
    nRy = Ry + h / 6 * (Vy + 2 * (Vy + a1y * h / 2) + 2 * (Vy + a2y * h / 2) + (Vy + a3y * h))
    nVx = Vx + h / 6 * (a1x + 2 * a2x + 2 * a3x + a4x)
    nVy = Vy + h / 6 * (a1y + 2 * a2y + 2 * a3y + a4y)
    return nRx, nRy, nVx, nVy


def to_nondim(rx, ry, vx, vy, k, m, c, g, L0):
    """SI -> 无量纲。"""
    w0 = np.sqrt(k / m)
    zeta = c / (2.0 * np.sqrt(k * m))
    gamma = m * g / (k * L0)
    return rx / L0, ry / L0, vx / (L0 * w0), vy / (L0 * w0), zeta, gamma


def from_nondim(Rx, Ry, Vx, Vy, k, m, L0):
    """无量纲 -> SI（不含参数）。"""
    w0 = np.sqrt(k / m)
    return Rx * L0, Ry * L0, Vx * L0 * w0, Vy * L0 * w0
