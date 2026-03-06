import os
import time
import math
import json
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import List, Tuple

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    from scipy.optimize import linear_sum_assignment
except Exception as e:
    raise ImportError("需要 scipy：pip install scipy") from e


# =========================================================
# 3D 场景假设与轨迹参数说明（10条无人机轨迹，固定种子可复现）
# =========================================================
# 全局：
#   - 目标数量：N_TARGETS = 10
#   - 仿真步数：T = 90
#   - 采样周期：DT = 1.0 s
#   - 3D状态定义：state = [x, y, z, vx, vy, vz]
#   - 固定随机种子：SEED = 42
#
# 画布铺满策略（让3D图更“好看”）：
#   - 初始位置 (x0,y0,z0) 在较大3D盒子内均匀采样，确保目标分布覆盖整个空间
#   - 初始速度方向随机 + 速度幅度分散；Scene2 增加爬升/下降/转弯/加减速
#   - 杂波在同一3D盒子内均匀产生，让点云铺满画面
#
# ---------------------------------------------------------
# 场景1（Scene1）：10条轨迹全部 3D 直线匀速（CV）
# ---------------------------------------------------------
# 每个无人机 i 的初始参数（固定种子下随机）：
#   - 初始位置：
#       x0_i ~ Uniform(-X0, X0)
#       y0_i ~ Uniform(-Y0, Y0)
#       z0_i ~ Uniform( Zmin, Zmax )     # 高度
#   - 初始速度（3D方向随机）：
#       speed_i ~ Uniform(vmin, vmax)
#       azimuth_i ~ Uniform(0, 2π)       # 水平面方位角
#       elev_i    ~ Uniform(-elev_max, +elev_max)  # 俯仰角，控制上升/下降分量
#       vx0_i = speed_i * cos(elev_i) * cos(azimuth_i)
#       vy0_i = speed_i * cos(elev_i) * sin(azimuth_i)
#       vz0_i = speed_i * sin(elev_i)
# 运动规律：
#   - v保持常量，x/y/z 按 v 积分（严格匀速直线）
#   - 加速度：ax=ay=az=0
#
# ---------------------------------------------------------
# 场景2（Scene2）：3D 分段机动（匀速 + 加/减速 + 水平转弯 + 爬升/下降变化 + 后段扰动）
# ---------------------------------------------------------
# 每个无人机 i 的初始参数同样随机（位置与3D速度方向），但额外生成机动剧本：
#   - 分段时刻：
#       t1_i ~ randint(18, 30)
#       t2_i = t1_i + randint(14, 22)     # 加/减速段
#       t3_i = t2_i + randint(14, 22)     # 转弯段
#       t4_i = t3_i + randint(10, 18)     # 爬升/下降变化段（轻机动）
#   - 加/减速强度（沿速度方向改变速度大小）：
#       accel_mag_i ~ Uniform(-0.45, 0.55)    # 近似离散 a_t (m/s^2)
#       v(k) = clip(v(k-1) + accel_mag_i, v_clip_min, v_clip_max)
#   - 转弯强度（主要在水平面 yaw 变化）：
#       turn_total_i ~ Uniform(-π/1.8, π/1.8)
#       turn_rate_i  = turn_total_i / (t3_i - t2_i)
#       yaw(k) = yaw(k-1) + turn_rate_i
#       速度大小保持 v(k)（来自上一段末尾）
#   - 爬升/下降变化（改变俯仰角 elev）：
#       climb_total_i ~ Uniform(-elev_delta_max, +elev_delta_max)
#       climb_rate_i  = climb_total_i / (t4_i - t3_i)
#       elev(k) = clip(elev(k-1) + climb_rate_i, -elev_max2, +elev_max2)
#   - 后段小扰动：
#       每步以 jitter_prob 概率对 yaw/elev 做小扰动
#
# 参数导出：
#   sim_outputs/Scene1_params.(csv/json)
#   sim_outputs/Scene2_params.(csv/json)
# =========================================================


# =========================
# 全局配置
# =========================
SEED = 42
np.random.seed(SEED)

OUT_DIR = "sim_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

DT = 1.0
T = 90
N_TARGETS = 10

# 画布范围（尽量铺满）
X0, Y0 = 800, 800
ZMIN, ZMAX = 50, 450

# 量测模型（3D位置）
SIGMA_Z = 6.0       # 3D位置测量噪声（m）
PD = 0.92
CLUTTER_LAMBDA = 10
AREA3D = (-900, 900, -900, 900, 0, 600)  # xmin,xmax,ymin,ymax,zmin,zmax

# 门限与代价：chi2(3)
GATE_CHI2 = 11.34   # approx chi2 df=3 at 0.99
MISS_COST = 25.0

# 速度范围
VMIN, VMAX = 10.0, 25.0


# =========================
# 轨迹生成（返回 truth_states + params_list）
# =========================
def gen_scene1_truth(n=N_TARGETS, T=T, dt=DT, seed=SEED):
    rng = np.random.default_rng(seed)
    states = np.zeros((T, n, 6), dtype=float)
    params_list = []

    x0 = rng.uniform(-X0, X0, size=n)
    y0 = rng.uniform(-Y0, Y0, size=n)
    z0 = rng.uniform(ZMIN, ZMAX, size=n)

    speed = rng.uniform(VMIN, VMAX, size=n)
    azim = rng.uniform(0, 2*np.pi, size=n)
    elev_max = np.deg2rad(12)
    elev = rng.uniform(-elev_max, elev_max, size=n)

    vx0 = speed * np.cos(elev) * np.cos(azim)
    vy0 = speed * np.cos(elev) * np.sin(azim)
    vz0 = speed * np.sin(elev)

    states[0, :, :] = np.stack([x0, y0, z0, vx0, vy0, vz0], axis=1)

    for i in range(n):
        params_list.append({
            "tid": i,
            "x0": float(x0[i]),
            "y0": float(y0[i]),
            "z0": float(z0[i]),
            "speed0": float(speed[i]),
            "azimuth0_rad": float(azim[i]),
            "elev0_rad": float(elev[i]),
            "vx0": float(vx0[i]),
            "vy0": float(vy0[i]),
            "vz0": float(vz0[i]),
            "motion": "3D_CV_straight",
            "accel_mag": 0.0,
            "turn_total_rad": 0.0,
            "turn_rate_rad_per_step": 0.0,
            "climb_total_rad": 0.0,
            "climb_rate_rad_per_step": 0.0,
            "t1": None, "t2": None, "t3": None, "t4": None
        })

    for k in range(1, T):
        x, y, z, vx, vy, vz = states[k-1].T
        x = x + vx * dt
        y = y + vy * dt
        z = z + vz * dt
        states[k, :, :] = np.stack([x, y, z, vx, vy, vz], axis=1)

    return states, params_list


def gen_scene2_truth(n=N_TARGETS, T=T, dt=DT, seed=SEED):
    rng = np.random.default_rng(seed + 7)
    states = np.zeros((T, n, 6), dtype=float)
    params_list = []

    # 初始位置更“铺满”
    x0 = rng.uniform(-X0, X0, size=n)
    y0 = rng.uniform(-Y0, Y0, size=n)
    z0 = rng.uniform(ZMIN, ZMAX, size=n)

    speed0 = rng.uniform(VMIN, VMAX, size=n)
    yaw0 = rng.uniform(0, 2*np.pi, size=n)  # 水平航向
    elev_max0 = np.deg2rad(10)
    elev0 = rng.uniform(-elev_max0, elev_max0, size=n)  # 初始俯仰

    vx0 = speed0 * np.cos(elev0) * np.cos(yaw0)
    vy0 = speed0 * np.cos(elev0) * np.sin(yaw0)
    vz0 = speed0 * np.sin(elev0)

    states[0, :, :] = np.stack([x0, y0, z0, vx0, vy0, vz0], axis=1)

    # 为每个目标生成机动脚本
    scripts = []
    for i in range(n):
        t1 = int(rng.integers(18, 30))
        t2 = int(t1 + rng.integers(14, 22))
        t3 = int(t2 + rng.integers(14, 22))
        t4 = int(min(T-1, t3 + rng.integers(10, 18)))

        accel_mag = float(rng.uniform(-0.45, 0.55))
        turn_total = float(rng.uniform(-np.pi/1.8, np.pi/1.8))
        turn_rate = float(turn_total / max(1, (t3 - t2)))

        elev_delta_max = np.deg2rad(10)
        climb_total = float(rng.uniform(-elev_delta_max, elev_delta_max))
        climb_rate = float(climb_total / max(1, (t4 - t3)))

        scripts.append((t1, t2, t3, t4, accel_mag, turn_total, turn_rate, climb_total, climb_rate))

        params_list.append({
            "tid": i,
            "x0": float(x0[i]),
            "y0": float(y0[i]),
            "z0": float(z0[i]),
            "speed0": float(speed0[i]),
            "yaw0_rad": float(yaw0[i]),
            "elev0_rad": float(elev0[i]),
            "vx0": float(vx0[i]),
            "vy0": float(vy0[i]),
            "vz0": float(vz0[i]),
            "motion": "3D_piecewise(CV + accel/decel + yaw_turn + climb + jitter)",
            "t1": t1, "t2": t2, "t3": t3, "t4": t4,
            "accel_mag": accel_mag,
            "turn_total_rad": turn_total,
            "turn_rate_rad_per_step": turn_rate,
            "climb_total_rad": climb_total,
            "climb_rate_rad_per_step": climb_rate,
            "jitter_prob": 0.06,
            "jitter_yaw_rad_range": [-0.10, 0.10],
            "jitter_elev_rad_range": [-0.06, 0.06],
            "speed_clip": [6.0, 30.0],
            "elev_clip_rad": [-np.deg2rad(18), np.deg2rad(18)]
        })

    # 逐目标推进
    v_clip_min, v_clip_max = 6.0, 30.0
    elev_clip_min, elev_clip_max = -np.deg2rad(18), np.deg2rad(18)

    for i in range(n):
        t1, t2, t3, t4, accel_mag, turn_total, turn_rate, climb_total, climb_rate = scripts[i]

        # 用 yaw/elev + speed 来更新 vx/vy/vz，便于清晰定义“转弯/爬升”
        yaw = yaw0[i]
        elev = elev0[i]
        v = speed0[i]

        for k in range(1, T):
            x, y, z, vx, vy, vz = states[k-1, i]

            if k < t1:
                # 段1：匀速（yaw/elev不变）
                pass
            elif k < t2:
                # 段2：加/减速（沿当前方向）
                v = np.clip(v + accel_mag, v_clip_min, v_clip_max)
            elif k < t3:
                # 段3：水平转弯（yaw逐步变化）
                yaw = yaw + turn_rate
            elif k < t4:
                # 段4：爬升/下降变化（elev逐步变化）
                elev = np.clip(elev + climb_rate, elev_clip_min, elev_clip_max)
            else:
                # 后段：小扰动
                if rng.random() < 0.06:
                    yaw = yaw + rng.uniform(-0.10, 0.10)
                    elev = np.clip(elev + rng.uniform(-0.06, 0.06), elev_clip_min, elev_clip_max)

            # 根据 (v, yaw, elev) 生成速度分量
            vx = v * np.cos(elev) * np.cos(yaw)
            vy = v * np.cos(elev) * np.sin(yaw)
            vz = v * np.sin(elev)

            # 积分更新位置
            x = x + vx * dt
            y = y + vy * dt
            z = z + vz * dt

            # 防止高度跑飞：轻微反弹到范围内（让轨迹仍在画布内）
            if z < AREA3D[4]:
                z = AREA3D[4] + (AREA3D[4] - z) * 0.3
            if z > AREA3D[5]:
                z = AREA3D[5] - (z - AREA3D[5]) * 0.3

            states[k, i] = np.array([x, y, z, vx, vy, vz], float)

    return states, params_list


def save_params(scene_name: str, params_list: List[dict], out_dir=OUT_DIR):
    csv_path = os.path.join(out_dir, f"{scene_name}_params.csv")
    keys = sorted({k for p in params_list for k in p.keys()})
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for p in params_list:
            row = []
            for k in keys:
                v = p.get(k, "")
                if isinstance(v, (list, dict)):
                    row.append(json.dumps(v, ensure_ascii=False))
                else:
                    row.append("" if v is None else str(v))
            f.write(",".join(row) + "\n")

    json_path = os.path.join(out_dir, f"{scene_name}_params.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params_list, f, ensure_ascii=False, indent=2)

    print(f"[saved params] {csv_path}")
    print(f"[saved params] {json_path}")


# =========================
# 量测生成（3D，带tid用于误跟/失跟评估）
# =========================
@dataclass
class Measurement:
    z: np.ndarray  # (3,)
    tid: int       # 真目标 0..N-1；杂波 -1
    detected: bool


def generate_measurements(truth_states: np.ndarray,
                          sigma_z=SIGMA_Z,
                          Pd=PD,
                          clutter_lambda=CLUTTER_LAMBDA,
                          area3d=AREA3D,
                          seed=SEED) -> Tuple[List[List[Measurement]], np.ndarray]:
    rng = np.random.default_rng(seed + 100)
    T, n, _ = truth_states.shape
    meas_list: List[List[Measurement]] = []
    det_flag = np.zeros((T, n), dtype=bool)

    xmin, xmax, ymin, ymax, zmin, zmax = area3d

    for k in range(T):
        frame = []
        for i in range(n):
            if rng.random() < Pd:
                det_flag[k, i] = True
                x, y, z = truth_states[k, i, 0], truth_states[k, i, 1], truth_states[k, i, 2]
                zz = np.array([x, y, z]) + rng.normal(0, sigma_z, size=3)
                frame.append(Measurement(z=zz, tid=i, detected=True))
            else:
                det_flag[k, i] = False

        m_cl = rng.poisson(clutter_lambda)
        for _ in range(m_cl):
            zz = np.array([rng.uniform(xmin, xmax),
                           rng.uniform(ymin, ymax),
                           rng.uniform(zmin, zmax)], float)
            frame.append(Measurement(z=zz, tid=-1, detected=False))

        meas_list.append(frame)

    return meas_list, det_flag


# =========================
# 3D KF / IMM
# =========================
def cv_mats_3d(dt: float, q: float):
    """
    3D CV: x=[px,py,pz,vx,vy,vz], dim=6
    """
    F = np.array([
        [1, 0, 0, dt, 0,  0 ],
        [0, 1, 0, 0,  dt, 0 ],
        [0, 0, 1, 0,  0,  dt],
        [0, 0, 0, 1,  0,  0 ],
        [0, 0, 0, 0,  1,  0 ],
        [0, 0, 0, 0,  0,  1 ],
    ], float)

    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2

    # block diag for x/y/z (white accel)
    Q1 = np.array([[dt4/4, dt3/2],
                   [dt3/2, dt2]], float) * q
    Q = np.zeros((6, 6), float)
    # x
    Q[0, 0] = Q1[0, 0]; Q[0, 3] = Q1[0, 1]
    Q[3, 0] = Q1[1, 0]; Q[3, 3] = Q1[1, 1]
    # y
    Q[1, 1] = Q1[0, 0]; Q[1, 4] = Q1[0, 1]
    Q[4, 1] = Q1[1, 0]; Q[4, 4] = Q1[1, 1]
    # z
    Q[2, 2] = Q1[0, 0]; Q[2, 5] = Q1[0, 1]
    Q[5, 2] = Q1[1, 0]; Q[5, 5] = Q1[1, 1]

    H = np.array([
        [1,0,0,0,0,0],
        [0,1,0,0,0,0],
        [0,0,1,0,0,0],
    ], float)

    R = (SIGMA_Z**2) * np.eye(3)
    return F, Q, H, R


def kf_predict(x, P, F, Q):
    x = F @ x
    P = F @ P @ F.T + Q
    return x, P


def kf_update(x, P, z, H, R):
    y = z - (H @ x)
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    P = (np.eye(len(x)) - K @ H) @ P
    return x, P, y, S


def maha2(innov, S):
    return float(innov.T @ np.linalg.inv(S) @ innov)


@dataclass
class TrackKF:
    tid: int
    x: np.ndarray
    P: np.ndarray
    q: float

    def predict(self, dt=DT):
        F, Q, H, R = cv_mats_3d(dt, self.q)
        self.x, self.P = kf_predict(self.x, self.P, F, Q)
        return H @ self.x, (H @ self.P @ H.T + R)

    def update(self, z, dt=DT):
        F, Q, H, R = cv_mats_3d(dt, self.q)
        self.x, self.P, innov, S = kf_update(self.x, self.P, z, H, R)
        return innov, S


@dataclass
class TrackIMM:
    tid: int
    x_list: List[np.ndarray]
    P_list: List[np.ndarray]
    mu: np.ndarray
    Pi: np.ndarray
    q_list: List[float]

    def mix(self):
        c = self.Pi.T @ self.mu
        M = len(self.mu)
        mu_ij = np.zeros((M, M))
        for j in range(M):
            for i in range(M):
                mu_ij[i, j] = self.Pi[i, j] * self.mu[i] / max(1e-12, c[j])

        x0, P0 = [], []
        for j in range(M):
            xj = sum(mu_ij[i, j] * self.x_list[i] for i in range(M))
            Pj = np.zeros_like(self.P_list[0])
            for i in range(M):
                dx = (self.x_list[i] - xj).reshape(-1, 1)
                Pj += mu_ij[i, j] * (self.P_list[i] + dx @ dx.T)
            x0.append(xj)
            P0.append(Pj)

        self.x_list, self.P_list = x0, P0
        self.mu = c / max(1e-12, np.sum(c))

    def predict(self, dt=DT):
        self.mix()
        zpred_list, Spred_list = [], []
        for m in range(2):
            F, Q, H, R = cv_mats_3d(dt, self.q_list[m])
            x, P = kf_predict(self.x_list[m], self.P_list[m], F, Q)
            self.x_list[m], self.P_list[m] = x, P
            zpred_list.append(H @ x)
            Spred_list.append(H @ P @ H.T + R)

        zpred = self.mu[0]*zpred_list[0] + self.mu[1]*zpred_list[1]
        Spred = self.mu[0]*Spred_list[0] + self.mu[1]*Spred_list[1]
        return zpred, Spred

    def update(self, z, dt=DT):
        lik = np.zeros(2)
        for m in range(2):
            F, Q, H, R = cv_mats_3d(dt, self.q_list[m])
            x, P, innov, S = kf_update(self.x_list[m], self.P_list[m], z, H, R)
            self.x_list[m], self.P_list[m] = x, P

            d2 = maha2(innov, S)
            detS = max(1e-12, np.linalg.det(S))
            lik[m] = math.exp(-0.5*d2) / math.sqrt(detS)

        self.mu = self.mu * lik
        self.mu = self.mu / max(1e-12, np.sum(self.mu))

        xbar = self.mu[0]*self.x_list[0] + self.mu[1]*self.x_list[1]
        Pbar = np.zeros_like(self.P_list[0])
        for m in range(2):
            dx = (self.x_list[m] - xbar).reshape(-1, 1)
            Pbar += self.mu[m] * (self.P_list[m] + dx @ dx.T)
        return xbar, Pbar


# =========================
# GNN（3D）
# =========================
def run_gnn(truth, meas_list):
    T, n, _ = truth.shape
    tracks = []
    for i in range(n):
        x0 = truth[0, i].copy()
        P0 = np.diag([80, 80, 80, 50, 50, 50]).astype(float)
        tracks.append(TrackKF(tid=i, x=x0, P=P0, q=0.8))

    est = np.zeros((T, n, 3), float)
    assoc_tid = -2 * np.ones((T, n), int)  # -2 miss, -1 clutter, else true tid

    for k in range(T):
        frame = meas_list[k]
        Z = np.array([m.z for m in frame], float) if len(frame) else np.zeros((0, 3), float)
        M = len(frame)

        zpred, Spred = [], []
        for tr in tracks:
            zp, Sp = tr.predict()
            zpred.append(zp)
            Spred.append(Sp)

        C = np.full((n, M + n), fill_value=1e6, dtype=float)
        for i in range(n):
            for j in range(M):
                innov = Z[j] - zpred[i]
                d2 = maha2(innov, Spred[i])
                if d2 <= GATE_CHI2:
                    C[i, j] = d2
            C[i, M + i] = MISS_COST

        row_ind, col_ind = linear_sum_assignment(C)

        for i, c in zip(row_ind, col_ind):
            if c < M and C[i, c] < 1e5:
                tracks[i].update(Z[c])
                assoc_tid[k, i] = frame[c].tid
            else:
                assoc_tid[k, i] = -2
            est[k, i] = tracks[i].x[:3]

    return est, assoc_tid


# =========================
# 近似 MHT / IMM-MHT（Top-K）
# =========================
@dataclass
class Hypothesis:
    tracks: List
    logw: float
    assoc: List[int]   # per track: measurement index, -1 miss
    used_meas: set


def log_gauss_likelihood(innov, S):
    d2 = maha2(innov, S)
    detS = max(1e-12, np.linalg.det(S))
    return -0.5*d2 - 0.5*math.log(detS)


def clone_track(track):
    if isinstance(track, TrackKF):
        return TrackKF(track.tid, track.x.copy(), track.P.copy(), track.q)
    elif isinstance(track, TrackIMM):
        return TrackIMM(
            track.tid,
            [x.copy() for x in track.x_list],
            [P.copy() for P in track.P_list],
            track.mu.copy(),
            track.Pi.copy(),
            list(track.q_list),
        )
    else:
        raise TypeError("Unknown track type")


def run_mht_like(truth, meas_list, use_imm=False, K_keep=35, branch_per_track=4):
    T, n, _ = truth.shape

    base_tracks = []
    for i in range(n):
        x0 = truth[0, i].copy()
        P0 = np.diag([80, 80, 80, 50, 50, 50]).astype(float)
        if not use_imm:
            base_tracks.append(TrackKF(tid=i, x=x0, P=P0, q=0.65))
        else:
            mu0 = np.array([0.7, 0.3], float)
            Pi = np.array([[0.95, 0.05],
                           [0.08, 0.92]], float)
            base_tracks.append(TrackIMM(
                tid=i,
                x_list=[x0.copy(), x0.copy()],
                P_list=[P0.copy(), P0.copy()],
                mu=mu0,
                Pi=Pi,
                q_list=[0.25, 2.2],  # low/high q
            ))

    hyps = [Hypothesis(tracks=[clone_track(t) for t in base_tracks],
                       logw=0.0,
                       assoc=[-1]*n,
                       used_meas=set())]

    est = np.zeros((T, n, 3), float)
    assoc_tid = -2 * np.ones((T, n), int)

    for k in range(T):
        frame = meas_list[k]
        Z = np.array([m.z for m in frame], float) if len(frame) else np.zeros((0, 3), float)
        M = len(frame)

        new_hyps = []

        for h in hyps:
            zpred, Spred = [], []
            for tr in h.tracks:
                zp, Sp = tr.predict()
                zpred.append(zp)
                Spred.append(Sp)

            partial = [Hypothesis(
                tracks=[clone_track(t) for t in h.tracks],
                logw=h.logw,
                assoc=[-1]*n,
                used_meas=set(),
            )]

            for i in range(n):
                next_partial = []
                for ph in partial:
                    cand = []
                    for j in range(M):
                        if j in ph.used_meas:
                            continue
                        innov = Z[j] - zpred[i]
                        d2 = maha2(innov, Spred[i])
                        if d2 <= GATE_CHI2:
                            cand.append((j, d2))
                    cand.sort(key=lambda x: x[1])
                    cand = cand[:branch_per_track]

                    # match branches
                    for (j, _d2) in cand:
                        new_tracks = [clone_track(t) for t in ph.tracks]
                        tr_i = new_tracks[i]

                        innov = Z[j] - zpred[i]
                        ll = log_gauss_likelihood(innov, Spred[i])

                        # update track state
                        if isinstance(tr_i, TrackKF):
                            tr_i.update(Z[j])
                        else:
                            tr_i.update(Z[j])

                        nh = Hypothesis(
                            tracks=new_tracks,
                            logw=ph.logw + ll,
                            assoc=ph.assoc.copy(),
                            used_meas=set(ph.used_meas),
                        )
                        nh.assoc[i] = j
                        nh.used_meas.add(j)
                        next_partial.append(nh)

                    # miss branch
                    miss_penalty = -0.5 * MISS_COST
                    nh = Hypothesis(
                        tracks=[clone_track(t) for t in ph.tracks],
                        logw=ph.logw + miss_penalty,
                        assoc=ph.assoc.copy(),
                        used_meas=set(ph.used_meas),
                    )
                    nh.assoc[i] = -1
                    next_partial.append(nh)

                next_partial.sort(key=lambda hh: hh.logw, reverse=True)
                partial = next_partial[:K_keep]

            new_hyps.extend(partial)

        new_hyps.sort(key=lambda hh: hh.logw, reverse=True)
        hyps = new_hyps[:K_keep]

        best = hyps[0]
        for i in range(n):
            if best.assoc[i] >= 0 and best.assoc[i] < M:
                assoc_tid[k, i] = frame[best.assoc[i]].tid
            else:
                assoc_tid[k, i] = -2

            tr = best.tracks[i]
            if isinstance(tr, TrackKF):
                est[k, i] = tr.x[:3]
            else:
                xbar = tr.mu[0]*tr.x_list[0] + tr.mu[1]*tr.x_list[1]
                est[k, i] = xbar[:3]

    return est, assoc_tid


# =========================
# 指标与绘图（3D）
# =========================
def compute_metrics(truth, est, assoc_tid, det_flag):
    """
    - ARMSE: mean over time of RMSE(t), RMSE(t)=sqrt(mean_i(||e_i||^2))
    - ErrorRate: 在“发生更新(非miss)”时，分配到的量测tid != 真实tid 的比例
    - MissRate: 在“真实目标被探测(det_flag=True)”时，却被判定miss 的比例
    """
    T, n, _ = truth.shape
    err = est - truth[:, :, :3]
    rmse_t = np.sqrt(np.mean(np.sum(err**2, axis=2), axis=1))  # 3D位置RMSE
    armse = float(np.mean(rmse_t))

    update_mask = (assoc_tid != -2)
    wrong, total_updates = 0, 0
    for k in range(T):
        for i in range(n):
            if update_mask[k, i]:
                total_updates += 1
                if assoc_tid[k, i] != i:
                    wrong += 1
    err_rate = 100.0 * wrong / max(1, total_updates)

    miss, total_det = 0, 0
    for k in range(T):
        for i in range(n):
            if det_flag[k, i]:
                total_det += 1
                if assoc_tid[k, i] == -2:
                    miss += 1
    miss_rate = 100.0 * miss / max(1, total_det)

    return armse, rmse_t, err_rate, miss_rate


def _set_3d_axes_limits(ax):
    xmin, xmax, ymin, ymax, zmin, zmax = AREA3D
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def plot_assoc_3d(truth, est, meas_list, title, out_path, view=(18, 35)):
    """
    Matlab风格 3D：plot3 + scatter3
    真实轨迹(更粗实线) vs 估计轨迹(更粗虚线)
    """
    fig = plt.figure(figsize=(9.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    # measurements (subsample to avoid too heavy)
    allz = []
    for frame in meas_list:
        if len(frame):
            allz.append(np.array([m.z for m in frame], float))
    if len(allz):
        allz = np.concatenate(allz, axis=0)
        # 为了画面观感，点太多就做下采样
        if allz.shape[0] > 8000:
            idx = np.random.default_rng(0).choice(allz.shape[0], size=8000, replace=False)
            allz = allz[idx]
        ax.scatter(allz[:, 0], allz[:, 1], allz[:, 2], s=6, alpha=0.18)

    n = truth.shape[1]
    for i in range(n):
        ax.plot(truth[:, i, 0], truth[:, i, 1], truth[:, i, 2],
                linewidth=2.8, alpha=0.95, label="Truth" if i == 0 else None)
        ax.plot(est[:, i, 0], est[:, i, 1], est[:, i, 2],
                linewidth=2.2, linestyle="--", alpha=0.95, label="Estimate" if i == 0 else None)

    ax.set_title(title)
    _set_3d_axes_limits(ax)
    ax.grid(True, alpha=0.25)

    elev, azim = view
    ax.view_init(elev=elev, azim=azim)

    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_rmse(rmse_t, title, out_path):
    plt.figure(figsize=(9.2, 4.6))
    plt.plot(np.arange(len(rmse_t)), rmse_t, linewidth=2.4)
    plt.title(title)
    plt.xlabel("time step")
    plt.ylabel("RMSE (3D position)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def run_one_scene(scene_name, truth_states, meas_list, det_flag):
    results = []

    # ---- GNN ----
    t0 = time.perf_counter()
    est_gnn, assoc_gnn = run_gnn(truth_states, meas_list)
    t1 = time.perf_counter()
    armse, rmse_t, err_rate, miss_rate = compute_metrics(truth_states, est_gnn, assoc_gnn, det_flag)
    results.append(("GNN", armse, (t1 - t0), err_rate, miss_rate))
    plot_assoc_3d(truth_states, est_gnn, meas_list,
                  f"{scene_name} | GNN | 3D Truth vs Tracking",
                  os.path.join(OUT_DIR, f"{scene_name}_GNN_assoc3d.png"))
    plot_rmse(rmse_t,
              f"{scene_name} | GNN | RMSE(t) in 3D",
              os.path.join(OUT_DIR, f"{scene_name}_GNN_rmse.png"))

    # ---- MHT ----
    t0 = time.perf_counter()
    est_mht, assoc_mht = run_mht_like(truth_states, meas_list, use_imm=False, K_keep=35, branch_per_track=4)
    t1 = time.perf_counter()
    armse, rmse_t, err_rate, miss_rate = compute_metrics(truth_states, est_mht, assoc_mht, det_flag)
    results.append(("MHT", armse, (t1 - t0), err_rate, miss_rate))
    plot_assoc_3d(truth_states, est_mht, meas_list,
                  f"{scene_name} | MHT | 3D Truth vs Tracking",
                  os.path.join(OUT_DIR, f"{scene_name}_MHT_assoc3d.png"))
    plot_rmse(rmse_t,
              f"{scene_name} | MHT | RMSE(t) in 3D",
              os.path.join(OUT_DIR, f"{scene_name}_MHT_rmse.png"))

    # ---- IMM-MHT ----
    t0 = time.perf_counter()
    est_im, assoc_im = run_mht_like(truth_states, meas_list, use_imm=True, K_keep=35, branch_per_track=4)
    t1 = time.perf_counter()
    armse, rmse_t, err_rate, miss_rate = compute_metrics(truth_states, est_im, assoc_im, det_flag)
    results.append(("IMM-MHT", armse, (t1 - t0), err_rate, miss_rate))
    plot_assoc_3d(truth_states, est_im, meas_list,
                  f"{scene_name} | IMM-MHT | 3D Truth vs Tracking",
                  os.path.join(OUT_DIR, f"{scene_name}_IMM-MHT_assoc3d.png"))
    plot_rmse(rmse_t,
              f"{scene_name} | IMM-MHT | RMSE(t) in 3D",
              os.path.join(OUT_DIR, f"{scene_name}_IMM-MHT_rmse.png"))

    return results


def print_table(all_rows):
    headers = ["Scene", "Algorithm", "ARMSE", "SimTime(s)", "ErrorRate(%)", "MissRate(%)"]
    colw = [10, 12, 10, 12, 14, 13]
    line = "".join(h.ljust(w) for h, w in zip(headers, colw))
    print(line)
    print("-" * sum(colw))
    for row in all_rows:
        scene, alg, armse, st, er, mr = row
        vals = [scene, alg, f"{armse:.3f}", f"{st:.4f}", f"{er:.2f}", f"{mr:.2f}"]
        print("".join(v.ljust(w) for v, w in zip(vals, colw)))


def main():
    all_rows = []

    # Scene1
    truth1, params1 = gen_scene1_truth()
    save_params("Scene1", params1)
    meas1, det1 = generate_measurements(truth1, seed=SEED)
    res1 = run_one_scene("Scene1", truth1, meas1, det1)
    for (alg, armse, st, er, mr) in res1:
        all_rows.append(("Scene1", alg, armse, st, er, mr))

    # Scene2
    truth2, params2 = gen_scene2_truth()
    save_params("Scene2", params2)
    meas2, det2 = generate_measurements(truth2, seed=SEED + 1)
    res2 = run_one_scene("Scene2", truth2, meas2, det2)
    for (alg, armse, st, er, mr) in res2:
        all_rows.append(("Scene2", alg, armse, st, er, mr))

    print("\n=== Metrics Table (3D) ===")
    print_table(all_rows)
    print(f"\n[Saved figures] -> {OUT_DIR}/")
    print("  - 12 figures: Scene{1,2} × {GNN,MHT,IMM-MHT} × {assoc3d,rmse}")
    print("  - params: Scene1_params.(csv/json), Scene2_params.(csv/json)")


if __name__ == "__main__":
    main()