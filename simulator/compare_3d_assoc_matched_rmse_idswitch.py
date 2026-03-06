import os
import time
import math
import json
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    from scipy.optimize import linear_sum_assignment
except Exception as e:
    raise ImportError("需要 scipy：pip install scipy") from e


# =========================================================
# 说明：Scene2 为什么 RMSE “发散”
# =========================================================
# 原因：固定编号 i 对 i 的 RMSE 会被 “ID swap（标签交换）” 放大。
# 在多目标接近/交叉 + 漏检 + 杂波 + 机动的场景下（Scene2），
# GNN/MHT/IMM-MHT 都可能发生轨迹编号交换：
#   est_i 实际在跟 truth_j（j!=i），几何轨迹仍然贴合，但编号错了。
# 固定ID RMSE 会变成“目标间距离”，并随目标群分离而越来越大，看似发散。
#
# 修复：
#   1) RMSE 使用 per-frame Hungarian 最优匹配（truth<->est）再算误差 => RMSE_matched
#   2) 额外统计 ID switches：连续帧的匹配关系发生变化的次数
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
SIGMA_Z = 6.0
PD = 0.92
CLUTTER_LAMBDA = 10
AREA3D = (-900, 900, -900, 900, 0, 600)  # xmin,xmax,ymin,ymax,zmin,zmax

# 门限与代价：chi2(3)
GATE_CHI2 = 11.34   # approx chi2 df=3 at 0.99
MISS_COST = 25.0

# 速度范围
VMIN, VMAX = 10.0, 25.0

# 额外输出固定ID RMSE（用于诊断 swap 影响）
PLOT_FIXED_ID_RMSE = True


# =========================
# 轨迹生成（3D）
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
            "x0": float(x0[i]), "y0": float(y0[i]), "z0": float(z0[i]),
            "speed0": float(speed[i]),
            "azimuth0_rad": float(azim[i]),
            "elev0_rad": float(elev[i]),
            "vx0": float(vx0[i]), "vy0": float(vy0[i]), "vz0": float(vz0[i]),
            "motion": "3D_CV_straight",
            "t1": None, "t2": None, "t3": None, "t4": None,
            "accel_mag": 0.0,
            "turn_total_rad": 0.0,
            "turn_rate_rad_per_step": 0.0,
            "climb_total_rad": 0.0,
            "climb_rate_rad_per_step": 0.0,
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

    x0 = rng.uniform(-X0, X0, size=n)
    y0 = rng.uniform(-Y0, Y0, size=n)
    z0 = rng.uniform(ZMIN, ZMAX, size=n)

    speed0 = rng.uniform(VMIN, VMAX, size=n)
    yaw0 = rng.uniform(0, 2*np.pi, size=n)
    elev_max0 = np.deg2rad(10)
    elev0 = rng.uniform(-elev_max0, elev_max0, size=n)

    vx0 = speed0 * np.cos(elev0) * np.cos(yaw0)
    vy0 = speed0 * np.cos(elev0) * np.sin(yaw0)
    vz0 = speed0 * np.sin(elev0)

    states[0, :, :] = np.stack([x0, y0, z0, vx0, vy0, vz0], axis=1)

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
            "x0": float(x0[i]), "y0": float(y0[i]), "z0": float(z0[i]),
            "speed0": float(speed0[i]),
            "yaw0_rad": float(yaw0[i]),
            "elev0_rad": float(elev0[i]),
            "vx0": float(vx0[i]), "vy0": float(vy0[i]), "vz0": float(vz0[i]),
            "motion": "3D_piecewise(CV + accel/decel + yaw_turn + climb + jitter)",
            "t1": t1, "t2": t2, "t3": t3, "t4": t4,
            "accel_mag": accel_mag,
            "turn_total_rad": turn_total,
            "turn_rate_rad_per_step": turn_rate,
            "climb_total_rad": climb_total,
            "climb_rate_rad_per_step": climb_rate,
            "jitter_prob": 0.06,
        })

    v_clip_min, v_clip_max = 6.0, 30.0
    elev_clip_min, elev_clip_max = -np.deg2rad(18), np.deg2rad(18)

    for i in range(n):
        t1, t2, t3, t4, accel_mag, turn_total, turn_rate, climb_total, climb_rate = scripts[i]

        yaw = yaw0[i]
        elev = elev0[i]
        v = speed0[i]

        for k in range(1, T):
            x, y, z, vx, vy, vz = states[k-1, i]

            if k < t1:
                pass
            elif k < t2:
                v = np.clip(v + accel_mag, v_clip_min, v_clip_max)
            elif k < t3:
                yaw = yaw + turn_rate
            elif k < t4:
                elev = np.clip(elev + climb_rate, elev_clip_min, elev_clip_max)
            else:
                if rng.random() < 0.06:
                    yaw = yaw + rng.uniform(-0.10, 0.10)
                    elev = np.clip(elev + rng.uniform(-0.06, 0.06), elev_clip_min, elev_clip_max)

            vx = v * np.cos(elev) * np.cos(yaw)
            vy = v * np.cos(elev) * np.sin(yaw)
            vz = v * np.sin(elev)

            x = x + vx * dt
            y = y + vy * dt
            z = z + vz * dt

            # keep z in box gently
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
# 量测生成（3D）
# =========================
@dataclass
class Measurement:
    z: np.ndarray
    tid: int
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

    Q1 = np.array([[dt4/4, dt3/2],
                   [dt3/2, dt2]], float) * q

    Q = np.zeros((6, 6), float)
    # x block
    Q[0, 0] = Q1[0, 0]; Q[0, 3] = Q1[0, 1]
    Q[3, 0] = Q1[1, 0]; Q[3, 3] = Q1[1, 1]
    # y block
    Q[1, 1] = Q1[0, 0]; Q[1, 4] = Q1[0, 1]
    Q[4, 1] = Q1[1, 0]; Q[4, 4] = Q1[1, 1]
    # z block
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
                q_list=[0.25, 2.2],
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

                    for (j, _d2) in cand:
                        new_tracks = [clone_track(t) for t in ph.tracks]
                        tr_i = new_tracks[i]

                        innov = Z[j] - zpred[i]
                        ll = log_gauss_likelihood(innov, Spred[i])

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
# RMSE / ID-switch 指标：基于 per-frame Hungarian matching
# =========================
def per_frame_assignment(truth_xyz: np.ndarray, est_xyz: np.ndarray):
    """
    返回：
      assign[k, i] = 被匹配到 truth_i 的 est 索引（0..N-1）
    其中 i 是 truth index。
    """
    Tt, N, _ = truth_xyz.shape
    assign = np.zeros((Tt, N), dtype=int)
    for k in range(Tt):
        diff = truth_xyz[k, :, None, :] - est_xyz[k, None, :, :]
        C = np.sum(diff**2, axis=2)  # (N,N)
        r, c = linear_sum_assignment(C)
        # r 应该是 [0..N-1]，但保险起见：
        for rr, cc in zip(r, c):
            assign[k, rr] = cc
    return assign


def rmse_matched(truth_xyz: np.ndarray, est_xyz: np.ndarray):
    Tt, N, _ = truth_xyz.shape
    rmse_t = np.zeros(Tt)
    for k in range(Tt):
        diff = truth_xyz[k, :, None, :] - est_xyz[k, None, :, :]
        C = np.sum(diff**2, axis=2)
        r, c = linear_sum_assignment(C)
        rmse_t[k] = np.sqrt(np.mean(C[r, c]))
    return rmse_t


def idswitch_count(assign: np.ndarray):
    """
    assign[k, truth_i] = est_index
    ID switch：同一个 truth_i 在相邻帧匹配到的 est_index 发生变化
    """
    Tt, N = assign.shape
    sw = 0
    for k in range(1, Tt):
        sw += int(np.sum(assign[k] != assign[k-1]))
    return sw


# =========================
# 误跟/失跟（仍按 assoc_tid 对比）
# =========================
def assoc_error_miss_rates(assoc_tid: np.ndarray, det_flag: np.ndarray):
    Tt, N = assoc_tid.shape
    # ErrorRate：发生更新时 assoc_tid != i
    wrong, total_updates = 0, 0
    for k in range(Tt):
        for i in range(N):
            if assoc_tid[k, i] != -2:  # updated
                total_updates += 1
                if assoc_tid[k, i] != i:
                    wrong += 1
    err_rate = 100.0 * wrong / max(1, total_updates)

    # MissRate：det_flag=True 但 assoc 被判 miss(-2)
    miss, total_det = 0, 0
    for k in range(Tt):
        for i in range(N):
            if det_flag[k, i]:
                total_det += 1
                if assoc_tid[k, i] == -2:
                    miss += 1
    miss_rate = 100.0 * miss / max(1, total_det)
    return err_rate, miss_rate


def rmse_fixed_id(truth_xyz: np.ndarray, est_xyz: np.ndarray):
    err = est_xyz - truth_xyz
    rmse_t = np.sqrt(np.mean(np.sum(err**2, axis=2), axis=1))
    return rmse_t


# =========================
# 绘图（3D轨迹图 + RMSE图）
# =========================
def _set_3d_axes_limits(ax):
    xmin, xmax, ymin, ymax, zmin, zmax = AREA3D
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def plot_assoc_3d(truth, est, meas_list, title, out_path, view=(18, 35)):
    fig = plt.figure(figsize=(9.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    allz = []
    for frame in meas_list:
        if len(frame):
            allz.append(np.array([m.z for m in frame], float))
    if len(allz):
        allz = np.concatenate(allz, axis=0)
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


def plot_rmse_curve(rmse_t, title, out_path, ylabel="RMSE (3D position)"):
    plt.figure(figsize=(9.2, 4.6))
    plt.plot(np.arange(len(rmse_t)), rmse_t, linewidth=2.4)
    plt.title(title)
    plt.xlabel("time step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_rmse_two_curves(rmse_matched_t, rmse_fixed_t, title, out_path):
    plt.figure(figsize=(9.2, 4.6))
    plt.plot(np.arange(len(rmse_matched_t)), rmse_matched_t, linewidth=2.4)
    plt.plot(np.arange(len(rmse_fixed_t)), rmse_fixed_t, linewidth=2.0, linestyle="--")
    plt.title(title)
    plt.xlabel("time step")
    plt.ylabel("RMSE (3D position)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# =========================
# 单场景运行 + 指标汇总
# =========================
def compute_all_metrics(truth_states, est_xyz, assoc_tid, det_flag):
    truth_xyz = truth_states[:, :, :3]

    # matched RMSE + assignment
    rmse_m = rmse_matched(truth_xyz, est_xyz)
    armse_m = float(np.mean(rmse_m))

    assign = per_frame_assignment(truth_xyz, est_xyz)
    sw = idswitch_count(assign)
    sw_rate = 100.0 * sw / max(1, (truth_xyz.shape[0] - 1) * truth_xyz.shape[1])

    # fixed ID RMSE (diagnostic)
    rmse_f = rmse_fixed_id(truth_xyz, est_xyz)
    armse_f = float(np.mean(rmse_f))

    # error/miss rates based on assoc_tid
    err_rate, miss_rate = assoc_error_miss_rates(assoc_tid, det_flag)

    return {
        "rmse_matched_t": rmse_m,
        "armse_matched": armse_m,
        "assign": assign,
        "idswitches": int(sw),
        "idswitch_rate": float(sw_rate),
        "rmse_fixed_t": rmse_f,
        "armse_fixed": armse_f,
        "error_rate": float(err_rate),
        "miss_rate": float(miss_rate),
    }


def run_one_scene(scene_name, truth_states, meas_list, det_flag):
    results_rows = []

    # ---------------- GNN ----------------
    t0 = time.perf_counter()
    est_gnn, assoc_gnn = run_gnn(truth_states, meas_list)
    t1 = time.perf_counter()
    m = compute_all_metrics(truth_states, est_gnn, assoc_gnn, det_flag)

    plot_assoc_3d(truth_states, est_gnn, meas_list,
                  f"{scene_name} | GNN | 3D Truth vs Tracking",
                  os.path.join(OUT_DIR, f"{scene_name}_GNN_assoc3d.png"))
    if PLOT_FIXED_ID_RMSE:
        plot_rmse_two_curves(m["rmse_matched_t"], m["rmse_fixed_t"],
                             f"{scene_name} | GNN | RMSE(t) Matched(solid) vs FixedID(dashed)",
                             os.path.join(OUT_DIR, f"{scene_name}_GNN_rmse.png"))
    else:
        plot_rmse_curve(m["rmse_matched_t"],
                        f"{scene_name} | GNN | RMSE(t) Matched",
                        os.path.join(OUT_DIR, f"{scene_name}_GNN_rmse.png"))

    results_rows.append({
        "Scene": scene_name,
        "Algorithm": "GNN",
        "ARMSE_matched": m["armse_matched"],
        "ARMSE_fixedID": m["armse_fixed"],
        "SimTime(s)": (t1 - t0),
        "ErrorRate(%)": m["error_rate"],
        "MissRate(%)": m["miss_rate"],
        "IDSwitches": m["idswitches"],
        "IDSwitchRate(%)": m["idswitch_rate"],
    })

    # ---------------- MHT ----------------
    t0 = time.perf_counter()
    est_mht, assoc_mht = run_mht_like(truth_states, meas_list, use_imm=False, K_keep=35, branch_per_track=4)
    t1 = time.perf_counter()
    m = compute_all_metrics(truth_states, est_mht, assoc_mht, det_flag)

    plot_assoc_3d(truth_states, est_mht, meas_list,
                  f"{scene_name} | MHT | 3D Truth vs Tracking",
                  os.path.join(OUT_DIR, f"{scene_name}_MHT_assoc3d.png"))
    if PLOT_FIXED_ID_RMSE:
        plot_rmse_two_curves(m["rmse_matched_t"], m["rmse_fixed_t"],
                             f"{scene_name} | MHT | RMSE(t) Matched(solid) vs FixedID(dashed)",
                             os.path.join(OUT_DIR, f"{scene_name}_MHT_rmse.png"))
    else:
        plot_rmse_curve(m["rmse_matched_t"],
                        f"{scene_name} | MHT | RMSE(t) Matched",
                        os.path.join(OUT_DIR, f"{scene_name}_MHT_rmse.png"))

    results_rows.append({
        "Scene": scene_name,
        "Algorithm": "MHT",
        "ARMSE_matched": m["armse_matched"],
        "ARMSE_fixedID": m["armse_fixed"],
        "SimTime(s)": (t1 - t0),
        "ErrorRate(%)": m["error_rate"],
        "MissRate(%)": m["miss_rate"],
        "IDSwitches": m["idswitches"],
        "IDSwitchRate(%)": m["idswitch_rate"],
    })

    # -------------- IMM-MHT --------------
    t0 = time.perf_counter()
    est_im, assoc_im = run_mht_like(truth_states, meas_list, use_imm=True, K_keep=35, branch_per_track=4)
    t1 = time.perf_counter()
    m = compute_all_metrics(truth_states, est_im, assoc_im, det_flag)

    plot_assoc_3d(truth_states, est_im, meas_list,
                  f"{scene_name} | IMM-MHT | 3D Truth vs Tracking",
                  os.path.join(OUT_DIR, f"{scene_name}_IMM-MHT_assoc3d.png"))
    if PLOT_FIXED_ID_RMSE:
        plot_rmse_two_curves(m["rmse_matched_t"], m["rmse_fixed_t"],
                             f"{scene_name} | IMM-MHT | RMSE(t) Matched(solid) vs FixedID(dashed)",
                             os.path.join(OUT_DIR, f"{scene_name}_IMM-MHT_rmse.png"))
    else:
        plot_rmse_curve(m["rmse_matched_t"],
                        f"{scene_name} | IMM-MHT | RMSE(t) Matched",
                        os.path.join(OUT_DIR, f"{scene_name}_IMM-MHT_rmse.png"))

    results_rows.append({
        "Scene": scene_name,
        "Algorithm": "IMM-MHT",
        "ARMSE_matched": m["armse_matched"],
        "ARMSE_fixedID": m["armse_fixed"],
        "SimTime(s)": (t1 - t0),
        "ErrorRate(%)": m["error_rate"],
        "MissRate(%)": m["miss_rate"],
        "IDSwitches": m["idswitches"],
        "IDSwitchRate(%)": m["idswitch_rate"],
    })

    return results_rows


# =========================
# 表格打印 + 保存汇总
# =========================
def print_table(rows: List[Dict]):
    headers = ["Scene", "Algorithm", "ARMSE_matched", "ARMSE_fixedID", "SimTime(s)",
               "ErrorRate(%)", "MissRate(%)", "IDSwitches", "IDSwitchRate(%)"]
    colw = [10, 10, 15, 15, 12, 14, 13, 12, 16]

    line = "".join(h.ljust(w) for h, w in zip(headers, colw))
    print(line)
    print("-" * sum(colw))

    for r in rows:
        vals = [
            r["Scene"],
            r["Algorithm"],
            f"{r['ARMSE_matched']:.3f}",
            f"{r['ARMSE_fixedID']:.3f}",
            f"{r['SimTime(s)']:.4f}",
            f"{r['ErrorRate(%)']:.2f}",
            f"{r['MissRate(%)']:.2f}",
            str(r["IDSwitches"]),
            f"{r['IDSwitchRate(%)']:.2f}",
        ]
        print("".join(v.ljust(w) for v, w in zip(vals, colw)))


def save_summary_csv(rows: List[Dict], out_path: str):
    headers = ["Scene", "Algorithm", "ARMSE_matched", "ARMSE_fixedID", "SimTime(s)",
               "ErrorRate(%)", "MissRate(%)", "IDSwitches", "IDSwitchRate(%)"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join([
                r["Scene"], r["Algorithm"],
                f"{r['ARMSE_matched']:.6f}",
                f"{r['ARMSE_fixedID']:.6f}",
                f"{r['SimTime(s)']:.6f}",
                f"{r['ErrorRate(%)']:.6f}",
                f"{r['MissRate(%)']:.6f}",
                str(r["IDSwitches"]),
                f"{r['IDSwitchRate(%)']:.6f}",
            ]) + "\n")


# =========================
# main
# =========================
def main():
    all_rows = []

    # Scene1
    truth1, params1 = gen_scene1_truth()
    save_params("Scene1", params1)
    meas1, det1 = generate_measurements(truth1, seed=SEED)
    all_rows.extend(run_one_scene("Scene1", truth1, meas1, det1))

    # Scene2
    truth2, params2 = gen_scene2_truth()
    save_params("Scene2", params2)
    meas2, det2 = generate_measurements(truth2, seed=SEED + 1)
    all_rows.extend(run_one_scene("Scene2", truth2, meas2, det2))

    print("\n=== Metrics Summary (Matched RMSE + ID Switch) ===")
    print_table(all_rows)

    summary_csv = os.path.join(OUT_DIR, "summary_metrics.csv")
    save_summary_csv(all_rows, summary_csv)
    print(f"\n[saved] {summary_csv}")

    print(f"\n[Saved figures] -> {OUT_DIR}/")
    print("  - 12 figures: Scene{1,2} × {GNN,MHT,IMM-MHT} × {assoc3d,rmse}")
    print("  - params: Scene1_params.(csv/json), Scene2_params.(csv/json)")
    print("  - summary: summary_metrics.csv")


if __name__ == "__main__":
    main()