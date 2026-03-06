import os
import time
import math
import json
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import List, Tuple, Dict

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    from scipy.optimize import linear_sum_assignment
except Exception as e:
    raise ImportError("需要 scipy：pip install scipy") from e


# =========================================================
# Scene2 only: compare GNN vs MHT vs IMM-MHT
# Fixes:
#   (1) MHT/IMM-MHT use proper target-vs-clutter log-likelihood ratio (LLR)
#   (2) Larger, correct P0 (variance-scale) for MHT/IMM-MHT
#   (3) TrackIMM.miss() bugfix + velocity damping on miss
#   (4) Joseph-form KF update (already used)
#   (5) Reacquire without extra predict (use Euclidean nearest)
#
# Outputs:
#   - Scene2_params.csv/json
#   - 6 figures: Scene2_{ALG}_{assoc3d,rmse}.png
#   - summary_metrics_scene2.csv
# =========================================================


# =========================
# Global config
# =========================
SEED = 42
np.random.seed(SEED)

OUT_DIR = "sim_outputs_scene2"
os.makedirs(OUT_DIR, exist_ok=True)

DT = 1.0
T = 90
N_TARGETS = 10

# bounds for plotting and clutter
X0, Y0 = 800, 800
ZMIN, ZMAX = 50, 450
AREA3D = (-900, 900, -900, 900, 0, 600)  # xmin,xmax,ymin,ymax,zmin,zmax

# measurement model
SIGMA_Z = 6.0
PD = 0.92
CLUTTER_LAMBDA = 10

# ---------- gating ----------
# chi2(df=3, 0.9999) ≈ 21.11
GATE_CHI2 = 21.11
MISS_COST = 12.0  # used only in GNN cost, not used for MHT weight now

# ---------- miss inflation ----------
P_INFLATE_POS = (SIGMA_Z * 2.5) ** 2   # m^2
P_INFLATE_VEL = (6.0) ** 2
P_INFLATE_GROW = 1.18
P_INFLATE_CAP = 10

# ---------- optional reacquire ----------
ENABLE_REACQUIRE = True
REACQUIRE_AFTER_MISSES = 6        # consecutive misses threshold
# NOTE: we now use Euclidean distance for reacquire, so we use a radius threshold (meters)
REACQUIRE_EUCLIDEAN_R = 25.0      # meters

# speed range
VMIN, VMAX = 10.0, 25.0

# plot
PLOT_FIXED_ID_RMSE = True


# =========================
# Clutter density (lambda / volume)
# =========================
xmin, xmax, ymin, ymax, zmin, zmax = AREA3D
CLUTTER_VOLUME = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
CLUTTER_DENSITY = CLUTTER_LAMBDA / max(1e-12, CLUTTER_VOLUME)  # lambda / V


# =========================
# Scene2 truth (3D)
# =========================
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
# measurements (3D position + clutter)
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
    Tt, n, _ = truth_states.shape
    meas_list: List[List[Measurement]] = []
    det_flag = np.zeros((Tt, n), dtype=bool)

    xmin, xmax, ymin, ymax, zmin, zmax = area3d

    for k in range(Tt):
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
# 3D KF / IMM utilities
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
    I = np.eye(len(x))
    P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T  # Joseph
    P = 0.5 * (P + P.T)
    return x, P, y, S


def maha2(innov, S):
    return float(innov.T @ np.linalg.inv(S) @ innov)


def log_gauss_pdf(innov, S):
    """log N(innov; 0, S) with full constant term."""
    d = len(innov)
    d2 = maha2(innov, S)
    detS = max(1e-12, np.linalg.det(S))
    return -0.5 * (d2 + math.log(detS) + d * math.log(2.0 * math.pi))


@dataclass
class TrackKF:
    tid: int
    x: np.ndarray
    P: np.ndarray
    q: float
    miss_streak: int = 0

    def predict(self, dt=DT):
        F, Q, H, R = cv_mats_3d(dt, self.q)
        self.x, self.P = kf_predict(self.x, self.P, F, Q)
        return H @ self.x, (H @ self.P @ H.T + R)

    def update(self, z, dt=DT):
        F, Q, H, R = cv_mats_3d(dt, self.q)
        self.x, self.P, innov, S = kf_update(self.x, self.P, z, H, R)
        self.miss_streak = 0
        return innov, S

    def miss(self):
        self.miss_streak += 1
        s = min(self.miss_streak, P_INFLATE_CAP)
        grow = (P_INFLATE_GROW ** s)
        self.P[0:3, 0:3] += np.eye(3) * (P_INFLATE_POS * grow)
        self.P[3:6, 3:6] += np.eye(3) * (P_INFLATE_VEL * grow)
        # velocity damping to reduce drift under long miss
        self.x[3:6] *= 0.97

    def hard_reset_to_meas(self, z):
        self.x[0:3] = z
        self.P[0:3, 0:3] = np.eye(3) * (SIGMA_Z**2 * 4.0)
        self.P[3:6, 3:6] = np.eye(3) * (50.0)
        self.miss_streak = 0


@dataclass
class TrackIMM:
    tid: int
    x_list: List[np.ndarray]
    P_list: List[np.ndarray]
    mu: np.ndarray
    Pi: np.ndarray
    q_list: List[float]
    miss_streak: int = 0

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
        # model likelihoods based on Gaussian innovation likelihood (full pdf)
        lik = np.zeros(2)
        for m in range(2):
            F, Q, H, R = cv_mats_3d(dt, self.q_list[m])
            x, P, innov, S = kf_update(self.x_list[m], self.P_list[m], z, H, R)
            self.x_list[m], self.P_list[m] = x, P
            lik[m] = math.exp(log_gauss_pdf(innov, S))

        self.mu = self.mu * lik
        self.mu = self.mu / max(1e-12, np.sum(self.mu))

        xbar = self.mu[0]*self.x_list[0] + self.mu[1]*self.x_list[1]
        Pbar = np.zeros_like(self.P_list[0])
        for m in range(2):
            dx = (self.x_list[m] - xbar).reshape(-1, 1)
            Pbar += self.mu[m] * (self.P_list[m] + dx @ dx.T)

        self.miss_streak = 0
        return xbar, Pbar

    def miss(self):
        self.miss_streak += 1
        s = min(self.miss_streak, P_INFLATE_CAP)
        grow = (P_INFLATE_GROW ** s)
        for m in range(2):
            self.P_list[m][0:3, 0:3] += np.eye(3) * (P_INFLATE_POS * grow)
            self.P_list[m][3:6, 3:6] += np.eye(3) * (P_INFLATE_VEL * grow)
            # ✅ important: damping INSIDE the loop
            self.x_list[m][3:6] *= 0.97

    def hard_reset_to_meas(self, z):
        for m in range(2):
            self.x_list[m][0:3] = z
            self.P_list[m][0:3, 0:3] = np.eye(3) * (SIGMA_Z**2 * 4.0)
            self.P_list[m][3:6, 3:6] = np.eye(3) * (50.0)
        self.miss_streak = 0


# =========================
# GNN association (Hungarian)
# =========================
def run_gnn(truth, meas_list):
    Tt, n, _ = truth.shape
    tracks = []
    for i in range(n):
        x0 = truth[0, i].copy()
        P0 = np.diag([
            250.0**2, 250.0**2, 120.0**2,   # pos var
            25.0**2,  25.0**2,  12.0**2     # vel var
        ]).astype(float)
        tracks.append(TrackKF(tid=i, x=x0, P=P0, q=0.8))

    est = np.zeros((Tt, n, 3), float)
    assoc_tid = -2 * np.ones((Tt, n), int)

    for k in range(Tt):
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
                tracks[i].miss()
                assoc_tid[k, i] = -2

        if ENABLE_REACQUIRE and M > 0:
            for i in range(n):
                if tracks[i].miss_streak >= REACQUIRE_AFTER_MISSES:
                    best_j, best_d2 = None, float("inf")
                    for j in range(M):
                        innov = Z[j] - zpred[i]
                        d2 = maha2(innov, Spred[i])
                        if d2 < best_d2:
                            best_d2, best_j = d2, j
                    # still use mahalanobis for GNN reacquire
                    if best_j is not None and best_d2 <= 30.0:
                        tracks[i].hard_reset_to_meas(Z[best_j])
                        assoc_tid[k, i] = frame[best_j].tid

        for i in range(n):
            est[k, i] = tracks[i].x[:3]

    return est, assoc_tid


# =========================
# MHT / IMM-MHT (approx top-K)
# =========================
@dataclass
class Hypothesis:
    tracks: List
    logw: float
    assoc: List[int]   # per track: measurement index, -1 miss
    used_meas: set


def clone_track(track):
    if isinstance(track, TrackKF):
        return TrackKF(track.tid, track.x.copy(), track.P.copy(), track.q, track.miss_streak)
    elif isinstance(track, TrackIMM):
        return TrackIMM(
            track.tid,
            [x.copy() for x in track.x_list],
            [P.copy() for P in track.P_list],
            track.mu.copy(),
            track.Pi.copy(),
            list(track.q_list),
            track.miss_streak,
        )
    else:
        raise TypeError("Unknown track type")


def _track_xyz(track):
    if isinstance(track, TrackKF):
        return track.x[:3]
    else:
        xbar = track.mu[0]*track.x_list[0] + track.mu[1]*track.x_list[1]
        return xbar[:3]


def run_mht_like(truth, meas_list, use_imm=False, K_keep=35, branch_per_track=4):
    """
    Approx MHT:
      - For each hypothesis, sequentially branch per track with top candidates + miss
      - Score uses log-likelihood ratio (target vs clutter):
          log(PD) + log N(innov;0,S) - log(lambda/V)
        miss:
          log(1-PD)  (optionally -extra_penalty)
    """
    Tt, n, _ = truth.shape

    # Important: variance-scale P0 (not std-scale)
    P0_big = np.diag([
        250.0**2, 250.0**2, 120.0**2,
        25.0**2,  25.0**2,  12.0**2
    ]).astype(float)

    base_tracks = []
    for i in range(n):
        x0 = truth[0, i].copy()
        if not use_imm:
            base_tracks.append(TrackKF(tid=i, x=x0, P=P0_big.copy(), q=0.65))
        else:
            mu0 = np.array([0.7, 0.3], float)
            Pi = np.array([[0.95, 0.05],
                           [0.08, 0.92]], float)
            base_tracks.append(TrackIMM(
                tid=i,
                x_list=[x0.copy(), x0.copy()],
                P_list=[P0_big.copy(), P0_big.copy()],
                mu=mu0,
                Pi=Pi,
                q_list=[0.25, 2.2],
            ))

    hyps = [Hypothesis(tracks=[clone_track(t) for t in base_tracks],
                       logw=0.0,
                       assoc=[-1]*n,
                       used_meas=set())]

    est = np.zeros((Tt, n, 3), float)
    assoc_tid = -2 * np.ones((Tt, n), int)

    # precompute constants
    log_pd = math.log(max(1e-12, PD))
    log_miss = math.log(max(1e-12, 1.0 - PD))
    log_clutter = math.log(max(1e-12, CLUTTER_DENSITY))

    # optional: extra penalty to discourage endless misses (tune 0~3)
    EXTRA_MISS_PENALTY = 0.0

    for k in range(Tt):
        frame = meas_list[k]
        Z = np.array([m.z for m in frame], float) if len(frame) else np.zeros((0, 3), float)
        M = len(frame)

        new_hyps = []

        for h in hyps:
            # predict for each track in hypothesis
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

                    # --- measurement branches ---
                    for (j, _d2) in cand:
                        new_tracks = [clone_track(t) for t in ph.tracks]
                        tr_i = new_tracks[i]

                        innov = Z[j] - zpred[i]
                        llr = log_pd + log_gauss_pdf(innov, Spred[i]) - log_clutter

                        tr_i.update(Z[j])

                        nh = Hypothesis(
                            tracks=new_tracks,
                            logw=ph.logw + llr,
                            assoc=ph.assoc.copy(),
                            used_meas=set(ph.used_meas),
                        )
                        nh.assoc[i] = j
                        nh.used_meas.add(j)
                        next_partial.append(nh)

                    # --- miss branch ---
                    miss_tracks = [clone_track(t) for t in ph.tracks]
                    miss_tracks[i].miss()

                    nh = Hypothesis(
                        tracks=miss_tracks,
                        logw=ph.logw + (log_miss - EXTRA_MISS_PENALTY),
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

        # optional reacquire on the best hypothesis only
        if ENABLE_REACQUIRE and M > 0:
            for i in range(n):
                tr = best.tracks[i]
                if tr.miss_streak >= REACQUIRE_AFTER_MISSES:
                    x_pred = _track_xyz(tr)
                    best_j, best_d2e = None, float("inf")
                    for j in range(M):
                        d2e = float(np.sum((Z[j] - x_pred) ** 2))
                        if d2e < best_d2e:
                            best_d2e, best_j = d2e, j
                    if best_j is not None and best_d2e <= (REACQUIRE_EUCLIDEAN_R ** 2):
                        tr.hard_reset_to_meas(Z[best_j])
                        best.assoc[i] = best_j

        for i in range(n):
            if best.assoc[i] >= 0 and best.assoc[i] < M:
                assoc_tid[k, i] = frame[best.assoc[i]].tid
            else:
                assoc_tid[k, i] = -2
            est[k, i] = _track_xyz(best.tracks[i])

    return est, assoc_tid


# =========================
# Metrics: matched RMSE + fixedID + IDs
# =========================
def per_frame_assignment(truth_xyz: np.ndarray, est_xyz: np.ndarray):
    Tt, N, _ = truth_xyz.shape
    assign = np.zeros((Tt, N), dtype=int)
    for k in range(Tt):
        diff = truth_xyz[k, :, None, :] - est_xyz[k, None, :, :]
        C = np.sum(diff**2, axis=2)
        r, c = linear_sum_assignment(C)
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
    Tt, N = assign.shape
    sw = 0
    for k in range(1, Tt):
        sw += int(np.sum(assign[k] != assign[k-1]))
    return sw


def assoc_error_miss_rates(assoc_tid: np.ndarray, det_flag: np.ndarray):
    Tt, N = assoc_tid.shape
    wrong, total_updates = 0, 0
    for k in range(Tt):
        for i in range(N):
            if assoc_tid[k, i] != -2:
                total_updates += 1
                if assoc_tid[k, i] != i:
                    wrong += 1
    err_rate = 100.0 * wrong / max(1, total_updates)

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


def compute_all_metrics(truth_states, est_xyz, assoc_tid, det_flag):
    truth_xyz = truth_states[:, :, :3]

    rmse_m = rmse_matched(truth_xyz, est_xyz)
    armse_m = float(np.mean(rmse_m))

    assign = per_frame_assignment(truth_xyz, est_xyz)
    sw = idswitch_count(assign)
    sw_rate = 100.0 * sw / max(1, (truth_xyz.shape[0] - 1) * truth_xyz.shape[1])

    rmse_f = rmse_fixed_id(truth_xyz, est_xyz)
    armse_f = float(np.mean(rmse_f))

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


# =========================
# Plotting
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


def plot_rmse_two_curves(rmse_matched_t, rmse_fixed_t, title, out_path):
    plt.figure(figsize=(9.2, 4.6))
    plt.plot(np.arange(len(rmse_matched_t)), rmse_matched_t, linewidth=2.4)
    plt.plot(np.arange(len(rmse_fixed_t)), rmse_fixed_t, linewidth=2.0, linestyle="--")
    plt.title(title)
    plt.xlabel("time step")
    plt.ylabel("RMSE (3D position)")
    plt.grid(True, alpha=0.3)

    ax = plt.gca()
    ax.ticklabel_format(style='plain', axis='y', useOffset=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# =========================
# Summary utilities
# =========================
def print_table(rows: List[Dict]):
    headers = ["Algorithm", "ARMSE_matched", "ARMSE_fixedID", "SimTime(s)",
               "ErrorRate(%)", "MissRate(%)", "IDSwitches", "IDSwitchRate(%)"]
    colw = [12, 15, 15, 12, 14, 13, 12, 16]

    line = "".join(h.ljust(w) for h, w in zip(headers, colw))
    print(line)
    print("-" * sum(colw))

    for r in rows:
        vals = [
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
# Run Scene2 only
# =========================
def run_scene2():
    truth2, params2 = gen_scene2_truth()
    save_params("Scene2", params2)

    meas2, det2 = generate_measurements(truth2, seed=SEED + 1)

    results_rows = []

    # ---- GNN ----
    t0 = time.perf_counter()
    est_gnn, assoc_gnn = run_gnn(truth2, meas2)
    t1 = time.perf_counter()
    m = compute_all_metrics(truth2, est_gnn, assoc_gnn, det2)

    plot_assoc_3d(truth2, est_gnn, meas2,
                  "Scene2 | GNN | 3D Truth vs Tracking",
                  os.path.join(OUT_DIR, "Scene2_GNN_assoc3d.png"))
    plot_rmse_two_curves(m["rmse_matched_t"], m["rmse_fixed_t"],
                         "Scene2 | GNN | RMSE(t) Matched(solid) vs FixedID(dashed)",
                         os.path.join(OUT_DIR, "Scene2_GNN_rmse.png"))

    results_rows.append({
        "Scene": "Scene2",
        "Algorithm": "GNN",
        "ARMSE_matched": m["armse_matched"],
        "ARMSE_fixedID": m["armse_fixed"],
        "SimTime(s)": (t1 - t0),
        "ErrorRate(%)": m["error_rate"],
        "MissRate(%)": m["miss_rate"],
        "IDSwitches": m["idswitches"],
        "IDSwitchRate(%)": m["idswitch_rate"],
    })

    # ---- MHT ----
    t0 = time.perf_counter()
    est_mht, assoc_mht = run_mht_like(truth2, meas2, use_imm=False, K_keep=35, branch_per_track=4)
    t1 = time.perf_counter()
    m = compute_all_metrics(truth2, est_mht, assoc_mht, det2)

    plot_assoc_3d(truth2, est_mht, meas2,
                  "Scene2 | MHT | 3D Truth vs Tracking",
                  os.path.join(OUT_DIR, "Scene2_MHT_assoc3d.png"))
    plot_rmse_two_curves(m["rmse_matched_t"], m["rmse_fixed_t"],
                         "Scene2 | MHT | RMSE(t) Matched(solid) vs FixedID(dashed)",
                         os.path.join(OUT_DIR, "Scene2_MHT_rmse.png"))

    results_rows.append({
        "Scene": "Scene2",
        "Algorithm": "MHT",
        "ARMSE_matched": m["armse_matched"],
        "ARMSE_fixedID": m["armse_fixed"],
        "SimTime(s)": (t1 - t0),
        "ErrorRate(%)": m["error_rate"],
        "MissRate(%)": m["miss_rate"],
        "IDSwitches": m["idswitches"],
        "IDSwitchRate(%)": m["idswitch_rate"],
    })

    # ---- IMM-MHT ----
    t0 = time.perf_counter()
    est_im, assoc_im = run_mht_like(truth2, meas2, use_imm=True, K_keep=35, branch_per_track=4)
    t1 = time.perf_counter()
    m = compute_all_metrics(truth2, est_im, assoc_im, det2)

    plot_assoc_3d(truth2, est_im, meas2,
                  "Scene2 | IMM-MHT | 3D Truth vs Tracking",
                  os.path.join(OUT_DIR, "Scene2_IMM-MHT_assoc3d.png"))
    plot_rmse_two_curves(m["rmse_matched_t"], m["rmse_fixed_t"],
                         "Scene2 | IMM-MHT | RMSE(t) Matched(solid) vs FixedID(dashed)",
                         os.path.join(OUT_DIR, "Scene2_IMM-MHT_rmse.png"))

    results_rows.append({
        "Scene": "Scene2",
        "Algorithm": "IMM-MHT",
        "ARMSE_matched": m["armse_matched"],
        "ARMSE_fixedID": m["armse_fixed"],
        "SimTime(s)": (t1 - t0),
        "ErrorRate(%)": m["error_rate"],
        "MissRate(%)": m["miss_rate"],
        "IDSwitches": m["idswitches"],
        "IDSwitchRate(%)": m["idswitch_rate"],
    })

    # print & save summary
    print("\n=== Scene2 Metrics Summary ===")
    print_table(results_rows)

    out_csv = os.path.join(OUT_DIR, "summary_metrics_scene2.csv")
    save_summary_csv(results_rows, out_csv)
    print(f"\n[saved] {out_csv}")
    print(f"[saved figures] -> {OUT_DIR}/")
    print("  - 6 figures: Scene2 × {GNN,MHT,IMM-MHT} × {assoc3d,rmse}")
    print("  - params: Scene2_params.(csv/json)")


def main():
    run_scene2()


if __name__ == "__main__":
    main()