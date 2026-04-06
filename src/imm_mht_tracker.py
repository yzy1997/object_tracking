# src/imm_mht_tracker.py
# -*- coding: utf-8 -*-

import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
from scipy.optimize import linear_sum_assignment


# -----------------------------
# Kalman CV model
# state: [cx, cy, vx, vy]
# meas : [cx, cy]
# -----------------------------
class KalmanCV:
    def __init__(self, dt=1.0, q_pos=1.0, q_vel=0.5, r_pos=6.0):
        self.dt = float(dt)

        self.x = np.zeros((4, 1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 500.0

        self.F = np.array(
            [[1, 0, self.dt, 0],
             [0, 1, 0, self.dt],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32
        )
        self.H = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32
        )

        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float32)
        self.R = np.diag([r_pos, r_pos]).astype(np.float32)
        self.I = np.eye(4, dtype=np.float32)

    def init_from_meas(self, cx, cy, vx=0.0, vy=0.0):
        self.x[:] = np.array([[cx], [cy], [vx], [vy]], dtype=np.float32)
        self.P[:] = np.eye(4, dtype=np.float32) * 50.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def innovation(self, z_xy: np.ndarray):
        z = z_xy.reshape(2, 1).astype(np.float32)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        return y, S

    def update(self, z_xy: np.ndarray):
        y, S = self.innovation(z_xy)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (self.I - K @ self.H) @ self.P
        return self.x


# -----------------------------
# IMM Filter (2x CV with different process noise)
# model 0: "smooth"  (small q)
# model 1: "agile"   (larger q)
# -----------------------------
class IMMFilter:
    def __init__(
        self,
        dt=1.0,
        r_pos=5.0,
        q0=(0.4, 0.1),   # (q_pos, q_vel) smooth - 更平滑
        q1=(1.5, 0.7),  # (q_pos, q_vel) agile - 降低灵活度
        trans_mat=None,  # model transition
        mu0=None         # initial mode prob
    ):
        self.dt = float(dt)
        self.models = [
            KalmanCV(dt=dt, q_pos=q0[0], q_vel=q0[1], r_pos=r_pos),
            KalmanCV(dt=dt, q_pos=q1[0], q_vel=q1[1], r_pos=r_pos),
        ]
        self.M = 2

        # 更高的自转移概率，保持模型稳定
        if trans_mat is None:
            self.PI = np.array([[0.95, 0.05],
                                [0.05, 0.95]], dtype=np.float32)
        else:
            self.PI = np.array(trans_mat, dtype=np.float32)

        if mu0 is None:
            # 初始偏向平滑模型
            self.mu = np.array([0.7, 0.3], dtype=np.float32)
        else:
            self.mu = np.array(mu0, dtype=np.float32)

        # fused state
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 500.0

    def init_from_meas(self, cx, cy, vx=0.0, vy=0.0):
        for m in self.models:
            m.init_from_meas(cx, cy, vx, vy)
        self._fuse()

    def _mix(self):
        # mixing probabilities
        c = self.PI.T @ self.mu  # normalizers for each destination model
        c = np.maximum(c, 1e-12)

        mu_ij = (self.PI * self.mu[None, :]).T / c[None, :]   # shape (i->j): (2,2)
        # mu_ij[i,j] = prob from i to j (after normalization)

        mixed = []
        for j in range(self.M):
            # mixed initial for model j
            xj = np.zeros((4, 1), dtype=np.float32)
            for i in range(self.M):
                xj += mu_ij[i, j] * self.models[i].x
            Pj = np.zeros((4, 4), dtype=np.float32)
            for i in range(self.M):
                dx = self.models[i].x - xj
                Pj += mu_ij[i, j] * (self.models[i].P + dx @ dx.T)
            mixed.append((xj, Pj))
        return mixed, c

    def predict(self):
        mixed, c = self._mix()

        # set mixed into models then predict
        for j in range(self.M):
            self.models[j].x = mixed[j][0].copy()
            self.models[j].P = mixed[j][1].copy()
            self.models[j].predict()

        # update mode prob only after measurement; keep c for update step
        self._fuse()
        return self.x

    @staticmethod
    def _gaussian_likelihood(y: np.ndarray, S: np.ndarray) -> float:
        # y: (2,1) , S: (2,2)
        try:
            Sinv = np.linalg.inv(S)
            v = float((y.T @ Sinv @ y)[0, 0])
            detS = float(np.linalg.det(S))
            detS = max(detS, 1e-12)
            # (2D) likelihood ~ exp(-0.5 v) / sqrt(detS)
            return float(np.exp(-0.5 * v) / np.sqrt(detS))
        except Exception:
            return 1e-12

    def update(self, z_xy: np.ndarray):
        # compute likelihoods for each model
        lamb = np.zeros((self.M,), dtype=np.float32)
        for j in range(self.M):
            y, S = self.models[j].innovation(z_xy)
            lamb[j] = self._gaussian_likelihood(y, S)

        # mode prob update: mu_bar = (PI^T mu) * lambda  (standard IMM)
        mu_bar = (self.PI.T @ self.mu) * lamb
        s = float(np.sum(mu_bar))
        if s <= 1e-12:
            mu_bar = np.array([0.5, 0.5], dtype=np.float32)
        else:
            mu_bar = mu_bar / s
        self.mu = mu_bar.astype(np.float32)

        # update each model with measurement
        for j in range(self.M):
            self.models[j].update(z_xy)

        self._fuse()
        return self.x

    def _fuse(self):
        # fused mean
        x = np.zeros((4, 1), dtype=np.float32)
        for j in range(self.M):
            x += self.mu[j] * self.models[j].x
        # fused covariance
        P = np.zeros((4, 4), dtype=np.float32)
        for j in range(self.M):
            dx = self.models[j].x - x
            P += self.mu[j] * (self.models[j].P + dx @ dx.T)

        self.x, self.P = x, P


# -----------------------------
# Track
# -----------------------------
@dataclass
class IMMTrack:
    track_id: int
    imm: IMMFilter
    w: float
    h: float
    hits: int = 0
    missed: int = 0
    confirmed: bool = False
    last_update_frame: int = -1
    total_updates: int = 0
    total_predictions: int = 0

    # for velocity consistency (optional)
    last_meas_xy: Optional[np.ndarray] = None

    # for direction consistency - 记录历史方向
    direction_history: Tuple[float, ...] = ()  # 存储最近的方向角


@dataclass
class Hypothesis:
    tracks: List[IMMTrack]
    log_score: float
    signature: List[Tuple[int, Tuple[int, ...]]]  # (frame_id, assignments per-track in tid order)


# -----------------------------
# IMM-MHT Tracker
# -----------------------------
class IMMMHTTracker:
    """
    Multi-Hypothesis with IMM tracks.

    Key improvements:
      - Mahalanobis gating using fused (IMM) covariance
      - cost = sqrt(mahal) + lambda_v * velocity_consistency + lambda_wh * size_consistency
      - birth/miss penalties tuned for fixed-#targets association
      - constrain confirmed track count (default 4) to prevent track explosion
    """

    def __init__(
        self,
        dt=1.0,
        gating_chi2=4.0,       # 2D chi-square 90% (更严格，减少误跟)
        max_hypotheses=30,    # 增加假设数量
        n_scan=5,             # 增加N-scan深度
        max_missed=6,         # 更严格：减少最大丢失帧数
        min_hits_to_confirm=3,# 增加确认所需命中次数
        max_confirmed=4,
        max_tracks_keep=10,
        miss_penalty=10.0,    # 更高惩罚
        birth_penalty=20.0,   # 更高出生惩罚
        lambda_v=0.8,         # 更高的速度一致性权重
        lambda_wh=0.1,        # 增加尺寸一致性权重
        r_pos=4.0,            # 更信任测量值
        imm_q0=(0.3, 0.08),   # 更平滑
        imm_q1=(1.0, 0.5),    # 降低灵活度
        wh_smooth=0.8,        # 更高的尺寸平滑
    ):
        self.dt = float(dt)
        self.gating_chi2 = float(gating_chi2)
        self.max_hypotheses = int(max_hypotheses)
        self.n_scan = int(n_scan)
        self.max_missed = int(max_missed)
        self.min_hits_to_confirm = int(min_hits_to_confirm)

        self.max_confirmed = int(max_confirmed)
        self.max_tracks_keep = int(max_tracks_keep)

        self.miss_penalty = float(miss_penalty)
        self.birth_penalty = float(birth_penalty)

        self.lambda_v = float(lambda_v)
        self.lambda_wh = float(lambda_wh)
        self.r_pos = float(r_pos)
        self.imm_q0 = imm_q0
        self.imm_q1 = imm_q1
        self.wh_smooth = float(wh_smooth)

        self.hypotheses: List[Hypothesis] = []

    def _new_imm(self) -> IMMFilter:
        return IMMFilter(
            dt=self.dt,
            r_pos=self.r_pos,
            q0=self.imm_q0,
            q1=self.imm_q1,
        )

    def initialize(self, tracks: List[IMMTrack]):
        self.hypotheses = [Hypothesis(tracks=copy.deepcopy(tracks), log_score=0.0, signature=[])]

    # ---------- gating & cost ----------
    def _mahalanobis2(self, tr: IMMTrack, z_xy: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Return (mahal^2, innovation y, innovation cov S)
        using fused IMM covariance.
        """
        # Hx -> predicted measurement
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=np.float32)
        x = tr.imm.x
        P = tr.imm.P
        z = z_xy.reshape(2, 1).astype(np.float32)
        y = z - (H @ x)
        S = H @ P @ H.T + np.diag([self.r_pos, self.r_pos]).astype(np.float32)
        try:
            Sinv = np.linalg.inv(S)
            m2 = float((y.T @ Sinv @ y)[0, 0])
        except Exception:
            m2 = 1e9
        return m2, y, S

    def _velocity_cost(self, tr: IMMTrack, z_xy: np.ndarray) -> float:
        """
        Encourage consistency between predicted velocity and measurement displacement.
        """
        if tr.last_meas_xy is None:
            return 0.0
        # predicted velocity
        vx = float(tr.imm.x[2, 0])
        vy = float(tr.imm.x[3, 0])
        pred_disp = np.array([vx * self.dt, vy * self.dt], dtype=np.float32)
        meas_disp = (z_xy.astype(np.float32) - tr.last_meas_xy.astype(np.float32))

        # 如果速度太小，忽略方向检查
        pred_norm = np.linalg.norm(pred_disp)
        if pred_norm < 0.5:
            return 0.0

        # 方向一致性：计算角度差
        pred_dir = pred_disp / pred_norm
        meas_norm = np.linalg.norm(meas_disp)
        if meas_norm < 0.1:
            return 5.0  # 测量点几乎没动，惩罚

        meas_dir = meas_disp / meas_norm

        # 方向夹角余弦值（-1到1），1表示同方向
        cos_angle = np.dot(pred_dir, meas_dir)
        # 转换为一个惩罚值：方向越一致惩罚越小
        # cos_angle=1时惩罚为0，cos_angle=-1时惩罚为2
        dir_penalty = 1.0 - cos_angle

        # 同时检查速度大小一致性
        speed_ratio = meas_norm / max(pred_norm, 0.1)
        speed_penalty = abs(speed_ratio - 1.0)

        return float(dir_penalty * 2.0 + speed_penalty * 0.5)

    def _size_cost(self, tr: IMMTrack, det_xywh) -> float:
        _, _, w, h = det_xywh
        return float(abs(float(w) - tr.w) + abs(float(h) - tr.h))

    def _build_cost_matrix(self, tracks: List[IMMTrack], detections_xywh: List[Tuple[float, float, float, float]]):
        if not tracks or not detections_xywh:
            return None, None

        det_centers = []
        for (x, y, w, h) in detections_xywh:
            det_centers.append(np.array([x + w / 2.0, y + h / 2.0], dtype=np.float32))
        det_centers = np.stack(det_centers, axis=0)  # (M,2)

        INF = 1e6
        cost = np.full((len(tracks), len(detections_xywh)), INF, dtype=np.float32)

        for i, tr in enumerate(tracks):
            for j, det in enumerate(detections_xywh):
                z = det_centers[j]
                m2, _, _ = self._mahalanobis2(tr, z)
                if m2 > self.gating_chi2:
                    continue

                c = np.sqrt(max(m2, 0.0))
                # add velocity consistency
                c += self.lambda_v * self._velocity_cost(tr, z)
                # add size consistency (weak)
                c += self.lambda_wh * self._size_cost(tr, det)
                cost[i, j] = float(c)

        return cost, det_centers

    # ---------- scoring helpers ----------
    def _adaptive_miss_penalty(self, tr: IMMTrack) -> float:
        # more uncertainty -> less penalty
        P = tr.imm.P
        pos_var = float(P[0, 0] + P[1, 1])
        # scale: when pos_var large => reduce penalty
        scale = 1.0 / (1.0 + 0.02 * pos_var)
        return self.miss_penalty * scale

    def _predict_tracks(self, tracks: List[IMMTrack]):
        for tr in tracks:
            tr.imm.predict()
            tr.missed += 1
            tr.total_predictions += 1

    def _update_track(self, tr: IMMTrack, frame_id: int, det_xywh, z_xy: np.ndarray):
        # 记录方向用于一致性检查
        if tr.last_meas_xy is not None:
            disp = z_xy - tr.last_meas_xy
            if np.linalg.norm(disp) > 0.5:
                # 计算方向角（弧度）
                direction = np.arctan2(float(disp[1]), float(disp[0]))
                # 保留最近5帧方向
                tr.direction_history = (tr.direction_history + (direction,))[-5:]

        tr.imm.update(z_xy)
        tr.missed = 0
        tr.hits += 1
        tr.last_update_frame = frame_id
        tr.total_updates += 1

        # update bbox size with smoothing
        _, _, w_meas, h_meas = det_xywh
        tr.w = (1.0 - self.wh_smooth) * tr.w + self.wh_smooth * float(w_meas)
        tr.h = (1.0 - self.wh_smooth) * tr.h + self.wh_smooth * float(h_meas)

        tr.last_meas_xy = z_xy.copy()

        if (not tr.confirmed) and tr.hits >= self.min_hits_to_confirm:
            tr.confirmed = True

    def create_track(self, next_id: int, frame_id: int, det_xywh) -> IMMTrack:
        x, y, w, h = det_xywh
        cx = float(x + w / 2.0)
        cy = float(y + h / 2.0)

        imm = self._new_imm()
        imm.init_from_meas(cx, cy, vx=0.0, vy=0.0)

        tr = IMMTrack(
            track_id=next_id,
            imm=imm,
            w=float(w),
            h=float(h),
            hits=1,
            missed=0,
            confirmed=False,
            last_update_frame=frame_id,
            last_meas_xy=np.array([cx, cy], dtype=np.float32)
        )
        return tr

    # ---------- MHT step ----------
    def step(self, detections_xywh, frame_id: int, next_id: int):
        new_hyps: List[Hypothesis] = []

        for hyp in self.hypotheses:
            tracks = copy.deepcopy(hyp.tracks)

            # 1) predict
            self._predict_tracks(tracks)

            # 2) assignment
            cost, det_centers = self._build_cost_matrix(tracks, detections_xywh)
            matched_det = set()

            log_score = float(hyp.log_score)
            assignment_map: Dict[int, int] = {}  # tid -> det_index, -1 if miss

            if cost is not None and cost.size > 0:
                row, col = linear_sum_assignment(cost)

                for r, c in zip(row, col):
                    if cost[r, c] >= 1e5:
                        continue
                    det = detections_xywh[c]
                    z = det_centers[c]

                    # update
                    self._update_track(tracks[r], frame_id, det, z)

                    log_score -= float(cost[r, c])
                    matched_det.add(int(c))
                    assignment_map[int(tracks[r].track_id)] = int(c)

            # 3) misses penalty (for those not updated)
            for tr in tracks:
                if tr.last_update_frame != frame_id:
                    assignment_map[int(tr.track_id)] = -1
                    log_score -= self._adaptive_miss_penalty(tr)

            # 4) births for unmatched detections
            for i, det in enumerate(detections_xywh):
                if i in matched_det:
                    continue

                # birth control: if confirmed already saturated, be stricter
                confirmed_cnt = sum(1 for t in tracks if t.confirmed)
                if confirmed_cnt >= self.max_confirmed:
                    # allow birth only if it's likely NOT close to any confirmed track (avoid duplicates)
                    z = np.array([det[0] + det[2] / 2.0, det[1] + det[3] / 2.0], dtype=np.float32)
                    too_close = False
                    for t in tracks:
                        if not t.confirmed:
                            continue
                        m2, _, _ = self._mahalanobis2(t, z)
                        if m2 <= self.gating_chi2:
                            too_close = True
                            break
                    if too_close:
                        # treat as clutter
                        log_score -= 1.0
                        continue

                tr = self.create_track(next_id, frame_id, det)
                tracks.append(tr)
                next_id += 1
                log_score -= self.birth_penalty

            # 5) delete dead tracks
            tracks = [t for t in tracks if t.missed <= self.max_missed]

            # 6) global cap (avoid explosion)
            if len(tracks) > self.max_tracks_keep:
                # keep best by confirmed first, then hits, then lowest missed
                tracks.sort(key=lambda t: (t.confirmed, t.hits, -t.missed), reverse=True)
                tracks = tracks[:self.max_tracks_keep]

            # build signature for N-scan pruning
            tid_sorted = sorted([t.track_id for t in tracks])
            assign_tuple = tuple(assignment_map.get(tid, -1) for tid in tid_sorted)
            sig = hyp.signature + [(frame_id, assign_tuple)]

            new_hyps.append(Hypothesis(tracks=tracks, log_score=log_score, signature=sig))

        # 7) prune by score
        new_hyps.sort(key=lambda h: h.log_score, reverse=True)
        self.hypotheses = new_hyps[:self.max_hypotheses]

        # 8) N-scan pruning: keep hypotheses sharing same prefix signature
        if len(self.hypotheses) > 1 and len(self.hypotheses[0].signature) > self.n_scan:
            ref_prefix = tuple(self.hypotheses[0].signature[:-self.n_scan])
            kept = []
            for h in self.hypotheses:
                if tuple(h.signature[:-self.n_scan]) == ref_prefix:
                    kept.append(h)
            if kept:
                self.hypotheses = kept

        best = self.hypotheses[0]
        return best.tracks, next_id


__all__ = ["KalmanCV", "IMMFilter", "IMMTrack", "IMMMHTTracker"]
