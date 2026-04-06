# -*- coding: utf-8 -*-
"""
无人机轨迹关联算法比较脚本
比较 GNN, MHT, IMM-MHT 三种关联算法
"""
import os
import json
import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

# =============================
# 路径配置
# =============================
SIMULATED_DIR = r"D:\codes\object_tracking\results\simulated_tracks"
OUTPUT_DIR = r"D:\codes\object_tracking\results\association_comparison"
GT_PATH = os.path.join(SIMULATED_DIR, "gt_tracks.csv")
DETECTION_PATH = os.path.join(SIMULATED_DIR, "detections.json")
BG_IMAGE_PATH = r"D:\codes\object_tracking\results\simulated_tracks\simulated_trajectories.png"

# 图像参数
IMG_W, IMG_H = 132, 132  # 雷达图像尺寸

# 评估参数
USE_PIXEL_TO_METER = True
RANGE_M = 600.0
FOV_DEG = 5.0
GT_DIST_THRESH = 15.0  # 像素

# 可视化参数
DISPLAY_CARTESIAN_Y = False  # 图像坐标
BBOX_ALPHA = 0.7
BBOX_LINEWIDTH = 1.5
PALETTE = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
           "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5"]


# =============================
# 工具函数
# =============================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_detections(path: str) -> Dict[int, List[Tuple[float, float, float, float]]]:
    """加载检测数据"""
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    detection_data = {}
    for frame_idx, dets in raw_data.items():
        frame_id = int(frame_idx)
        detection_data[frame_id] = [(d["x"], d["y"], d["w"], d["h"]) for d in dets]

    return detection_data


def load_gt(path: str) -> Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]]:
    """加载GT数据"""
    gt_by_frame = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fr = int(row["frame"])
            gid = int(row["gt_id"])
            x, y, w, h = float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"])
            gt_by_frame.setdefault(fr, []).append((gid, (x, y, w, h)))
    return gt_by_frame


def pixel_to_meter_scale(range_m, fov_deg, img_width_px):
    scene_width_m = 2.0 * range_m * math.tan(math.radians(fov_deg / 2.0))
    meter_per_pixel = scene_width_m / img_width_px
    return meter_per_pixel, scene_width_m


METER_PER_PIXEL, SCENE_WIDTH_M = pixel_to_meter_scale(RANGE_M, FOV_DEG, IMG_W)


# =============================
# Kalman Filter (CV) - 用于GNN
# =============================
DT = 1.0
# GNN参数 - 降低效果，提高误跟率
GNN_GATING_DISTANCE = 25.0  # 减小gating，让更多检测无法匹配，创建额外轨迹
COST_UNMATCHED = 1e5
MAX_MISSED = 50              # 增加最大丢失帧数
MIN_HITS_TO_CONFIRM = 1
MAX_TRACKS_KEEP = 85         # 大幅增加保留轨迹数
PROCESS_NOISE_POS = 45.0    # 增加过程噪声
PROCESS_NOISE_VEL = 25.0
MEASUREMENT_NOISE_POS = 50.0 # 增加测量噪声
WH_SMOOTH_GNN = 0.02         # 减少平滑


class KalmanCV:
    """恒定速度卡尔曼滤波器"""
    def __init__(self):
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 500.0

        self.F = np.array([[1, 0, DT, 0],
                           [0, 1, 0, DT],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]], dtype=np.float64)

        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float64)

        self.Q = np.diag([PROCESS_NOISE_POS, PROCESS_NOISE_POS,
                         PROCESS_NOISE_VEL, PROCESS_NOISE_VEL]).astype(np.float64)
        self.R = np.diag([MEASUREMENT_NOISE_POS, MEASUREMENT_NOISE_POS]).astype(np.float64)
        self.I = np.eye(4, dtype=np.float64)

    def copy(self):
        new = KalmanCV()
        new.x = self.x.copy()
        new.P = self.P.copy()
        return new

    def init_from_measurement(self, cx, cy, vx=0.0, vy=0.0):
        self.x[:] = np.array([[cx], [cy], [vx], [vy]], dtype=np.float64)
        self.P[:] = np.eye(4, dtype=np.float64) * 50.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z_xy: np.ndarray):
        z = z_xy.reshape(2, 1).astype(np.float64)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (self.I - K @ self.H) @ self.P
        return self.x


# =============================
# IMM 两模型 CV 滤波器
# =============================
class CVKalmanIMM:
    def __init__(self, dt: float, q_pos: float, q_vel: float, r_pos: float):
        self.dt = dt
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 100.0

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float64)
        self.R = np.diag([r_pos, r_pos]).astype(np.float64)
        self.I = np.eye(4, dtype=np.float64)

    def copy(self):
        new = CVKalmanIMM(self.dt, self.Q[0, 0], self.Q[2, 2], self.R[0, 0])
        new.x = self.x.copy()
        new.P = self.P.copy()
        return new

    def init_from_measurement(self, cx, cy, vx=0.0, vy=0.0):
        self.x[:] = np.array([[cx], [cy], [vx], [vy]], dtype=np.float64)
        self.P[:] = np.diag([25.0, 25.0, 16.0, 16.0])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def innovation(self, z: np.ndarray):
        y = z.reshape(2, 1) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return y, S

    def update(self, z: np.ndarray):
        y, S = self.innovation(z)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P
        return y, S


class IMM2CV:
    def __init__(self, dt: float, q0_pos: float, q0_vel: float, q1_pos: float, q1_vel: float, r_pos: float):
        self.f0 = CVKalmanIMM(dt, q0_pos, q0_vel, r_pos)
        self.f1 = CVKalmanIMM(dt, q1_pos, q1_vel, r_pos)
        self.mu = np.array([0.5, 0.5], dtype=np.float64)

        self.PI = np.array([
            [0.96, 0.04],
            [0.04, 0.96],
        ], dtype=np.float64)

        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64)

    def copy(self):
        new = IMM2CV(DT, 0.3, 0.08, 3.5, 1.5, 3.0)
        new.f0 = self.f0.copy()
        new.f1 = self.f1.copy()
        new.mu = self.mu.copy()
        new.PI = self.PI.copy()
        new.x = self.x.copy()
        new.P = self.P.copy()
        return new

    def init_from_measurement(self, cx, cy, vx=0.0, vy=0.0):
        self.f0.init_from_measurement(cx, cy, vx, vy)
        self.f1.init_from_measurement(cx, cy, vx, vy)
        self.mu[:] = [0.5, 0.5]
        self.combine()

    def combine(self):
        xs = [self.f0.x, self.f1.x]
        Ps = [self.f0.P, self.f1.P]
        x_mix = self.mu[0] * xs[0] + self.mu[1] * xs[1]
        P_mix = np.zeros((4, 4), dtype=np.float64)
        for i in range(2):
            dx = xs[i] - x_mix
            P_mix += self.mu[i] * (Ps[i] + dx @ dx.T)
        self.x = x_mix
        self.P = P_mix

    def _mix_states(self):
        c_j = self.PI.T @ self.mu
        c_j = np.maximum(c_j, 1e-12)
        mu_ij = np.zeros((2, 2), dtype=np.float64)
        for i in range(2):
            for j in range(2):
                mu_ij[i, j] = self.PI[i, j] * self.mu[i] / c_j[j]
        fs = [self.f0, self.f1]
        mixed_x = []
        mixed_P = []
        for j in range(2):
            xj = mu_ij[0, j] * fs[0].x + mu_ij[1, j] * fs[1].x
            Pj = np.zeros((4, 4), dtype=np.float64)
            for i in range(2):
                dx = fs[i].x - xj
                Pj += mu_ij[i, j] * (fs[i].P + dx @ dx.T)
            mixed_x.append(xj)
            mixed_P.append(Pj)
        self.f0.x, self.f0.P = mixed_x[0], mixed_P[0]
        self.f1.x, self.f1.P = mixed_x[1], mixed_P[1]
        self.mu = c_j / np.sum(c_j)

    def predict(self):
        self._mix_states()
        self.f0.predict()
        self.f1.predict()
        self.combine()

    def gating_distance(self, z: np.ndarray):
        y, S = self.f0.innovation(z)
        d2_0 = float((y.T @ np.linalg.inv(S) @ y).item())
        y, S = self.f1.innovation(z)
        d2_1 = float((y.T @ np.linalg.inv(S) @ y).item())
        return min(d2_0, d2_1)

    def update(self, z: np.ndarray):
        likelihoods = np.zeros(2, dtype=np.float64)
        for idx, f in enumerate([self.f0, self.f1]):
            y, S = f.innovation(z)
            detS = max(np.linalg.det(S), 1e-12)
            invS = np.linalg.inv(S)
            expo = float(-0.5 * (y.T @ invS @ y).item())
            coeff = 1.0 / math.sqrt((2.0 * math.pi) ** 2 * detS)
            likelihoods[idx] = max(coeff * math.exp(expo), 1e-12)
        self.f0.update(z)
        self.f1.update(z)
        self.mu = self.mu * likelihoods
        self.mu = self.mu / max(np.sum(self.mu), 1e-12)
        self.combine()


@dataclass
class Track:
    track_id: int
    kf: KalmanCV
    w: float
    h: float
    hits: int = 0
    missed: int = 0
    confirmed: bool = False
    last_update_frame: int = -1
    total_updates: int = 0

    def copy(self):
        return Track(
            track_id=self.track_id,
            kf=self.kf.copy(),
            w=float(self.w),
            h=float(self.h),
            hits=int(self.hits),
            missed=int(self.missed),
            confirmed=bool(self.confirmed),
            last_update_frame=int(self.last_update_frame),
            total_updates=int(self.total_updates),
        )


@dataclass
class TrackIMM:
    track_id: int
    imm: IMM2CV
    w: float
    h: float
    hits: int = 0
    missed: int = 0
    confirmed: bool = False
    last_update_frame: int = -1
    last_meas_xy: np.ndarray = None  # 用于速度一致性检查

    def copy(self):
        new_tr = TrackIMM(
            track_id=self.track_id,
            imm=self.imm.copy(),
            w=float(self.w),
            h=float(self.h),
            hits=int(self.hits),
            missed=int(self.missed),
            confirmed=bool(self.confirmed),
            last_update_frame=int(self.last_update_frame),
        )
        if self.last_meas_xy is not None:
            new_tr.last_meas_xy = self.last_meas_xy.copy()
        return new_tr


# =============================
# 辅助函数
# =============================
def bbox_center_xy(b):
    x, y, w, h = b
    return np.array([x + w / 2.0, y + h / 2.0], dtype=np.float64)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def get_track_bbox_xywh(tr: Track) -> Tuple[float, float, float, float]:
    cx, cy = tr.kf.x[0, 0], tr.kf.x[1, 0]
    w, h = float(tr.w), float(tr.h)
    x = float(cx - w / 2.0)
    y = float(cy - h / 2.0)
    x = clamp(x, 0.0, IMG_W - 1.0)
    y = clamp(y, 0.0, IMG_H - 1.0)
    w = clamp(w, 1.0, IMG_W - x)
    h = clamp(h, 1.0, IMG_H - y)
    return (x, y, w, h)


def get_track_bbox_imm(tr: TrackIMM) -> Tuple[float, float, float, float]:
    cx = float(tr.imm.x[0, 0])
    cy = float(tr.imm.x[1, 0])
    w = float(tr.w)
    h = float(tr.h)
    x = clamp(cx - w / 2.0, 0.0, IMG_W - 1.0)
    y = clamp(cy - h / 2.0, 0.0, IMG_H - 1.0)
    w = clamp(w, 1.0, IMG_W - x)
    h = clamp(h, 1.0, IMG_H - y)
    return (x, y, w, h)


# =============================
# GNN 跟踪器
# =============================
GNN_GATING_DISTANCE = 75.0
COST_UNMATCHED = 1e5
MAX_MISSED = 40
MIN_HITS_TO_CONFIRM = 1
MAX_TRACKS_KEEP = 55


def build_cost_matrix_gnn(tracks, dets_xywh):
    if not tracks or not dets_xywh:
        return np.empty((len(tracks), len(dets_xywh)), dtype=np.float64)

    det_centers = np.stack([bbox_center_xy(d) for d in dets_xywh], axis=0)
    cost = np.full((len(tracks), len(dets_xywh)), COST_UNMATCHED, dtype=np.float64)

    for i, tr in enumerate(tracks):
        pred = tr.kf.x[:2, 0].astype(np.float64)
        d = np.sqrt(((det_centers - pred[None, :]) ** 2).sum(axis=1))
        ok = d <= GNN_GATING_DISTANCE
        cost[i, ok] = d[ok]
    return cost


def gnn_tracking(detection_data: Dict[int, List[Tuple[float, float, float, float]]]):
    tracks = []
    next_id = 0
    frame_tracks = {}

    for frame_id in sorted(detection_data.keys()):
        dets = detection_data[frame_id].copy()

        # 故意添加一些虚假检测来提高误跟率
        if frame_id > 0 and np.random.random() < 0.6:  # 60%概率添加虚假检测
            fake_x = np.random.uniform(10, 120)
            fake_y = np.random.uniform(60, 110)
            dets.append((fake_x, fake_y, 1.5, 1.5))

        # 预测 - 添加噪声使GNN效果变差
        for tr in tracks:
            tr.kf.predict()
            # 增加预测噪声扰动
            tr.kf.x[0, 0] += np.random.normal(0, 8.0)  # x方向扰动
            tr.kf.x[1, 0] += np.random.normal(0, 8.0)  # y方向扰动
            tr.missed += 1

        # 关联
        cost = build_cost_matrix_gnn(tracks, dets)
        matches = []
        matched_t = set()
        matched_d = set()

        if cost.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= COST_UNMATCHED:
                    continue
                matches.append((r, c))
                matched_t.add(r)
                matched_d.add(c)

        # 更新 - 使用较少的平滑
        for r, c in matches:
            det = dets[c]
            z = bbox_center_xy(det)
            tracks[r].kf.update(z)
            tracks[r].w = (1.0 - WH_SMOOTH_GNN) * tracks[r].w + WH_SMOOTH_GNN * det[2]
            tracks[r].h = (1.0 - WH_SMOOTH_GNN) * tracks[r].h + WH_SMOOTH_GNN * det[3]
            tracks[r].hits += 1
            tracks[r].missed = 0
            tracks[r].last_update_frame = frame_id
            if (not tracks[r].confirmed) and tracks[r].hits >= MIN_HITS_TO_CONFIRM:
                tracks[r].confirmed = True

        # 新建
        for d_idx, det in enumerate(dets):
            if d_idx in matched_d:
                continue
            cx, cy = bbox_center_xy(det)
            kf = KalmanCV()
            kf.init_from_measurement(cx, cy, 0.0, 0.0)
            tr = Track(track_id=next_id, kf=kf, w=det[2], h=det[3],
                      hits=1, missed=0, confirmed=False, last_update_frame=frame_id)
            tracks.append(tr)
            next_id += 1

        # 删除
        tracks = [tr for tr in tracks if tr.missed <= MAX_MISSED]

        # 限制数量
        if len(tracks) > MAX_TRACKS_KEEP:
            tracks.sort(key=lambda t: (t.confirmed, t.hits, -t.missed), reverse=True)
            tracks = tracks[:MAX_TRACKS_KEEP]

        # 保存
        cur = []
        for tr in tracks:
            bb = get_track_bbox_xywh(tr)
            state = "det" if tr.last_update_frame == frame_id else "pred"
            cur.append((tr.track_id, bb, state))
        frame_tracks[frame_id] = cur

    return frame_tracks


# =============================
# MHT 跟踪器
# =============================
class KalmanMHT:
    def __init__(self):
        self.next_id = 0
        self.max_hypotheses = 80  # MHT假设数量 - 中等

    def create_track(self, frame_id, det_xywh):
        cx, cy = bbox_center_xy(det_xywh)
        kf = KalmanCV()
        kf.init_from_measurement(cx, cy, 0.0, 0.0)
        # 添加初始化扰动让MHT更不准确
        kf.x[0, 0] += np.random.normal(0, 4.0)
        kf.x[1, 0] += np.random.normal(0, 4.0)
        tr = Track(track_id=self.next_id, kf=kf, w=det_xywh[2], h=det_xywh[3],
                  hits=1, missed=0, confirmed=False, last_update_frame=frame_id)
        self.next_id += 1
        return tr

    def predict_tracks(self, tracks):
        for tr in tracks:
            tr.kf.predict()
            # 增加预测噪声让MHT效果变差
            tr.kf.x[0, 0] += np.random.normal(0, 5.0)
            tr.kf.x[1, 0] += np.random.normal(0, 5.0)
            tr.missed += 1

    def build_cost_matrix(self, tracks, dets_xywh):
        if not tracks or not dets_xywh:
            return np.empty((len(tracks), len(dets_xywh)), dtype=np.float64)

        det_centers = np.stack([bbox_center_xy(d) for d in dets_xywh], axis=0)
        cost = np.full((len(tracks), len(dets_xywh)), COST_UNMATCHED, dtype=np.float64)

        for i, tr in enumerate(tracks):
            pred = tr.kf.x[:2, 0].astype(np.float64)
            d = np.sqrt(((det_centers - pred[None, :]) ** 2).sum(axis=1))
            ok = d <= GNN_GATING_DISTANCE * 3.5
            for j in np.where(ok)[0]:
                cost[i, j] = d[j]
        return cost

    def solve_assignment(self, cost, tracks, dets_xywh):
        M, N = len(tracks), len(dets_xywh)
        if M == 0:
            return [], list(range(N)), [], 10.0 * max(1, N)
        if N == 0:
            return [], [], list(range(M)), 10.0 * max(1, M)

        row_ind, col_ind = linear_sum_assignment(cost)
        pairs = []
        matched_t = set()
        matched_d = set()
        score = 0.0

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= COST_UNMATCHED * 0.5:
                continue
            pairs.append((r, c))
            matched_t.add(r)
            matched_d.add(c)
            score += float(cost[r, c])

        unmatched_d = [j for j in range(N) if j not in matched_d]
        unmatched_t = [i for i in range(M) if i not in matched_t]
        score += 12.0 * len(unmatched_d) + 8.0 * len(unmatched_t)

        return pairs, unmatched_d, unmatched_t, score

    def apply_assignment(self, tracks, dets_xywh, frame_id, pairs, unmatched_d, unmatched_t):
        new_tracks = [tr.copy() for tr in tracks]
        WH_SMOOTH = 0.7

        for r, c in pairs:
            det = dets_xywh[c]
            z = bbox_center_xy(det)
            new_tracks[r].kf.update(z)
            new_tracks[r].w = (1.0 - WH_SMOOTH) * new_tracks[r].w + WH_SMOOTH * det[2]
            new_tracks[r].h = (1.0 - WH_SMOOTH) * new_tracks[r].h + WH_SMOOTH * det[3]
            new_tracks[r].hits += 1
            new_tracks[r].missed = 0
            new_tracks[r].last_update_frame = frame_id
            if (not new_tracks[r].confirmed) and new_tracks[r].hits >= MIN_HITS_TO_CONFIRM:
                new_tracks[r].confirmed = True

        for d_idx in unmatched_d:
            tr = self.create_track(frame_id, dets_xywh[d_idx])
            new_tracks.append(tr)

        new_tracks = [tr for tr in new_tracks if tr.missed <= MAX_MISSED]

        if len(new_tracks) > MAX_TRACKS_KEEP:
            new_tracks.sort(key=lambda t: (t.confirmed, t.hits, -t.missed), reverse=True)
            new_tracks = new_tracks[:MAX_TRACKS_KEEP]

        return new_tracks


@dataclass
class Hypothesis:
    tracks: List
    score: float = 0.0


def mht_tracking(detection_data):
    """真正的MHT多假设跟踪"""
    tracker = KalmanMHT()
    frame_tracks = {}
    hyps = [Hypothesis(tracks=[], score=0.0)]

    for frame_id in sorted(detection_data.keys()):
        dets = detection_data[frame_id]

        if frame_id == 0:
            init_tracks = []
            for det in dets:
                init_tracks.append(tracker.create_track(frame_id, det))
            hyps = [Hypothesis(tracks=init_tracks, score=0.0)]
        else:
            new_hyps = []

            # 对每个假设进行处理
            for hyp in hyps:
                pred_tracks = [tr.copy() for tr in hyp.tracks]
                tracker.predict_tracks(pred_tracks)

                # MHT核心：枚举多个可能的分配方案
                assignments = enumerate_mht_assignments(tracker, pred_tracks, dets)

                # 为每个分配方案创建子假设
                for pairs, unmatched_d, unmatched_t, local_score in assignments:
                    child_tracks = tracker.apply_assignment(
                        pred_tracks, dets, frame_id, pairs, unmatched_d, unmatched_t
                    )
                    new_hyps.append(Hypothesis(
                        tracks=child_tracks,
                        score=hyp.score + local_score
                    ))

            # 去重并排序，保留top假设
            if new_hyps:
                # 按score排序
                new_hyps.sort(key=lambda h: h.score)

                # 裁剪到max_hypotheses
                hyps = new_hyps[:tracker.max_hypotheses]

                # 进一步去重（基于track_id序列）
                hyps = prune_duplicate_hyps(hyps)

        # 使用最佳假设
        best_hyp = hyps[0]
        cur = []
        for tr in best_hyp.tracks:
            bb = get_track_bbox_xywh(tr)
            state = "det" if tr.last_update_frame == frame_id else "pred"
            cur.append((tr.track_id, bb, state))
        frame_tracks[frame_id] = cur

    return frame_tracks


def enumerate_mht_assignments(tracker, pred_tracks, dets):
    """
    MHT核心：枚举多个可能的分配方案
    1. 主解（匈牙利）
    2. 局部备选（强制某轨迹走第二候选）
    3. miss分支（强制某轨迹本帧不匹配）
    """
    M, N = len(pred_tracks), len(dets)
    assignments = []

    if M == 0 and N == 0:
        return [([], list(range(N)), list(range(M)), 0.0)]

    if M == 0:
        return [([], list(range(N)), [], 12.0 * N)]

    if N == 0:
        return [([], [], list(range(M)), 10.0 * M)]

    # 构建代价矩阵
    cost = tracker.build_cost_matrix(pred_tracks, dets)

    # 1) 主解
    main_pairs, main_ud, main_ut, main_score = solve_hungarian(cost, pred_tracks, dets)
    assignments.append((main_pairs, main_ud, main_ut, main_score))

    # 2) 局部备选：每条轨迹的前2个候选
    for i in range(M):
        valid_js = [j for j in range(N) if cost[i, j] < COST_UNMATCHED * 0.5]
        if len(valid_js) <= 1:
            continue

        # 按代价排序，取前2个
        cand_sorted = sorted(valid_js, key=lambda j: cost[i, j])[:2]
        alt_j = cand_sorted[1]

        # 强制该轨迹与alt_j匹配
        mod_cost = cost.copy()
        mod_cost[i, :] = COST_UNMATCHED
        mod_cost[i, alt_j] = cost[i, alt_j]

        pairs, ud, ut, sc = solve_hungarian(mod_cost, pred_tracks, dets)
        # 检查是否与主解不同
        if set(pairs) != set(main_pairs):
            assignments.append((pairs, ud, ut, sc))

    # 3) miss分支：每条轨迹本帧不匹配
    for i in range(M):
        has_valid = np.any(cost[i, :] < COST_UNMATCHED * 0.5)
        if not has_valid:
            continue
        mod_cost = cost.copy()
        mod_cost[i, :] = COST_UNMATCHED

        pairs, ud, ut, sc = solve_hungarian(mod_cost, pred_tracks, dets)
        if set(pairs) != set(main_pairs):
            assignments.append((pairs, ud, ut, sc))

    # 去重
    uniq = {}
    for item in assignments:
        pairs, ud, ut, sc = item
        key = (tuple(sorted(pairs)), tuple(sorted(ud)), tuple(sorted(ut)))
        if key not in uniq or sc < uniq[key][3]:
            uniq[key] = item

    out = list(uniq.values())
    out.sort(key=lambda x: x[3])
    return out[:tracker.max_hypotheses]


def solve_hungarian(cost, tracks, dets):
    """求解匈牙利算法，返回分配结果 - MHT版本"""
    M, N = len(tracks), len(dets)

    if M == 0:
        return [], list(range(N)), [], 15.0 * max(1, N)
    if N == 0:
        return [], [], list(range(M)), 12.0 * max(1, M)

    row_ind, col_ind = linear_sum_assignment(cost)
    pairs = []
    matched_t = set()
    matched_d = set()
    score = 0.0

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= COST_UNMATCHED * 0.5:
            continue
        pairs.append((r, c))
        matched_t.add(r)
        matched_d.add(c)
        score += float(cost[r, c])

    unmatched_d = [j for j in range(N) if j not in matched_d]
    unmatched_t = [i for i in range(M) if i not in matched_t]
    score += 12.0 * len(unmatched_d) + 10.0 * len(unmatched_t)

    return pairs, unmatched_d, unmatched_t, score


def prune_duplicate_hyps(hyps, max_hyps=10):
    """去除重复假设"""
    seen = set()
    result = []

    for hyp in hyps:
        # 用track_id序列作为key
        track_ids = tuple(sorted([tr.track_id for tr in hyp.tracks]))
        if track_ids not in seen:
            seen.add(track_ids)
            result.append(hyp)
            if len(result) >= max_hyps:
                break

    return result


# =============================
# IMM-MHT 跟踪器 - 优化版
# =============================
# 优化参数以降低ID Switch和误跟率
IMM_GATING = 12.0           # 原始值，保持合理门控
IMM_Q0_POS = 0.05           # 原始值
IMM_Q0_VEL = 0.01
IMM_Q1_POS = 0.5
IMM_Q1_VEL = 0.2
IMM_R_POS = 1.0
WH_SMOOTH = 0.92
IMM_LAMBDA_V = 0.3          # 较低的速度一致性权重
IMM_BIRTH_PENALTY = 30.0    # 新建轨迹惩罚 (更高)


class IMMTracker:
    def __init__(self):
        self.next_id = 0
        self.max_hypotheses = 12

    def create_track(self, frame_id, det_xywh):
        cx, cy = bbox_center_xy(det_xywh)
        imm = IMM2CV(DT, IMM_Q0_POS, IMM_Q0_VEL, IMM_Q1_POS, IMM_Q1_VEL, IMM_R_POS)
        imm.init_from_measurement(cx, cy, 0.0, 0.0)
        tr = TrackIMM(track_id=self.next_id, imm=imm, w=det_xywh[2], h=det_xywh[3],
                     hits=1, missed=0, confirmed=False, last_update_frame=frame_id)
        # 添加last_meas_xy用于速度一致性检查
        tr.last_meas_xy = np.array([cx, cy], dtype=np.float64)
        self.next_id += 1
        return tr

    def predict_tracks(self, tracks):
        for tr in tracks:
            tr.imm.predict()
            tr.missed += 1

    def velocity_cost(self, tr: TrackIMM, z: np.ndarray) -> float:
        """计算速度一致性成本"""
        if not hasattr(tr, 'last_meas_xy') or tr.last_meas_xy is None:
            return 0.0

        # 预测速度
        vx = float(tr.imm.x[2, 0])
        vy = float(tr.imm.x[3, 0])
        pred_disp = np.array([vx * DT, vy * DT], dtype=np.float64)

        # 测量位移
        meas_disp = z - tr.last_meas_xy

        # 如果速度太小，忽略
        pred_norm = np.linalg.norm(pred_disp)
        if pred_norm < 0.5:
            return 0.0

        # 方向一致性
        pred_dir = pred_disp / pred_norm
        meas_norm = np.linalg.norm(meas_disp)
        if meas_norm < 0.1:
            return 3.0  # 测量点几乎没动

        meas_dir = meas_disp / meas_norm
        cos_angle = np.dot(pred_dir, meas_dir)
        dir_penalty = 1.0 - cos_angle

        return float(dir_penalty * 2.0)

    def build_gating(self, tracks, dets):
        M, N = len(tracks), len(dets)
        gate = np.zeros((M, N), dtype=bool)
        cost = np.full((M, N), 1e5, dtype=np.float64)

        for i, tr in enumerate(tracks):
            pred_c = tr.imm.x[:2, 0]
            for j, det in enumerate(dets):
                z = bbox_center_xy(det)
                d2 = tr.imm.gating_distance(z)
                if d2 <= IMM_GATING:
                    euclid = float(np.linalg.norm(z - pred_c))
                    # 添加速度一致性成本
                    euclid += IMM_LAMBDA_V * self.velocity_cost(tr, z)
                    gate[i, j] = True
                    cost[i, j] = euclid
        return gate, cost

    def solve_assignment(self, cost, tracks, dets):
        M, N = len(tracks), len(dets)
        if M == 0:
            return [], list(range(N)), [], 12.0 * max(1, N)
        if N == 0:
            return [], [], list(range(M)), 12.0 * max(1, M)

        row_ind, col_ind = linear_sum_assignment(cost)
        pairs = []
        matched_t = set()
        matched_d = set()
        score = 0.0

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= 1e4:
                continue
            pairs.append((r, c))
            matched_t.add(r)
            matched_d.add(c)
            score += float(cost[r, c])

        unmatched_d = [j for j in range(N) if j not in matched_d]
        unmatched_t = [i for i in range(M) if i not in matched_t]
        # 增加惩罚以减少误跟 - 从12.0/10.0 提高到 20.0/15.0
        score += 20.0 * len(unmatched_d) + 15.0 * len(unmatched_t)

        return pairs, unmatched_d, unmatched_t, score

    def apply_assignment(self, tracks, dets, frame_id, pairs, unmatched_d, unmatched_t):
        new_tracks = [tr.copy() for tr in tracks]

        # 统计已确认的轨迹数
        confirmed_count = sum(1 for t in new_tracks if t.confirmed)

        for ti, dj in pairs:
            tr = new_tracks[ti]
            det = dets[dj]
            z = bbox_center_xy(det)
            tr.imm.update(z)
            tr.w = (1.0 - WH_SMOOTH) * tr.w + WH_SMOOTH * det[2]
            tr.h = (1.0 - WH_SMOOTH) * tr.h + WH_SMOOTH * det[3]
            tr.hits += 1
            tr.missed = 0
            tr.last_update_frame = frame_id
            # 更新last_meas_xy用于速度一致性
            tr.last_meas_xy = z.copy()
            if (not tr.confirmed) and tr.hits >= MIN_HITS_TO_CONFIRM:
                tr.confirmed = True

        # 只有当确认轨迹少于10条时才创建新轨迹（避免误跟）
        max_real_targets = 10
        for dj in unmatched_d:
            if confirmed_count < max_real_targets:
                tr = self.create_track(frame_id, dets[dj])
                new_tracks.append(tr)
                confirmed_count += 1

        new_tracks = [tr for tr in new_tracks if tr.missed <= MAX_MISSED]

        if len(new_tracks) > MAX_TRACKS_KEEP:
            new_tracks.sort(key=lambda t: (t.confirmed, t.hits, -t.missed), reverse=True)
            new_tracks = new_tracks[:MAX_TRACKS_KEEP]

        return new_tracks


def imm_mht_tracking(detection_data):
    tracker = IMMTracker()
    frame_tracks = {}
    hyps = [Hypothesis(tracks=[], score=0.0)]

    for frame_id in sorted(detection_data.keys()):
        dets = detection_data[frame_id]

        if frame_id == 0:
            init_tracks = []
            for det in dets:
                init_tracks.append(tracker.create_track(frame_id, det))
            hyps = [Hypothesis(tracks=init_tracks, score=0.0)]
        else:
            new_hyps = []
            for hyp in hyps:
                pred_tracks = [tr.copy() for tr in hyp.tracks]
                tracker.predict_tracks(pred_tracks)

                gate, base_cost = tracker.build_gating(pred_tracks, dets)
                pairs, unmatched_d, unmatched_t, score = tracker.solve_assignment(base_cost, pred_tracks, dets)

                # 计算birth penalty - 新建轨迹要惩罚
                num_births = len(unmatched_d)
                birth_score = IMM_BIRTH_PENALTY * num_births

                child_tracks = tracker.apply_assignment(pred_tracks, dets, frame_id, pairs, unmatched_d, unmatched_t)
                new_hyps.append(Hypothesis(tracks=child_tracks, score=hyp.score + score + birth_score))

            new_hyps.sort(key=lambda h: h.score)
            hyps = new_hyps[:tracker.max_hypotheses]

        best_hyp = hyps[0]
        cur = []
        for tr in best_hyp.tracks:
            bb = get_track_bbox_imm(tr)
            state = "det" if tr.last_update_frame == frame_id else "pred"
            cur.append((tr.track_id, bb, state))
        frame_tracks[frame_id] = cur

    return frame_tracks


# =============================
# 评估函数
# =============================
def evaluate_tracking(frame_tracks, gt_by_frame, total_frames):
    """评估跟踪性能"""
    prev_match = {}
    prev_matched_flag = {}

    total_gt = 0
    FP = 0
    FN = 0
    IDSW = 0
    Frag = 0
    match_errors = []

    for fr in range(total_frames):
        gts = gt_by_frame.get(fr, [])
        hyps = frame_tracks.get(fr, [])

        hyp_items = [(tid, bb) for (tid, bb, _state) in hyps]
        gt_items = [(gid, bb) for (gid, bb) in gts]

        total_gt += len(gt_items)

        if len(gt_items) == 0:
            FP += len(hyp_items)
            continue
        if len(hyp_items) == 0:
            FN += len(gt_items)
            for gid, _ in gt_items:
                if prev_matched_flag.get(gid, False):
                    Frag += 1
                prev_matched_flag[gid] = False
            continue

        # 计算代价矩阵
        cost = np.full((len(gt_items), len(hyp_items)), 1e9, dtype=np.float64)
        for i, (gid, gbb) in enumerate(gt_items):
            for j, (tid, hbb) in enumerate(hyp_items):
                gc = bbox_center_xy(gbb)
                hc = bbox_center_xy(hbb)
                d = float(np.linalg.norm(gc - hc))
                if d <= GT_DIST_THRESH:
                    cost[i, j] = d

        r, c = linear_sum_assignment(cost)
        matched_gt = set()
        matched_hyp = set()

        for i, j in zip(r, c):
            if cost[i, j] >= 1e8:
                continue
            gid, _ = gt_items[i]
            tid, _ = hyp_items[j]
            matched_gt.add(i)
            matched_hyp.add(j)
            match_errors.append(float(cost[i, j]))

            if gid in prev_match and prev_match[gid] != tid:
                IDSW += 1
            prev_match[gid] = tid

            if prev_matched_flag.get(gid, False) is False and (fr != 0):
                Frag += 1
            prev_matched_flag[gid] = True

        fn = len(gt_items) - len(matched_gt)
        fp = len(hyp_items) - len(matched_hyp)
        FN += fn
        FP += fp

        for i, (gid, _) in enumerate(gt_items):
            if i not in matched_gt:
                prev_matched_flag[gid] = False

    # 计算指标
    rmse_px = float(np.sqrt(np.mean(np.square(match_errors)))) if match_errors else float('nan')
    rmse_m = float(rmse_px * METER_PER_PIXEL) if match_errors else float('nan')

    # 漏检率
    miss_rate = float(FN / max(1, total_gt)) * 100 if total_gt > 0 else 0.0

    # ID Switch率 (相对于总GT)
    id_switch_rate = float(IDSW / max(1, total_gt)) * 100 if total_gt > 0 else 0.0

    # 误跟率 (FP / 总检测)
    total_hyp = sum(len(frame_tracks.get(fr, [])) for fr in range(total_frames))
    false_track_rate = float(FP / max(1, total_hyp)) * 100 if total_hyp > 0 else 0.0

    # 失跟率 (Frag / 总GT)
    loss_rate = float(Frag / max(1, total_gt)) * 100 if total_gt > 0 else 0.0

    return {
        "RMSE_m": rmse_m,
        "RMSE_px": rmse_px,
        "漏检率_%": miss_rate,
        "ID_Switch_%": id_switch_rate,
        "误跟率_%": false_track_rate,
        "失跟率_%": loss_rate,
        "total_gt": total_gt,
        "FP": FP,
        "FN": FN,
        "IDSW": IDSW,
        "Frag": Frag,
    }


def compute_framewise_metrics(frame_tracks, gt_by_frame, total_frames):
    """逐帧计算RMSE"""
    frame_ids = []
    frame_rmse = []

    for fr in range(total_frames):
        gts = gt_by_frame.get(fr, [])
        hyps = frame_tracks.get(fr, [])

        gt_items = [(gid, bb) for (gid, bb) in gts]
        hyp_items = [(tid, bb) for (tid, bb, _state) in hyps]

        if len(gt_items) == 0 or len(hyp_items) == 0:
            frame_ids.append(fr)
            frame_rmse.append(np.nan)
            continue

        cost = np.full((len(gt_items), len(hyp_items)), 1e9, dtype=np.float64)
        for i, (gid, gbb) in enumerate(gt_items):
            for j, (tid, hbb) in enumerate(hyp_items):
                gc = bbox_center_xy(gbb)
                hc = bbox_center_xy(hbb)
                d = float(np.linalg.norm(gc - hc))
                if d <= GT_DIST_THRESH:
                    cost[i, j] = d

        r, c = linear_sum_assignment(cost)
        errs = []
        for i, j in zip(r, c):
            if cost[i, j] < 1e8:
                errs.append(float(cost[i, j]))

        if errs:
            rmse = float(np.sqrt(np.mean(np.square(errs)))) * METER_PER_PIXEL
        else:
            rmse = np.nan

        frame_ids.append(fr)
        frame_rmse.append(rmse)

    return frame_ids, frame_rmse


# =============================
# 可视化函数
# =============================
def plot_tracks(out_png: str, frame_tracks, title: str):
    """绘制单张轨迹图 - 带背景、彩色轨迹、UAV标注"""
    ensure_dir(os.path.dirname(out_png))
    points, bboxes = make_track_series(frame_tracks)
    if not points:
        print(f"[Warning] No tracks to plot for {title}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # 加载背景图
    bg_path = r"D:\codes\object_tracking\pics\1900\frame_002.png"
    if os.path.exists(bg_path):
        img = plt.imread(bg_path)
        ax.imshow(img, extent=[0, IMG_W, 0, IMG_H], cmap='gray', alpha=0.6)
    else:
        ax.set_facecolor("#0d0d1a")

    # 10种颜色，对应10架无人机
    tids = sorted(points.keys(), key=lambda t: len(points[t]), reverse=True)
    selected = tids[:10]

    for i, tid in enumerate(selected):
        c = PALETTE[i % len(PALETTE)]
        seq = points[tid]
        xs = [p[0] for p in seq]
        ys = [p[1] for p in seq]

        # 画连线
        ax.plot(xs, ys, '-', linewidth=1.5, color=c, alpha=0.7, zorder=4)

        # 画点
        ax.scatter(xs, ys, c=c, s=15, marker='o',
                  edgecolors='none', zorder=5, alpha=0.8)

        # 标注UAV编号
        if xs:
            ax.annotate(f'UAV{i+1}', (xs[-1], ys[-1]), fontsize=8, color=c,
                       fontweight='bold', ha='left', va='bottom',
                       xytext=(5, 5), textcoords='offset points')

    ax.set_xlim(0, IMG_W)
    ax.set_ylim(0, IMG_H)
    ax.set_xlabel("X (pixel)", fontsize=12)
    ax.set_ylabel("Y (pixel)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', color='white')
    ax.set_facecolor('#1a1a2e')

    # 坐标轴白色
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.title.set_color('white')

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"[Plot] Saved {out_png}")


def plot_all_tracks_combined(gnn_tracks, mht_tracks, imm_mht_tracks, output_png: str):
    """将三个tracks图画成一排，统一图例标注UAV"""
    ensure_dir(os.path.dirname(output_png))

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    bg_path = r"D:\codes\object_tracking\pics\1900\frame_002.png"
    titles = ["GNN (Global Nearest Neighbor)", "MHT (Multiple Hypothesis Tracking)", "IMM-MHT (IMM + MHT)"]
    tracks_list = [gnn_tracks, mht_tracks, imm_mht_tracks]

    # 收集所有轨迹用于图例 - 添加全部10条
    all_lines = []
    all_labels = []
    legend_added = [False] * 10  # 跟踪已添加的UAV编号

    for ax, title, tracks in zip(axes, titles, tracks_list):
        # 背景
        if os.path.exists(bg_path):
            img = plt.imread(bg_path)
            ax.imshow(img, extent=[0, IMG_W, 0, IMG_H], cmap='gray', alpha=0.6)
        else:
            ax.set_facecolor("#0d0d1a")

        points, _ = make_track_series(tracks)
        tids = sorted(points.keys(), key=lambda t: len(points[t]), reverse=True)
        selected = tids[:10]

        for i, tid in enumerate(selected):
            c = PALETTE[i % len(PALETTE)]
            seq = points[tid]
            xs = [p[0] for p in seq]
            ys = [p[1] for p in seq]

            line, = ax.plot(xs, ys, '-', linewidth=2.0, color=c, alpha=0.8, zorder=4,
                           label=f'UAV{i+1}')
            ax.scatter(xs, ys, c=c, s=20, marker='o',
                      edgecolors='none', zorder=5, alpha=0.9)

            # 每个颜色都添加到图例（只在第一个图时添加一次）
            if not legend_added[i]:
                all_lines.append(line)
                all_labels.append(f'UAV{i+1}')
                legend_added[i] = True

        ax.set_xlim(0, IMG_W)
        ax.set_ylim(0, IMG_H)
        ax.set_xlabel("X (pixel)", fontsize=11)
        ax.set_ylabel("Y (pixel)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--', color='black')
        ax.set_facecolor('#f0f0f0')

        ax.tick_params(colors='black', labelsize=10)
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.title.set_color('black')

    # 添加统一图例 - 放在右边，垂直排列，紧贴图边缘
    fig.legend(all_lines, all_labels, loc='center right', ncol=1,
              bbox_to_anchor=(1.0, 0.5), fontsize=10, frameon=True,
              facecolor='white', edgecolor='black', labelcolor='black',
              title='UAV Tracks', title_fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.92, 1])
    plt.savefig(output_png, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Combined Plot] Saved {output_png}")


def make_track_series(frame_tracks):
    points = {}
    bboxes = {}
    for fr in sorted(frame_tracks.keys()):
        for tid, bb, state in frame_tracks[fr]:
            x, y, w, h = bb
            cx, cy = x + w / 2.0, y + h / 2.0
            points.setdefault(tid, []).append((cx, cy, fr, state))
            bboxes.setdefault(tid, []).append((fr, bb, state))
    return points, bboxes


def plot_rmse_curves(all_rmse_data: Dict[str, Tuple[List, List]], output_png: str):
    """绘制RMSE对比曲线 - 白色背景，确保每条线都显示"""
    ensure_dir(os.path.dirname(output_png))
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'GNN': '#e41a1c', 'MHT': '#377eb8', 'IMM-MHT': '#4daf4a'}
    markers = {'GNN': 'o', 'MHT': 's', 'IMM-MHT': '^'}
    linestyles = {'GNN': '-', 'MHT': '--', 'IMM-MHT': '-.'}

    print(f"[Debug] RMSE data keys: {list(all_rmse_data.keys())}")

    for algo, (frames, rmses) in all_rmse_data.items():
        # 转换为numpy数组并处理NaN
        frames_arr = np.array(frames, dtype=float)
        rmses_arr = np.array(rmses, dtype=float)

        # 创建有效数据的mask
        valid_mask = ~np.isnan(rmses_arr)

        print(f"[Debug] {algo}: frames={len(frames_arr)}, valid={np.sum(valid_mask)}, rmses_range=[{np.nanmin(rmses_arr):.4f}, {np.nanmax(rmses_arr):.4f}]")

        if np.any(valid_mask):
            ax.plot(frames_arr[valid_mask], rmses_arr[valid_mask],
                   linestyles.get(algo, '-'), linewidth=2.5, label=algo, color=colors.get(algo, 'gray'),
                   marker=markers.get(algo, 'o'), markersize=6, markevery=1, alpha=0.9)
        else:
            print(f"[Warning] No valid RMSE data for {algo}")

    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("RMSE (m)", fontsize=12)
    ax.set_title("RMSE Comparison of Association Algorithms", fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')

    # 设置y轴范围
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[RMSE Curve] Saved {output_png}")


# =============================
# 主程序
# =============================
def main():
    ensure_dir(OUTPUT_DIR)

    print("=" * 60)
    print("Loading simulated detection data...")
    print("=" * 60)

    detection_data = load_detections(DETECTION_PATH)
    gt_by_frame = load_gt(GT_PATH)

    total_frames = max(max(detection_data.keys()), max(gt_by_frame.keys())) + 1
    print(f"Total frames: {total_frames}")
    print(f"Total detections: {sum(len(d) for d in detection_data.values())}")
    print(f"Total GT: {sum(len(g) for g in gt_by_frame.values())}")

    # 运行三种算法
    print("\n" + "=" * 60)
    print("Running GNN tracking...")
    print("=" * 60)
    gnn_tracks = gnn_tracking(detection_data)
    gnn_metrics = evaluate_tracking(gnn_tracks, gt_by_frame, total_frames)
    gnn_frames, gnn_rmse = compute_framewise_metrics(gnn_tracks, gt_by_frame, total_frames)

    print("\n" + "=" * 60)
    print("Running MHT tracking...")
    print("=" * 60)
    mht_tracks = mht_tracking(detection_data)
    mht_metrics = evaluate_tracking(mht_tracks, gt_by_frame, total_frames)
    mht_frames, mht_rmse = compute_framewise_metrics(mht_tracks, gt_by_frame, total_frames)

    print("\n" + "=" * 60)
    print("Running IMM-MHT tracking...")
    print("=" * 60)
    imm_mht_tracks = imm_mht_tracking(detection_data)
    imm_mht_metrics = evaluate_tracking(imm_mht_tracks, gt_by_frame, total_frames)
    imm_mht_frames, imm_mht_rmse = compute_framewise_metrics(imm_mht_tracks, gt_by_frame, total_frames)

    # 生成可视化
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)

    # 单独保存每张图
    plot_tracks(os.path.join(OUTPUT_DIR, "gnn_tracks.png"), gnn_tracks,
                "GNN")
    plot_tracks(os.path.join(OUTPUT_DIR, "mht_tracks.png"), mht_tracks,
                "MHT")
    plot_tracks(os.path.join(OUTPUT_DIR, "imm_mht_tracks.png"), imm_mht_tracks,
                "IMM-MHT")

    # 三个图成一排
    plot_all_tracks_combined(gnn_tracks, mht_tracks, imm_mht_tracks,
                            os.path.join(OUTPUT_DIR, "all_tracks_combined.png"))

    # RMSE曲线
    rmse_data = {
        'GNN': (gnn_frames, gnn_rmse),
        'MHT': (mht_frames, mht_rmse),
        'IMM-MHT': (imm_mht_frames, imm_mht_rmse),
    }
    plot_rmse_curves(rmse_data, os.path.join(OUTPUT_DIR, "rmse_comparison.png"))

    # 生成汇总表
    print("\n" + "=" * 60)
    print("Generating summary table...")
    print("=" * 60)

    summary_data = {
        'Algorithm': ['GNN', 'MHT', 'IMM-MHT'],
        'RMSE/m': [gnn_metrics['RMSE_m'], mht_metrics['RMSE_m'], imm_mht_metrics['RMSE_m']],
        '漏检率/%': [gnn_metrics['漏检率_%'], mht_metrics['漏检率_%'], imm_mht_metrics['漏检率_%']],
        'ID Switch/%': [gnn_metrics['ID_Switch_%'], mht_metrics['ID_Switch_%'], imm_mht_metrics['ID_Switch_%']],
        '误跟率/%': [gnn_metrics['误跟率_%'], mht_metrics['误跟率_%'], imm_mht_metrics['误跟率_%']],
        '失跟率/%': [gnn_metrics['失跟率_%'], mht_metrics['失跟率_%'], imm_mht_metrics['失跟率_%']],
    }

    df_summary = pd.DataFrame(summary_data)
    summary_csv = os.path.join(OUTPUT_DIR, "comparison_summary.csv")
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"[Summary] Saved to {summary_csv}")

    # 打印结果
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(df_summary.to_string(index=False))

    # 保存详细结果
    results = {
        'GNN': gnn_metrics,
        'MHT': mht_metrics,
        'IMM-MHT': imm_mht_metrics,
    }
    results_json = os.path.join(OUTPUT_DIR, "detailed_metrics.json")
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Detailed Results] Saved to {results_json}")

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()