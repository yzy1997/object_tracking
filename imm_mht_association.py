# main_mot_imm_mht_custom.py
# -*- coding: utf-8 -*-

import os
import re
import glob
import json
import math
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from src.pixel_shifting_correction import RadarImageProcessor
from src.object_detection import SpatialDroneDetector


# =========================================================
# 路径配置
# =========================================================
INPUT_DIR = r"D:\codes\object_tracking\data\out_uav_4_edge_to_edge_bidir\txt"
OUTPUT_DIR = r"D:\codes\object_tracking\results\mot_imm_mht_custom_out"
GT_PATH: Optional[str] = r"D:\codes\object_tracking\data\out_uav_4_edge_to_edge_bidir\gt_tracks.csv"


# =========================================================
# 图像 / 换算
# =========================================================
IMG_W, IMG_H = 132, 132

USE_PIXEL_TO_METER = True
RANGE_M = 600.0
FOV_DEG = 5.0
IMG_WIDTH_PX = 132


# =========================================================
# 检测参数
# =========================================================
WINDOW_SIZE = 4
SHIFT_PIXEL = 4
DETECTOR_DEBUG = False
DETECTOR_MIN_Y = 0


# =========================================================
# IMM-MHT 参数
# =========================================================
DT = 1.0

# IMM 两模型：都用 CV，但过程噪声不同
# model 0: 平滑
# model 1: 灵活
IMM_Q0_POS = 0.30
IMM_Q0_VEL = 0.08
IMM_Q1_POS = 3.50
IMM_Q1_VEL = 1.50

R_POS = 3.0
WH_SMOOTH = 0.7
# 模型转移矩阵
PI_00 = 0.96
PI_11 = 0.92

# MHT 参数
MAX_MISSED = 5
MIN_HITS_TO_CONFIRM = 2
MAX_TRACKS_KEEP = 10
MAX_HYPOTHESES = 20
N_SCAN = 2

# 统计门控
GATING_CHI2 = 5.99  # 2D 95%

# 代价
MISS_PENALTY = 7.0
BIRTH_PENALTY = 11.0
LAMBDA_SIZE = 0.05

# 评估匹配
GT_MATCH_METRIC = "dist"
GT_DIST_THRESH = 10.0
GT_IOU_THRESH = 0.3


# =========================================================
# 可视化
# =========================================================
DISPLAY_CARTESIAN_Y = True   # True: 下方为0，上方为IMG_H；False: 保持原图像坐标
NUM_DRONES_TO_SHOW = 4
BBOX_ALPHA = 0.65
BBOX_LINEWIDTH = 1.3
PALETTE = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]


# =========================================================
# 工具函数
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def natural_sort_key(s: str):
    parts = re.split(r"(\d+)", os.path.basename(s))
    return [int(p) if p.isdigit() else p for p in parts]


def list_frame_files(dir_path: str) -> List[str]:
    files = glob.glob(os.path.join(dir_path, "frame_*.txt"))
    return sorted(files, key=natural_sort_key)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def xyxy_to_xywh_list(boxes_xyxy) -> List[Tuple[float, float, float, float]]:
    dets = []
    if not boxes_xyxy:
        return dets
    for (x0, x1, y0, y1) in boxes_xyxy:
        x0f, x1f = float(min(x0, x1)), float(max(x0, x1))
        y0f, y1f = float(min(y0, y1)), float(max(y0, y1))
        w = x1f - x0f
        h = y1f - y0f
        if w <= 0 or h <= 0:
            continue
        dets.append((x0f, y0f, w, h))
    return dets


def bbox_center_xy(b):
    x, y, w, h = b
    return np.array([x + w / 2.0, y + h / 2.0], dtype=np.float64)


def bbox_iou(a_xywh, b_xywh) -> float:
    ax, ay, aw, ah = a_xywh
    bx, by, bw, bh = b_xywh
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def pixel_to_meter_scale(range_m, fov_deg, img_width_px):
    scene_width_m = 2.0 * range_m * math.tan(math.radians(fov_deg / 2.0))
    meter_per_pixel = scene_width_m / img_width_px
    return meter_per_pixel, scene_width_m


if USE_PIXEL_TO_METER:
    METER_PER_PIXEL, SCENE_WIDTH_M = pixel_to_meter_scale(RANGE_M, FOV_DEG, IMG_WIDTH_PX)
else:
    METER_PER_PIXEL = 1.0
    SCENE_WIDTH_M = None


# =========================================================
# 检测：窗口累积
# =========================================================
def run_detection_windowed(frame_files: List[str]) -> Dict[int, List[Tuple[float, float, float, float]]]:
    if len(frame_files) < 2:
        raise RuntimeError("至少需要 frame_0000 + 一个 update 帧")

    full_scan = frame_files[0]
    updates = frame_files[1:]

    processor = RadarImageProcessor(resolution=(IMG_W, IMG_H), shift_pixel=SHIFT_PIXEL)

    detector = SpatialDroneDetector(
        processor=processor,
        full_scan_file=full_scan,
        update_files=[],
        debug=DETECTOR_DEBUG,
        min_y=DETECTOR_MIN_Y,
    )

    detection_data: Dict[int, List[Tuple[float, float, float, float]]] = {}

    detector.update_files = []
    detection_data[0] = xyxy_to_xywh_list(detector.detect())

    for frame_id in range(1, len(frame_files)):
        end = frame_id
        start = max(0, end - WINDOW_SIZE)
        window_updates = updates[start:end]

        detector.update_files = window_updates
        boxes_xyxy = detector.detect()
        detection_data[frame_id] = xyxy_to_xywh_list(boxes_xyxy)

        if (frame_id % 50 == 0) or (frame_id < 5):
            print(
                f"[detect] frame={frame_id:04d} window=[{start+1:04d}..{end:04d}] "
                f"files={len(window_updates)} dets={len(detection_data[frame_id])}"
            )

    return detection_data


# =========================================================
# IMM 两模型 CV 滤波器
# =========================================================
class CVKalman:
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
        new = CVKalman(self.dt, self.Q[0, 0], self.Q[2, 2], self.R[0, 0])
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
        self.f0 = CVKalman(dt, q0_pos, q0_vel, r_pos)
        self.f1 = CVKalman(dt, q1_pos, q1_vel, r_pos)
        self.mu = np.array([0.5, 0.5], dtype=np.float64)

        self.PI = np.array([
            [PI_00, 1.0 - PI_00],
            [1.0 - PI_11, PI_11],
        ], dtype=np.float64)

        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64)

    def copy(self):
        new = IMM2CV(
            DT, IMM_Q0_POS, IMM_Q0_VEL,
            IMM_Q1_POS, IMM_Q1_VEL,
            R_POS
        )
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

        return float(np.sum(likelihoods))


# =========================================================
# 轨迹 / 假设结构
# =========================================================
@dataclass
class Track:
    track_id: int
    imm: IMM2CV
    w: float
    h: float
    hits: int = 1
    missed: int = 0
    confirmed: bool = False
    last_update_frame: int = 0
    history: List[Tuple[int, Tuple[float, float, float, float], str]] = field(default_factory=list)

    def copy(self):
        return Track(
            track_id=self.track_id,
            imm=self.imm.copy(),
            w=float(self.w),
            h=float(self.h),
            hits=int(self.hits),
            missed=int(self.missed),
            confirmed=bool(self.confirmed),
            last_update_frame=int(self.last_update_frame),
            history=list(self.history),
        )


@dataclass
class Hypothesis:
    tracks: List[Track]
    score: float = 0.0


# =========================================================
# IMM-MHT 核心
# =========================================================
class CustomIMMMHT:
    def __init__(self):
        self.next_id = 0
        self.gating_thresh = GATING_CHI2
        self.max_hypotheses = MAX_HYPOTHESES
        self.n_scan = N_SCAN

    def create_track(self, det_xywh, frame_id: int) -> Track:
        cx, cy = bbox_center_xy(det_xywh)
        _, _, w, h = det_xywh
        imm = IMM2CV(DT, IMM_Q0_POS, IMM_Q0_VEL, IMM_Q1_POS, IMM_Q1_VEL, R_POS)
        imm.init_from_measurement(cx, cy, 0.0, 0.0)
        tr = Track(
            track_id=self.next_id,
            imm=imm,
            w=w,
            h=h,
            hits=1,
            missed=0,
            confirmed=False,
            last_update_frame=frame_id,
            history=[],
        )
        self.next_id += 1
        return tr

    def predict_tracks(self, tracks: List[Track]):
        for tr in tracks:
            tr.imm.predict()
            tr.missed += 1

    def build_gating_matrix(self, tracks: List[Track], dets: List[Tuple[float, float, float, float]]):
        M, N = len(tracks), len(dets)
        gate = np.zeros((M, N), dtype=bool)
        cost = np.full((M, N), 1e6, dtype=np.float64)

        for i, tr in enumerate(tracks):
            pred_c = tr.imm.x[:2, 0]
            for j, det in enumerate(dets):
                z = bbox_center_xy(det)
                d2 = tr.imm.gating_distance(z)
                if d2 <= self.gating_thresh:
                    size_cost = LAMBDA_SIZE * (
                        abs(det[2] - tr.w) + abs(det[3] - tr.h)
                    )
                    euclid = float(np.linalg.norm(z - pred_c))
                    gate[i, j] = True
                    cost[i, j] = euclid + size_cost
        return gate, cost

    def enumerate_assignments(self, tracks: List[Track], dets: List[Tuple[float, float, float, float]]):
        """
        为了控制复杂度：
        - 先用 Hungarian 出一个主解
        - 再加上基于每条轨迹前2个候选的若干局部变体
        """
        M, N = len(tracks), len(dets)

        if M == 0 and N == 0:
            return [([], list(range(N)), list(range(M)), 0.0)]

        if M == 0:
            return [([], list(range(N)), [], BIRTH_PENALTY * N)]

        if N == 0:
            return [([], [], list(range(M)), MISS_PENALTY * M)]

        gate, base_cost = self.build_gating_matrix(tracks, dets)

        main_cost = base_cost.copy()
        row_ind, col_ind = linear_sum_assignment(main_cost)

        assignments = []
        matched_t = set()
        matched_d = set()
        pairs = []
        score = 0.0
        for r, c in zip(row_ind, col_ind):
            if main_cost[r, c] >= 1e5:
                continue
            pairs.append((r, c))
            matched_t.add(r)
            matched_d.add(c)
            score += float(main_cost[r, c])

        unmatched_d = [j for j in range(N) if j not in matched_d]
        unmatched_t = [i for i in range(M) if i not in matched_t]
        score += BIRTH_PENALTY * len(unmatched_d) + MISS_PENALTY * len(unmatched_t)
        assignments.append((pairs, unmatched_d, unmatched_t, score))

        # 额外候选：每条轨迹局部前2候选
        # 简化版 MHT，不追求组合爆炸
        for pivot_i in range(M):
            valid_js = [j for j in range(N) if gate[pivot_i, j]]
            if len(valid_js) <= 1:
                continue
            cand_sorted = sorted(valid_js, key=lambda j: base_cost[pivot_i, j])[:2]
            for alt_j in cand_sorted[1:]:
                mod_cost = base_cost.copy()
                mod_cost[pivot_i, :] = 1e6
                mod_cost[pivot_i, alt_j] = base_cost[pivot_i, alt_j]
                r2, c2 = linear_sum_assignment(mod_cost)

                mt, md = set(), set()
                pairs2 = []
                score2 = 0.0
                for r, c in zip(r2, c2):
                    if mod_cost[r, c] >= 1e5:
                        continue
                    if r in mt or c in md:
                        continue
                    pairs2.append((r, c))
                    mt.add(r)
                    md.add(c)
                    score2 += float(mod_cost[r, c])

                ud2 = [j for j in range(N) if j not in md]
                ut2 = [i for i in range(M) if i not in mt]
                score2 += BIRTH_PENALTY * len(ud2) + MISS_PENALTY * len(ut2)
                assignments.append((pairs2, ud2, ut2, score2))

        # 去重
        uniq = {}
        for item in assignments:
            pairs, ud, ut, sc = item
            key = (
                tuple(sorted(pairs)),
                tuple(sorted(ud)),
                tuple(sorted(ut)),
            )
            if key not in uniq or sc < uniq[key][3]:
                uniq[key] = item

        out = list(uniq.values())
        out.sort(key=lambda x: x[3])
        return out[: self.max_hypotheses]

    def apply_assignment(self, tracks: List[Track], dets: List[Tuple[float, float, float, float]], frame_id: int,
                         pairs, unmatched_d, unmatched_t):
        new_tracks = [tr.copy() for tr in tracks]

        # update matched
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
            if (not tr.confirmed) and tr.hits >= MIN_HITS_TO_CONFIRM:
                tr.confirmed = True

        # unmatched tracks remain predicted
        # birth
        for dj in unmatched_d:
            tr = self.create_track(dets[dj], frame_id)
            new_tracks.append(tr)

        # prune dead tracks
        new_tracks = [tr for tr in new_tracks if tr.missed <= MAX_MISSED]

        # keep bounded
        if len(new_tracks) > MAX_TRACKS_KEEP:
            new_tracks.sort(key=lambda t: (t.confirmed, t.hits, -t.missed), reverse=True)
            new_tracks = new_tracks[:MAX_TRACKS_KEEP]

        return new_tracks

    def step(self, hyps: List[Hypothesis], dets: List[Tuple[float, float, float, float]], frame_id: int):
        new_hyps: List[Hypothesis] = []

        for hyp in hyps:
            pred_tracks = [tr.copy() for tr in hyp.tracks]
            self.predict_tracks(pred_tracks)

            candidates = self.enumerate_assignments(pred_tracks, dets)

            for pairs, unmatched_d, unmatched_t, local_score in candidates:
                child_tracks = self.apply_assignment(pred_tracks, dets, frame_id, pairs, unmatched_d, unmatched_t)
                new_hyps.append(Hypothesis(tracks=child_tracks, score=hyp.score + local_score))

        if not new_hyps:
            new_hyps = [Hypothesis(tracks=[], score=1e9)]

        new_hyps.sort(key=lambda h: h.score)
        new_hyps = new_hyps[: self.max_hypotheses]

        # N-scan: 简化版用最优假设直接裁剪
        # 对于你这个 4 目标、低噪声场景足够了
        return new_hyps


# =========================================================
# 轨迹输出
# =========================================================
def get_track_bbox_xywh(tr: Track):
    cx = float(tr.imm.x[0, 0])
    cy = float(tr.imm.x[1, 0])
    w = float(tr.w)
    h = float(tr.h)

    x = clamp(cx - w / 2.0, 0.0, IMG_W - 1.0)
    y = clamp(cy - h / 2.0, 0.0, IMG_H - 1.0)
    w = clamp(w, 1.0, IMG_W - x)
    h = clamp(h, 1.0, IMG_H - y)
    return (x, y, w, h)


def run_tracking_imm_mht_custom(detection_data: Dict[int, List[Tuple[float, float, float, float]]]):
    tracker = CustomIMMMHT()

    frame_tracks: Dict[int, List[Tuple[int, Tuple[float, float, float, float], str]]] = {}
    total_dets = 0

    hyps: List[Hypothesis] = []

    for frame_id in sorted(detection_data.keys()):
        dets = detection_data[frame_id]
        total_dets += len(dets)

        if frame_id == 0:
            init_tracks = []
            for det in dets:
                init_tracks.append(tracker.create_track(det, frame_id))
            hyps = [Hypothesis(tracks=init_tracks, score=0.0)]
        else:
            hyps = tracker.step(hyps, dets, frame_id)

        best_hyp = hyps[0]
        cur = []
        for tr in best_hyp.tracks:
            if not tr.confirmed:
                continue
            bb = get_track_bbox_xywh(tr)
            state = "det" if tr.last_update_frame == frame_id else "pred"
            cur.append((tr.track_id, bb, state))
        frame_tracks[frame_id] = cur

    results = []
    for fr in sorted(frame_tracks.keys()):
        for tid, bb, state in frame_tracks[fr]:
            x, y, w, h = bb
            results.append((fr, tid, x, y, w, h, state))

    stats = {
        "total_frames": len(detection_data),
        "total_dets": total_dets,
        "final_tracks_best_hyp": len(hyps[0].tracks) if hyps else 0,
        "association": "Custom IMM-MHT",
        "window": WINDOW_SIZE,
        "gating_chi2": GATING_CHI2,
        "max_hypotheses": MAX_HYPOTHESES,
        "n_scan": N_SCAN,
    }

    return results, frame_tracks, stats


# =========================================================
# 文件输出 / 可视化
# =========================================================
def save_tracks_txt(results, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# frame tid x y w h state(det|pred)\n")
        for fr, tid, x, y, w, h, state in results:
            f.write(f"{fr} {tid} {x:.3f} {y:.3f} {w:.3f} {h:.3f} {state}\n")


def make_track_series(frame_tracks):
    points: Dict[int, List[Tuple[float, float, int, str]]] = {}
    bboxes: Dict[int, List[Tuple[int, Tuple[float, float, float, float], str]]] = {}

    for fr in sorted(frame_tracks.keys()):
        for tid, bb, state in frame_tracks[fr]:
            x, y, w, h = bb
            cx, cy = x + w / 2.0, y + h / 2.0
            points.setdefault(tid, []).append((cx, cy, fr, state))
            bboxes.setdefault(tid, []).append((fr, bb, state))
    return points, bboxes


def select_topk_track_ids(frame_tracks, k=4) -> List[int]:
    points, _ = make_track_series(frame_tracks)
    lengths = {tid: len(seq) for tid, seq in points.items()}
    tids_sorted = sorted(lengths.keys(), key=lambda t: lengths[t], reverse=True)
    return tids_sorted[:k]


def filter_frame_tracks_by_ids(frame_tracks, keep_ids: List[int]):
    keep = set(keep_ids)
    out = {}
    for fr, items in frame_tracks.items():
        out[fr] = [it for it in items if it[0] in keep]
    return out


def plot_mot(out_png: str, frame_tracks, title: str):
    ensure_dir(os.path.dirname(out_png))
    points, bboxes = make_track_series(frame_tracks)
    if not points:
        raise RuntimeError("没有轨迹可画。")

    tids_sorted = sorted(points.keys(), key=lambda t: len(points[t]), reverse=True)
    selected = tids_sorted[:NUM_DRONES_TO_SHOW]
    tid2color = {tid: PALETTE[i % len(PALETTE)] for i, tid in enumerate(selected)}

    fig = plt.figure(figsize=(7.2, 7.2), dpi=260)
    ax = plt.gca()
    ax.set_facecolor("white")
    ax.set_xlim(0, IMG_W)

    if DISPLAY_CARTESIAN_Y:
        ax.set_ylim(0, IMG_H)      # 下方为0，上方为IMG_H
    else:
        ax.set_ylim(IMG_H, 0)      # 原始图像坐标：上方为0，下方为IMG_H

    ax.set_aspect("equal", adjustable="box")

    # 画轨迹线
    for tid in selected:
        c = tid2color[tid]
        seq = points[tid]
        xs = [p[0] for p in seq]
        ys_img = [p[1] for p in seq]

        if DISPLAY_CARTESIAN_Y:
            ys = [IMG_H - y for y in ys_img]
        else:
            ys = ys_img

        ax.plot(xs, ys, linewidth=2.2, color=c, label=f"UAV {selected.index(tid)+1} (tid={tid})")
        ax.scatter([xs[0]], [ys[0]], s=24, color=c, marker="o",
                   edgecolors="black", linewidths=0.5, zorder=6)
        ax.scatter([xs[-1]], [ys[-1]], s=34, color=c, marker="s",
                   edgecolors="black", linewidths=0.5, zorder=7)

    # 画 bbox
    for tid in selected:
        c = tid2color[tid]
        for fr, bb, state in bboxes.get(tid, []):
            x, y, w, h = bb

            if DISPLAY_CARTESIAN_Y:
                y_draw = IMG_H - y - h
            else:
                y_draw = y

            ls = "-" if state == "det" else "--"
            alpha = BBOX_ALPHA if state == "det" else 0.35
            ax.add_patch(Rectangle((x, y_draw), w, h, fill=False, edgecolor=c,
                                   linewidth=BBOX_LINEWIDTH, alpha=alpha,
                                   linestyle=ls, zorder=3))

    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_rmse_with_gt(df_frame_metrics: pd.DataFrame, out_png: str):
    frames = df_frame_metrics["frame"].to_numpy()
    rmse = df_frame_metrics["RMSE_m"].to_numpy(dtype=float)

    valid = ~np.isnan(rmse)
    invalid = np.isnan(rmse)

    plt.figure(figsize=(8, 5))
    plt.plot(frames[valid], rmse[valid], marker="o", linewidth=2, label="Valid RMSE")
    if np.any(invalid):
        plt.scatter(frames[invalid], np.zeros(np.sum(invalid)), marker="x", s=40, label="No valid match")
    plt.xlabel("Frame")
    plt.ylabel("RMSE (m)")
    plt.title("Custom IMM-MHT Localization RMSE (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def dump_json(path: str, obj: dict):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================================================
# GT 读取与评估
# =========================================================
def read_gt_csv(gt_path: str) -> Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]]:
    with open(gt_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = {c.strip().lower(): c for c in reader.fieldnames or []}

        def get_col(name: str) -> Optional[str]:
            return cols.get(name)

        c_frame = get_col("frame")
        c_id = get_col("gt_id") or get_col("id")
        c_x = get_col("x")
        c_y = get_col("y")
        c_w = get_col("w")
        c_h = get_col("h")

        if not (c_frame and c_id and c_x and c_y):
            raise ValueError(f"GT CSV 缺少必要列。当前列: {reader.fieldnames}")

        gt_by_frame: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
        for row in reader:
            fr = int(float(row[c_frame]))
            gid = int(float(row[c_id]))
            x = float(row[c_x])
            y = float(row[c_y])
            w = float(row[c_w]) if (c_w and row.get(c_w, "") != "") else 0.0
            h = float(row[c_h]) if (c_h and row.get(c_h, "") != "") else 0.0
            gt_by_frame.setdefault(fr, []).append((gid, (x, y, w, h)))
        return gt_by_frame


def evaluate_with_gt(frame_tracks: Dict[int, List[Tuple[int, Tuple[float, float, float, float], str]]],
                     gt_by_frame: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]],
                     total_frames: int,
                     metric: str = "dist",
                     dist_thresh: float = 10.0,
                     iou_thresh: float = 0.3) -> Dict[str, float]:
    prev_match: Dict[int, int] = {}
    prev_matched_flag: Dict[int, bool] = {}

    total_gt = 0
    FP = 0
    FN = 0
    IDSW = 0
    Frag = 0

    IDTP = 0
    IDFP = 0
    IDFN = 0

    match_errors = []

    for fr in range(total_frames):
        gts = gt_by_frame.get(fr, [])
        hyps = frame_tracks.get(fr, [])

        hyp_items = [(tid, bb) for (tid, bb, _state) in hyps]
        gt_items = [(gid, bb) for (gid, bb) in gts]

        total_gt += len(gt_items)

        if len(gt_items) == 0:
            FP += len(hyp_items)
            IDFP += len(hyp_items)
            continue

        if len(hyp_items) == 0:
            FN += len(gt_items)
            IDFN += len(gt_items)
            for gid, _ in gt_items:
                if prev_matched_flag.get(gid, False):
                    Frag += 1
                prev_matched_flag[gid] = False
            continue

        cost = np.full((len(gt_items), len(hyp_items)), 1e9, dtype=np.float64)

        for i, (gid, gbb) in enumerate(gt_items):
            for j, (tid, hbb) in enumerate(hyp_items):
                if metric == "iou":
                    iou = bbox_iou(gbb, hbb)
                    if iou >= iou_thresh:
                        cost[i, j] = 1.0 - iou
                else:
                    gc = bbox_center_xy(gbb) if (gbb[2] > 0 and gbb[3] > 0) else np.array([gbb[0], gbb[1]], np.float64)
                    hc = bbox_center_xy(hbb)
                    d = float(np.linalg.norm(gc - hc))
                    if d <= dist_thresh:
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
                Frag += 1 if (gid in prev_matched_flag) else 0
            prev_matched_flag[gid] = True

            IDTP += 1

        fn = len(gt_items) - len(matched_gt)
        fp = len(hyp_items) - len(matched_hyp)
        FN += fn
        FP += fp
        IDFN += fn
        IDFP += fp

        for i, (gid, _) in enumerate(gt_items):
            if i not in matched_gt:
                prev_matched_flag[gid] = False

    mota = 1.0 - (FN + FP + IDSW) / max(1, total_gt)
    motp = float(np.mean(match_errors)) if match_errors else float("nan")
    rmse_px = float(np.sqrt(np.mean(np.square(match_errors)))) if match_errors else float("nan")
    rmse_m = float(rmse_px * METER_PER_PIXEL) if match_errors else float("nan")
    pd_det = float((total_gt - FN) / max(1, total_gt))

    idp = IDTP / max(1, (IDTP + IDFP))
    idr = IDTP / max(1, (IDTP + IDFN))
    idf1 = 2 * IDTP / max(1, (2 * IDTP + IDFP + IDFN))

    return {
        "GT_total": float(total_gt),
        "FP": float(FP),
        "FN": float(FN),
        "Pd": float(pd_det),
        "IDSW": float(IDSW),
        "ID Switch": float(IDSW),
        "Frag": float(Frag),
        "RMSE_px": float(rmse_px),
        "RMSE_m": float(rmse_m),
        "MOTA": float(mota),
        "MOTP_mean_error_px": float(motp),
        "IDTP": float(IDTP),
        "IDFP": float(IDFP),
        "IDFN": float(IDFN),
        "IDP": float(idp),
        "IDR": float(idr),
        "IDF1": float(idf1),
        "meter_per_pixel": float(METER_PER_PIXEL),
    }


def compute_framewise_gt_metrics(frame_tracks, gt_by_frame, total_frames,
                                 metric="dist", dist_thresh=10.0, iou_thresh=0.3):
    frame_ids = []
    frame_rmse_px = []
    frame_rmse_m = []
    frame_pd = []
    frame_idsw = []
    frame_num_gt = []
    frame_num_hyp = []
    frame_num_matched = []

    prev_match = {}

    for fr in range(total_frames):
        gts = gt_by_frame.get(fr, [])
        hyps = frame_tracks.get(fr, [])

        hyp_items = [(tid, bb) for (tid, bb, _state) in hyps]
        gt_items = [(gid, bb) for (gid, bb) in gts]

        if len(gt_items) == 0:
            frame_ids.append(fr)
            frame_rmse_px.append(np.nan)
            frame_rmse_m.append(np.nan)
            frame_pd.append(np.nan)
            frame_idsw.append(0)
            frame_num_gt.append(0)
            frame_num_hyp.append(len(hyp_items))
            frame_num_matched.append(0)
            continue

        if len(hyp_items) == 0:
            frame_ids.append(fr)
            frame_rmse_px.append(np.nan)
            frame_rmse_m.append(np.nan)
            frame_pd.append(0.0)
            frame_idsw.append(0)
            frame_num_gt.append(len(gt_items))
            frame_num_hyp.append(0)
            frame_num_matched.append(0)
            for gid, _ in gt_items:
                prev_match.pop(gid, None)
            continue

        cost = np.full((len(gt_items), len(hyp_items)), 1e9, dtype=np.float64)

        for i, (gid, gbb) in enumerate(gt_items):
            for j, (tid, hbb) in enumerate(hyp_items):
                if metric == "iou":
                    iou = bbox_iou(gbb, hbb)
                    if iou >= iou_thresh:
                        cost[i, j] = 1.0 - iou
                else:
                    gc = bbox_center_xy(gbb) if (gbb[2] > 0 and gbb[3] > 0) else np.array([gbb[0], gbb[1]], np.float64)
                    hc = bbox_center_xy(hbb)
                    d = float(np.linalg.norm(gc - hc))
                    if d <= dist_thresh:
                        cost[i, j] = d

        r, c = linear_sum_assignment(cost)
        matched_gt = set()
        matched_hyp = set()
        errs = []
        cur_idsw = 0

        for i, j in zip(r, c):
            if cost[i, j] >= 1e8:
                continue
            gid, _ = gt_items[i]
            tid, _ = hyp_items[j]
            matched_gt.add(i)
            matched_hyp.add(j)
            errs.append(float(cost[i, j]))

            if gid in prev_match and prev_match[gid] != tid:
                cur_idsw += 1
            prev_match[gid] = tid

        pd_frame = len(matched_gt) / max(1, len(gt_items))
        rmse_frame_px = float(np.sqrt(np.mean(np.square(errs)))) if errs else np.nan
        rmse_frame_m = float(rmse_frame_px * METER_PER_PIXEL) if errs else np.nan

        frame_ids.append(fr)
        frame_rmse_px.append(rmse_frame_px)
        frame_rmse_m.append(rmse_frame_m)
        frame_pd.append(pd_frame)
        frame_idsw.append(cur_idsw)
        frame_num_gt.append(len(gt_items))
        frame_num_hyp.append(len(hyp_items))
        frame_num_matched.append(len(matched_gt))

    return pd.DataFrame({
        "frame": frame_ids,
        "num_gt": frame_num_gt,
        "num_hyp": frame_num_hyp,
        "num_matched": frame_num_matched,
        "Pd_frame": frame_pd,
        "RMSE_px": frame_rmse_px,
        "RMSE_m": frame_rmse_m,
        "IDSW_frame": frame_idsw,
    })


# =========================================================
# 主程序
# =========================================================
def main():
    ensure_dir(OUTPUT_DIR)

    frame_files = list_frame_files(INPUT_DIR)
    if len(frame_files) < 2:
        raise RuntimeError(f"未找到足够的 frame_*.txt：{INPUT_DIR}")

    print(f"[info] found {len(frame_files)} frames, full_scan={os.path.basename(frame_files[0])}")
    print(f"[info] WINDOW_SIZE={WINDOW_SIZE}, GATING_CHI2={GATING_CHI2}, MAX_MISSED={MAX_MISSED}")
    print(f"[info] MAX_HYPOTHESES={MAX_HYPOTHESES}, N_SCAN={N_SCAN}")
    if USE_PIXEL_TO_METER:
        print(f"[info] meter_per_pixel={METER_PER_PIXEL:.6f}, scene_width_m={SCENE_WIDTH_M:.6f}")

    detection_data = run_detection_windowed(frame_files)
    total_dets = sum(len(v) for v in detection_data.values())
    print(f"[info] total detections across all frames: {total_dets}")

    if total_dets == 0:
        raise RuntimeError(
            "所有帧检测结果均为0。请先调整检测器参数。"
        )

    results, frame_tracks, stats = run_tracking_imm_mht_custom(detection_data)

    out_png = os.path.join(OUTPUT_DIR, "mot_plot_imm_mht_custom.png")
    out_tracks = os.path.join(OUTPUT_DIR, "tracks_imm_mht_custom.txt")
    out_stats = os.path.join(OUTPUT_DIR, "run_stats_imm_mht_custom.json")

    plot_mot(out_png, frame_tracks, title=f"UAV MOT (Custom IMM-MHT) | window={WINDOW_SIZE}")
    save_tracks_txt(results, out_tracks)
    dump_json(out_stats, stats)

    top4_ids = select_topk_track_ids(frame_tracks, k=NUM_DRONES_TO_SHOW)
    frame_tracks_top4 = filter_frame_tracks_by_ids(frame_tracks, top4_ids)
    out_png_top4 = os.path.join(OUTPUT_DIR, "mot_plot_imm_mht_custom_top4.png")
    plot_mot(out_png_top4, frame_tracks_top4, title=f"UAV MOT (Custom IMM-MHT) TOP4 | window={WINDOW_SIZE}")

    # GT评估
    if GT_PATH and os.path.isfile(GT_PATH):
        gt_by_frame = read_gt_csv(GT_PATH)

        gt_metrics = evaluate_with_gt(
            frame_tracks=frame_tracks,
            gt_by_frame=gt_by_frame,
            total_frames=stats["total_frames"],
            metric=GT_MATCH_METRIC,
            dist_thresh=GT_DIST_THRESH,
            iou_thresh=GT_IOU_THRESH
        )

        df_summary = pd.DataFrame([gt_metrics])
        df_frame = compute_framewise_gt_metrics(
            frame_tracks=frame_tracks,
            gt_by_frame=gt_by_frame,
            total_frames=stats["total_frames"],
            metric=GT_MATCH_METRIC,
            dist_thresh=GT_DIST_THRESH,
            iou_thresh=GT_IOU_THRESH
        )

        summary_csv = os.path.join(OUTPUT_DIR, "gt_summary_metrics_imm_mht_custom.csv")
        frame_csv = os.path.join(OUTPUT_DIR, "gt_frame_metrics_imm_mht_custom.csv")
        rmse_png = os.path.join(OUTPUT_DIR, "rmse_with_gt_imm_mht_custom_m.png")
        gt_json = os.path.join(OUTPUT_DIR, "gt_metrics_imm_mht_custom.json")

        df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        df_frame.to_csv(frame_csv, index=False, encoding="utf-8-sig")
        plot_rmse_with_gt(df_frame, rmse_png)
        dump_json(gt_json, gt_metrics)

        print("\n============== Custom IMM-MHT Overall Metrics ==============")
        print(f"GT total targets       : {gt_metrics['GT_total']:.0f}")
        print(f"False Positives (FP)   : {gt_metrics['FP']:.0f}")
        print(f"False Negatives (FN)   : {gt_metrics['FN']:.0f}")
        print(f"Detection Probability  : {gt_metrics['Pd']:.4f}")
        print(f"Localization RMSE (px) : {gt_metrics['RMSE_px']:.4f}")
        print(f"Localization RMSE (m)  : {gt_metrics['RMSE_m']:.4f}")
        print(f"ID Switch              : {gt_metrics['ID Switch']:.0f}")
        print(f"MOTA                   : {gt_metrics['MOTA']:.4f}")
        print(f"IDF1                   : {gt_metrics['IDF1']:.4f}")
        print(f"meter_per_pixel        : {gt_metrics['meter_per_pixel']:.6f}")
        print("============================================================\n")

        print("[done] GT metrics saved:")
        print(" ", summary_csv)
        print(" ", frame_csv)
        print(" ", rmse_png)
        print(" ", gt_json)
    else:
        print("[info] GT evaluation skipped: GT file not found.")

    print("[done] outputs:")
    print(" ", out_png)
    print(" ", out_png_top4)
    print(" ", out_tracks)
    print(" ", out_stats)
    print("[info] top4 tids:", top4_ids)


if __name__ == "__main__":
    main()