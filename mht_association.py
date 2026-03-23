# -*- coding: utf-8 -*-
"""
一键运行：4 UAV 轨迹（Kalman + MHT，多假设版本）
说明：
- 这版不再是单假设 Hungarian
- 这版是多假设 MHT：每帧保留多个 hypothesis，进行打分和裁剪
- 为控制复杂度，采用工程化简化：
    Hungarian 主解 + 局部备选分支 + top-K 裁剪
- 同时保留 y 轴翻转显示：
    下方为 0，上方为 IMG_H

输入:
  D:\codes\object_tracking\data\out_uav_4_edge_to_edge_bidir\txt\frame_*.txt

输出:
  D:\codes\object_tracking\results\mot_compare_out\
    - mot_plot.png
    - tracks.txt
    - report.txt
    - report_top4.txt
    - metrics_top4.json
    - report_all.json
    - gt_summary_metrics.csv
    - gt_frame_metrics.csv
    - rmse_with_gt_m.png
    - gt_metrics.json

依赖:
  numpy, matplotlib, scipy, pandas
  src/pixel_shifting_correction.py -> RadarImageProcessor
  src/object_detection.py -> SpatialDroneDetector
  src.eval_no_gt.py -> evaluate(tracks_txt, out_json, out_txt)
"""

import os
import re
import glob
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment

from src.pixel_shifting_correction import RadarImageProcessor
from src.object_detection import SpatialDroneDetector
from src.eval_no_gt import evaluate as eval_no_gt_evaluate


# -----------------------------
# 固定路径
# -----------------------------
INPUT_DIR = r"D:\codes\object_tracking\data\out_uav_4_edge_to_edge_bidir\txt"
OUTPUT_DIR = r"D:\codes\object_tracking\results\mot_compare_out"

# -----------------------------
# GT 文件
# -----------------------------
GT_PATH: Optional[str] = r"D:\codes\object_tracking\data\out_uav_4_edge_to_edge_bidir\gt_tracks.csv"

# GT 评估参数
GT_MATCH_METRIC = "dist"     # "dist" or "iou"
GT_DIST_THRESH = 10.0
GT_IOU_THRESH = 0.3

# -----------------------------
# RMSE 单位换算：px -> m
# -----------------------------
USE_PIXEL_TO_METER = True
RANGE_M = 600.0
FOV_DEG = 5.0
IMG_WIDTH_PX = 132

# -----------------------------
# 图像尺寸
# -----------------------------
IMG_W, IMG_H = 132, 132

# -----------------------------
# 检测参数
# -----------------------------
WINDOW_SIZE = 6
SHIFT_PIXEL = 4
DETECTOR_DEBUG = False
DETECTOR_MIN_Y = 0

# -----------------------------
# 跟踪参数（Kalman + MHT）
# -----------------------------
DT = 1.0

# 门控距离（欧氏距离）
GATING_DISTANCE = 14.0

# 轨迹生命周期
MAX_MISSED = 10
MIN_HITS_TO_CONFIRM = 2
MAX_TRACKS_KEEP = 12

# Kalman 参数
PROCESS_NOISE_POS = 1.0
PROCESS_NOISE_VEL = 0.5
MEASUREMENT_NOISE_POS = 6.0

# 框平滑
WH_SMOOTH = 0.7

# MHT 参数
MAX_HYPOTHESES = 24
N_SCAN = 2   # 这里保留参数；当前实现是工程化简化版，用 top-K 剪枝为主
MISS_PENALTY = 7.0
BIRTH_PENALTY = 11.0
LAMBDA_SIZE = 0.05
COST_INF = 1e6

# -----------------------------
# 可视化
# -----------------------------
NUM_DRONES_TO_SHOW = 4
BBOX_ALPHA = 0.65
BBOX_LINEWIDTH = 1.3
PALETTE = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]

# True: 下方为0，上方为IMG_H
# False: 保持图像坐标（上方为0，下方为IMG_H）
DISPLAY_CARTESIAN_Y = True


# -----------------------------
# 工具
# -----------------------------
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
    return np.array([x + w / 2.0, y + h / 2.0], dtype=np.float32)

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


# -----------------------------
# Kalman Filter
# -----------------------------
class KalmanCV:
    """
    状态: [cx, cy, vx, vy]
    观测: [cx, cy]
    """
    def __init__(self):
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 500.0

        self.F = np.array([[1, 0, DT, 0],
                           [0, 1, 0, DT],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]], dtype=np.float32)

        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)

        q_pos = PROCESS_NOISE_POS
        q_vel = PROCESS_NOISE_VEL
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float32)

        r = MEASUREMENT_NOISE_POS
        self.R = np.diag([r, r]).astype(np.float32)

        self.I = np.eye(4, dtype=np.float32)

    def copy(self):
        new = KalmanCV()
        new.x = self.x.copy()
        new.P = self.P.copy()
        return new

    def init_from_measurement(self, cx, cy, vx=0.0, vy=0.0):
        self.x[:] = np.array([[cx], [cy], [vx], [vy]], dtype=np.float32)
        self.P[:] = np.eye(4, dtype=np.float32) * 50.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z_xy: np.ndarray):
        z = z_xy.reshape(2, 1).astype(np.float32)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        self.P = (self.I - K @ self.H) @ self.P
        return self.x


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
    total_predictions: int = 0

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
            total_predictions=int(self.total_predictions),
        )


@dataclass
class Hypothesis:
    tracks: List[Track]
    score: float = 0.0


# -----------------------------
# 检测：窗口累积
# -----------------------------
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
        min_y=DETECTOR_MIN_Y
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
            print(f"[detect] frame={frame_id:04d} window=[{start+1:04d}..{end:04d}] "
                  f"files={len(window_updates)} dets={len(detection_data[frame_id])}")

    return detection_data


# -----------------------------
# MHT 核心
# -----------------------------
class KalmanMHT:
    def __init__(self):
        self.next_id = 0
        self.max_hypotheses = MAX_HYPOTHESES
        self.n_scan = N_SCAN

    def create_track(self, frame_id: int, det_xywh: Tuple[float, float, float, float]) -> Track:
        cx, cy = bbox_center_xy(det_xywh)
        _, _, w, h = det_xywh

        kf = KalmanCV()
        kf.init_from_measurement(cx, cy, vx=0.0, vy=0.0)

        tr = Track(
            track_id=self.next_id,
            kf=kf,
            w=w,
            h=h,
            hits=1,
            missed=0,
            confirmed=False,
            last_update_frame=frame_id,
            total_updates=0,
            total_predictions=0,
        )
        self.next_id += 1
        return tr

    def predict_tracks(self, tracks: List[Track]):
        for tr in tracks:
            tr.kf.predict()
            tr.missed += 1
            tr.total_predictions += 1

    def build_cost_matrix(self, tracks: List[Track], dets_xywh: List[Tuple[float, float, float, float]]) -> np.ndarray:
        if not tracks or not dets_xywh:
            return np.empty((len(tracks), len(dets_xywh)), dtype=np.float32)

        det_centers = np.stack([bbox_center_xy(d) for d in dets_xywh], axis=0)
        cost = np.full((len(tracks), len(dets_xywh)), COST_INF, dtype=np.float32)

        for i, tr in enumerate(tracks):
            pred = tr.kf.x[:2, 0].astype(np.float32)
            d = np.sqrt(((det_centers - pred[None, :]) ** 2).sum(axis=1))
            ok = d <= GATING_DISTANCE
            for j in np.where(ok)[0]:
                size_cost = LAMBDA_SIZE * (abs(dets_xywh[j][2] - tr.w) + abs(dets_xywh[j][3] - tr.h))
                cost[i, j] = d[j] + size_cost
        return cost

    def enumerate_assignments(self, tracks: List[Track], dets_xywh: List[Tuple[float, float, float, float]]):
        """
        输出若干候选关联：
        - Hungarian 主解
        - 对每条轨迹构造局部备选解（强制第二候选）
        - 对每条轨迹构造 miss 分支
        """
        M, N = len(tracks), len(dets_xywh)

        if M == 0 and N == 0:
            return [([], list(range(N)), list(range(M)), 0.0)]

        if M == 0:
            return [([], list(range(N)), [], BIRTH_PENALTY * N)]

        if N == 0:
            return [([], [], list(range(M)), MISS_PENALTY * M)]

        cost = self.build_cost_matrix(tracks, dets_xywh)
        assignments = []

        def solve_from_cost(mat):
            row_ind, col_ind = linear_sum_assignment(mat)
            matched_t = set()
            matched_d = set()
            pairs = []
            score = 0.0

            for r, c in zip(row_ind, col_ind):
                if mat[r, c] >= COST_INF * 0.5:
                    continue
                if r in matched_t or c in matched_d:
                    continue
                pairs.append((r, c))
                matched_t.add(r)
                matched_d.add(c)
                score += float(mat[r, c])

            unmatched_d = [j for j in range(N) if j not in matched_d]
            unmatched_t = [i for i in range(M) if i not in matched_t]
            score += BIRTH_PENALTY * len(unmatched_d) + MISS_PENALTY * len(unmatched_t)

            return pairs, unmatched_d, unmatched_t, score

        # 1) 主解
        assignments.append(solve_from_cost(cost.copy()))

        # 2) 局部备选：强制某条轨迹走第二候选
        for i in range(M):
            valid_js = [j for j in range(N) if cost[i, j] < COST_INF * 0.5]
            if len(valid_js) <= 1:
                continue

            cand_sorted = sorted(valid_js, key=lambda j: float(cost[i, j]))[:2]
            alt_j = cand_sorted[1]

            mod_cost = cost.copy()
            mod_cost[i, :] = COST_INF
            mod_cost[i, alt_j] = cost[i, alt_j]
            assignments.append(solve_from_cost(mod_cost))

        # 3) miss 分支：强制某条轨迹本帧不匹配
        for i in range(M):
            has_valid = np.any(cost[i, :] < COST_INF * 0.5)
            if not has_valid:
                continue
            mod_cost = cost.copy()
            mod_cost[i, :] = COST_INF
            assignments.append(solve_from_cost(mod_cost))

        # 去重 + 保留最好若干个
        uniq = {}
        for item in assignments:
            pairs, ud, ut, sc = item
            key = (tuple(sorted(pairs)), tuple(sorted(ud)), tuple(sorted(ut)))
            if key not in uniq or sc < uniq[key][3]:
                uniq[key] = item

        out = list(uniq.values())
        out.sort(key=lambda x: x[3])
        return out[:self.max_hypotheses]

    def apply_assignment(self,
                         tracks: List[Track],
                         dets_xywh: List[Tuple[float, float, float, float]],
                         frame_id: int,
                         pairs,
                         unmatched_d,
                         unmatched_t):
        new_tracks = [tr.copy() for tr in tracks]

        # 已匹配轨迹：更新
        for r, c in pairs:
            det = dets_xywh[c]
            z = bbox_center_xy(det)

            new_tracks[r].kf.update(z)

            _, _, w_meas, h_meas = det
            new_tracks[r].w = (1.0 - WH_SMOOTH) * new_tracks[r].w + WH_SMOOTH * w_meas
            new_tracks[r].h = (1.0 - WH_SMOOTH) * new_tracks[r].h + WH_SMOOTH * h_meas

            new_tracks[r].hits += 1
            new_tracks[r].missed = 0
            new_tracks[r].last_update_frame = frame_id
            new_tracks[r].total_updates += 1

            if (not new_tracks[r].confirmed) and new_tracks[r].hits >= MIN_HITS_TO_CONFIRM:
                new_tracks[r].confirmed = True

        # 未匹配检测：birth
        for d_idx in unmatched_d:
            tr = self.create_track(frame_id, dets_xywh[d_idx])
            new_tracks.append(tr)

        # 未匹配轨迹：保持预测状态，不用额外处理；missed 已在 predict 阶段加过

        # 删除长时间失配轨迹
        new_tracks = [tr for tr in new_tracks if tr.missed <= MAX_MISSED]

        # 限制轨迹数量
        if len(new_tracks) > MAX_TRACKS_KEEP:
            new_tracks.sort(key=lambda t: (t.confirmed, t.hits, -t.missed), reverse=True)
            new_tracks = new_tracks[:MAX_TRACKS_KEEP]

        return new_tracks

    def step(self, hyps: List[Hypothesis], dets_xywh: List[Tuple[float, float, float, float]], frame_id: int):
        new_hyps: List[Hypothesis] = []

        for hyp in hyps:
            pred_tracks = [tr.copy() for tr in hyp.tracks]
            self.predict_tracks(pred_tracks)

            candidates = self.enumerate_assignments(pred_tracks, dets_xywh)

            for pairs, unmatched_d, unmatched_t, local_score in candidates:
                child_tracks = self.apply_assignment(
                    pred_tracks, dets_xywh, frame_id, pairs, unmatched_d, unmatched_t
                )
                new_hyps.append(Hypothesis(tracks=child_tracks, score=hyp.score + local_score))

        if not new_hyps:
            new_hyps = [Hypothesis(tracks=[], score=1e9)]

        new_hyps.sort(key=lambda h: h.score)
        new_hyps = new_hyps[:self.max_hypotheses]

        return new_hyps


# -----------------------------
# 轨迹输出
# -----------------------------
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

def run_tracking_kalman_mht(detection_data: Dict[int, List[Tuple[float, float, float, float]]]):
    tracker = KalmanMHT()

    frame_tracks: Dict[int, List[Tuple[int, Tuple[float, float, float, float], str]]] = {}
    total_dets = 0
    hyps: List[Hypothesis] = []

    for frame_id in sorted(detection_data.keys()):
        dets = detection_data[frame_id]
        total_dets += len(dets)

        if frame_id == 0:
            init_tracks = []
            for det in dets:
                init_tracks.append(tracker.create_track(frame_id, det))
            hyps = [Hypothesis(tracks=init_tracks, score=0.0)]
        else:
            hyps = tracker.step(hyps, dets, frame_id)

        best_hyp = hyps[0]
        cur = []
        for tr in best_hyp.tracks:
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
        "association": "Kalman + MHT",
        "window": WINDOW_SIZE,
        "gating_distance": GATING_DISTANCE,
        "max_hypotheses": MAX_HYPOTHESES,
        "n_scan": N_SCAN,
    }
    return results, frame_tracks, stats


# -----------------------------
# 输出与可视化
# -----------------------------
def save_tracks_txt(results, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# frame tid x y w h state(det|pred)\n")
        for fr, tid, x, y, w, h, state in results:
            f.write(f"{fr} {tid} {x:.3f} {y:.3f} {w:.3f} {h:.3f} {state}\n")

def make_track_series(frame_tracks: Dict[int, List[Tuple[int, Tuple[float, float, float, float], str]]]):
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

def plot_mot(out_png: str, frame_tracks):
    ensure_dir(os.path.dirname(out_png))

    points, bboxes = make_track_series(frame_tracks)
    if not points:
        raise RuntimeError("没有轨迹可画：检测阶段未输出bbox或跟踪未产生轨迹。")

    tids_sorted = sorted(points.keys(), key=lambda t: len(points[t]), reverse=True)
    selected = tids_sorted[:NUM_DRONES_TO_SHOW]
    tid2color = {tid: PALETTE[i % len(PALETTE)] for i, tid in enumerate(selected)}

    fig = plt.figure(figsize=(7.2, 7.2), dpi=260)
    ax = plt.gca()
    ax.set_facecolor("white")
    ax.set_xlim(0, IMG_W)

    if DISPLAY_CARTESIAN_Y:
        ax.set_ylim(0, IMG_H)
    else:
        ax.set_ylim(IMG_H, 0)

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

    ax.set_title(f"UAV MOT (Kalman+MHT) | window={WINDOW_SIZE}", fontsize=12)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# 评价：无GT
# -----------------------------
def compute_no_gt_metrics(frame_tracks: Dict[int, List[Tuple[int, Tuple[float, float, float, float], str]]],
                          total_frames: int) -> Dict[str, float]:
    points, _ = make_track_series(frame_tracks)
    if not points:
        return {"num_tracks_total": 0.0}

    lengths = {tid: len(seq) for tid, seq in points.items()}
    tids_sorted = sorted(lengths.keys(), key=lambda t: lengths[t], reverse=True)
    top4 = tids_sorted[:4]

    breaks = []
    pred_ratios = []
    mean_speeds = []
    mean_accs = []

    for tid, seq in points.items():
        frames = np.array([p[2] for p in seq], dtype=np.int32)
        states = [p[3] for p in seq]
        br = int(np.sum((frames[1:] - frames[:-1]) > 1)) if len(frames) > 1 else 0
        breaks.append(br)

        pr = states.count("pred") / max(1, len(states))
        pred_ratios.append(pr)

        xy = np.array([(p[0], p[1]) for p in seq], dtype=np.float32)
        if len(xy) >= 2:
            v = np.linalg.norm(xy[1:] - xy[:-1], axis=1) / DT
            mean_speeds.append(float(np.mean(v)))
            if len(v) >= 2:
                a = np.abs(v[1:] - v[:-1]) / DT
                mean_accs.append(float(np.mean(a)))
        else:
            mean_speeds.append(0.0)
            mean_accs.append(0.0)

    top4_coverage = sum(lengths[t] for t in top4) / max(1, sum(lengths.values()))

    active_counts = []
    for fr in range(total_frames):
        active_counts.append(len(frame_tracks.get(fr, [])))

    return {
        "num_tracks_total": float(len(points)),
        "top4_coverage_ratio": float(top4_coverage),
        "avg_track_len": float(np.mean(list(lengths.values()))),
        "median_track_len": float(np.median(list(lengths.values()))),
        "avg_breaks_per_track": float(np.mean(breaks)),
        "median_breaks_per_track": float(np.median(breaks)),
        "avg_pred_ratio": float(np.mean(pred_ratios)),
        "median_pred_ratio": float(np.median(pred_ratios)),
        "avg_speed(px/frame)": float(np.mean(mean_speeds)),
        "avg_acc(px/frame^2)": float(np.mean(mean_accs)),
        "avg_active_tracks_per_frame": float(np.mean(active_counts)),
        "max_active_tracks_in_a_frame": float(np.max(active_counts)),
    }


# -----------------------------
# 读取GT
# -----------------------------
def read_gt_csv(gt_path: str) -> Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]]:
    import csv

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
            raise ValueError(
                "GT CSV 需要至少列：frame, gt_id, x, y （可选 w,h）\n"
                f"当前列: {reader.fieldnames}"
            )

        gt_by_frame: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
        for row in reader:
            fr = int(float(row[c_frame]))
            gid = int(float(row[c_id]))
            x = float(row[c_x]); y = float(row[c_y])
            w = float(row[c_w]) if (c_w and row.get(c_w, "") != "") else 0.0
            h = float(row[c_h]) if (c_h and row.get(c_h, "") != "") else 0.0
            gt_by_frame.setdefault(fr, []).append((gid, (x, y, w, h)))
        return gt_by_frame


# -----------------------------
# GT评估：总指标
# -----------------------------
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

        cost = np.full((len(gt_items), len(hyp_items)), 1e9, dtype=np.float32)

        for i, (gid, gbb) in enumerate(gt_items):
            for j, (tid, hbb) in enumerate(hyp_items):
                if metric == "iou":
                    iou = bbox_iou(gbb, hbb)
                    if iou >= iou_thresh:
                        cost[i, j] = 1.0 - iou
                else:
                    gc = bbox_center_xy(gbb) if (gbb[2] > 0 and gbb[3] > 0) else np.array([gbb[0], gbb[1]], np.float32)
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


# -----------------------------
# GT评估：逐帧指标
# -----------------------------
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

        cost = np.full((len(gt_items), len(hyp_items)), 1e9, dtype=np.float32)

        for i, (gid, gbb) in enumerate(gt_items):
            for j, (tid, hbb) in enumerate(hyp_items):
                if metric == "iou":
                    iou = bbox_iou(gbb, hbb)
                    if iou >= iou_thresh:
                        cost[i, j] = 1.0 - iou
                else:
                    gc = bbox_center_xy(gbb) if (gbb[2] > 0 and gbb[3] > 0) else np.array([gbb[0], gbb[1]], np.float32)
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


def plot_rmse_with_gt(df_frame_metrics: pd.DataFrame, out_png: str):
    df_plot = df_frame_metrics.copy()
    df_plot["RMSE_m"] = df_plot["RMSE_m"].ffill()

    plt.figure(figsize=(8, 5))
    plt.plot(df_plot["frame"].to_numpy(),
             df_plot["RMSE_m"].to_numpy(),
             marker='o', linewidth=2)
    plt.xlabel("Frame")
    plt.ylabel("RMSE (m)")
    plt.title("MHT Localization RMSE (m)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def write_report(out_path: str, stats: dict, frame_tracks, gt_path: Optional[str] = None):
    points, _ = make_track_series(frame_tracks)
    lengths = {tid: len(seq) for tid, seq in points.items()}
    top = sorted(lengths.items(), key=lambda kv: kv[1], reverse=True)[:10]

    breaks = {}
    for tid, seq in points.items():
        frames = [p[2] for p in seq]
        b = sum(1 for i in range(1, len(frames)) if frames[i] - frames[i-1] > 1)
        breaks[tid] = b

    no_gt = compute_no_gt_metrics(frame_tracks, total_frames=stats["total_frames"])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== Tracking report ===\n\n")
        f.write("[Run stats]\n")
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")

        f.write("\n[No-GT quality metrics]\n")
        for k, v in no_gt.items():
            f.write(f"{k}: {v}\n")

        f.write("\n[Top tracks by length]\n")
        for tid, L in top:
            f.write(f"  tid={tid}, len={L}, breaks={breaks.get(tid,0)}\n")

        if gt_path:
            f.write("\n[GT evaluation]\n")
            try:
                gt_by_frame = read_gt_csv(gt_path)
                gt_metrics = evaluate_with_gt(
                    frame_tracks=frame_tracks,
                    gt_by_frame=gt_by_frame,
                    total_frames=stats["total_frames"],
                    metric=GT_MATCH_METRIC,
                    dist_thresh=GT_DIST_THRESH,
                    iou_thresh=GT_IOU_THRESH
                )
                f.write(f"gt_file: {gt_path}\n")
                f.write(f"match_metric: {GT_MATCH_METRIC}\n")
                if GT_MATCH_METRIC == "dist":
                    f.write(f"dist_thresh: {GT_DIST_THRESH}\n")
                else:
                    f.write(f"iou_thresh: {GT_IOU_THRESH}\n")
                for k, v in gt_metrics.items():
                    f.write(f"{k}: {v}\n")
            except Exception as e:
                f.write(f"GT evaluation failed: {repr(e)}\n")
                f.write("Check GT CSV format: frame,gt_id,x,y,(optional w,h)\n")
        else:
            f.write("\n[GT evaluation]\n")
            f.write("No GT file provided.\n")

def dump_json(path: str, obj: dict):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -----------------------------
# 主程序
# -----------------------------
def main():
    ensure_dir(OUTPUT_DIR)

    frame_files = list_frame_files(INPUT_DIR)
    if len(frame_files) < 2:
        raise RuntimeError(f"未找到足够的 frame_*.txt：{INPUT_DIR}")

    print(f"[info] found {len(frame_files)} frames, full_scan={os.path.basename(frame_files[0])}")
    print(f"[info] WINDOW_SIZE={WINDOW_SIZE}, GATING_DISTANCE={GATING_DISTANCE}, MAX_MISSED={MAX_MISSED}")
    print(f"[info] MAX_HYPOTHESES={MAX_HYPOTHESES}, N_SCAN={N_SCAN}")
    if USE_PIXEL_TO_METER:
        print(f"[info] meter_per_pixel={METER_PER_PIXEL:.6f}, scene_width_m={SCENE_WIDTH_M:.6f}")

    detection_data = run_detection_windowed(frame_files)
    total_dets = sum(len(v) for v in detection_data.values())
    print(f"[info] total detections across all frames: {total_dets}")
    if total_dets == 0:
        raise RuntimeError(
            "所有帧检测结果均为0：请先让 SpatialDroneDetector 在每帧能输出bbox。\n"
            "优先调整 WINDOW_SIZE，或在 SpatialDroneDetector 内调整阈值/连通域面积过滤。"
        )

    results, frame_tracks, stats = run_tracking_kalman_mht(detection_data)

    out_png = os.path.join(OUTPUT_DIR, "mot_plot.png")
    out_tracks = os.path.join(OUTPUT_DIR, "tracks.txt")
    out_report = os.path.join(OUTPUT_DIR, "report.txt")

    plot_mot(out_png, frame_tracks)
    save_tracks_txt(results, out_tracks)
    write_report(out_report, stats, frame_tracks, gt_path=GT_PATH)

    top4_ids = select_topk_track_ids(frame_tracks, k=NUM_DRONES_TO_SHOW)
    frame_tracks_top4 = filter_frame_tracks_by_ids(frame_tracks, top4_ids)

    out_report_top4 = os.path.join(OUTPUT_DIR, "report_top4.txt")
    write_report(out_report_top4, stats, frame_tracks_top4, gt_path=GT_PATH)

    metrics_all = compute_no_gt_metrics(frame_tracks, total_frames=stats["total_frames"])
    metrics_top4 = compute_no_gt_metrics(frame_tracks_top4, total_frames=stats["total_frames"])
    dump_json(os.path.join(OUTPUT_DIR, "report_all.json"), {"run": stats, "no_gt": metrics_all})
    dump_json(os.path.join(OUTPUT_DIR, "metrics_top4.json"), {"run": stats, "top4_ids": top4_ids, "no_gt": metrics_top4})

    # 外部无GT评估器
    eval_no_gt_evaluate(out_tracks,
                        os.path.join(OUTPUT_DIR, "metrics_no_gt.json"),
                        os.path.join(OUTPUT_DIR, "report_no_gt.txt"))

    # ---------------- GT输出 ----------------
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

        df_summary.to_csv(os.path.join(OUTPUT_DIR, "gt_summary_metrics.csv"), index=False, encoding="utf-8-sig")
        df_frame.to_csv(os.path.join(OUTPUT_DIR, "gt_frame_metrics.csv"), index=False, encoding="utf-8-sig")

        plot_rmse_with_gt(df_frame, os.path.join(OUTPUT_DIR, "rmse_with_gt_m.png"))
        dump_json(os.path.join(OUTPUT_DIR, "gt_metrics.json"), gt_metrics)

        print("\n================= GT Overall Metrics =================")
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
        print("=====================================================\n")

        print("[done] GT metrics saved:")
        print(" ", os.path.join(OUTPUT_DIR, "gt_summary_metrics.csv"))
        print(" ", os.path.join(OUTPUT_DIR, "gt_frame_metrics.csv"))
        print(" ", os.path.join(OUTPUT_DIR, "rmse_with_gt_m.png"))
        print(" ", os.path.join(OUTPUT_DIR, "gt_metrics.json"))
    else:
        print("[info] GT evaluation skipped: GT file not found.")

    print("[done] outputs:")
    print(" ", out_png)
    print(" ", out_tracks)
    print(" ", out_report)
    print(" ", out_report_top4)
    print(" ", os.path.join(OUTPUT_DIR, "metrics_top4.json"))
    print("[info] top4 tids:", top4_ids)


if __name__ == "__main__":
    main()