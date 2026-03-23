# -*- coding: utf-8 -*-
"""
一键运行：4 UAV 轨迹（Kalman + GNN/Hungarian）
说明：
- 这里的 GNN 指 Global Nearest Neighbor（全局最近邻）
- 不是 Graph Neural Network（图神经网络）

输入:
  D:\codes\object_tracking\data\out_uav_4_edge_to_edge_bidir\txt\frame_*.txt
  D:\codes\object_tracking\data\out_uav_4_edge_to_edge_bidir\gt_tracks.csv

输出:
  D:\codes\object_tracking\results\gnn_compare_out\
    - gnn_plot.png
    - tracks.txt
    - report.txt
    - report_top4.txt
    - report_all.json
    - metrics_top4.json
    - gt_summary_metrics.csv
    - gt_frame_metrics.csv
    - rmse_with_gt_m.png
    - gt_metrics.json

依赖:
  numpy, matplotlib, scipy, pandas
以及你的工程内：
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


# =============================
# 路径配置
# =============================
INPUT_DIR = r"D:\codes\object_tracking\data\out_uav_4_edge_to_edge_bidir\txt"
OUTPUT_DIR = r"D:\codes\object_tracking\results\gnn_compare_out"
GT_PATH: Optional[str] = r"D:\codes\object_tracking\data\out_uav_4_edge_to_edge_bidir\gt_tracks.csv"

# =============================
# GT 评估参数
# =============================
GT_MATCH_METRIC = "dist"   # "dist" or "iou"
GT_DIST_THRESH = 10.0      # px
GT_IOU_THRESH = 0.3

# =============================
# px -> m 换算
# =============================
USE_PIXEL_TO_METER = True
RANGE_M = 600.0
FOV_DEG = 5.0
IMG_WIDTH_PX = 132

# =============================
# 图像尺寸
# =============================
IMG_W, IMG_H = 132, 132

# =============================
# 检测参数
# =============================
WINDOW_SIZE = 6
SHIFT_PIXEL = 4
DETECTOR_DEBUG = False
DETECTOR_MIN_Y = 0

# =============================
# GNN 跟踪参数
# =============================
DT = 1.0

# GNN关联门限
GNN_GATING_DISTANCE = 14.0
COST_UNMATCHED = 1e6

MAX_MISSED = 10
MIN_HITS_TO_CONFIRM = 2
MAX_TRACKS_KEEP = 12

PROCESS_NOISE_POS = 1.0
PROCESS_NOISE_VEL = 0.5
MEASUREMENT_NOISE_POS = 6.0
WH_SMOOTH = 0.7

# =============================
# 可视化
# =============================
DISPLAY_CARTESIAN_Y = True # True: 下方为0，上方为IMG_H
NUM_DRONES_TO_SHOW = 4
BBOX_ALPHA = 0.65
BBOX_LINEWIDTH = 1.3
PALETTE = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]


# =============================
# 工具函数
# =============================
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


# =============================
# Kalman Filter (CV)
# =============================
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


# =============================
# 检测模块
# =============================
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


# =============================
# GNN 关联：构建代价矩阵
# =============================
def build_cost_matrix_gnn(tracks: List[Track], dets_xywh: List[Tuple[float, float, float, float]]) -> np.ndarray:
    if not tracks or not dets_xywh:
        return np.empty((len(tracks), len(dets_xywh)), dtype=np.float32)

    det_centers = np.stack([bbox_center_xy(d) for d in dets_xywh], axis=0)

    cost = np.full((len(tracks), len(dets_xywh)), COST_UNMATCHED, dtype=np.float32)
    for i, tr in enumerate(tracks):
        pred = tr.kf.x[:2, 0].astype(np.float32)
        d = np.sqrt(((det_centers - pred[None, :]) ** 2).sum(axis=1))
        ok = d <= GNN_GATING_DISTANCE
        cost[i, ok] = d[ok]
    return cost


def create_track(next_id: int, frame_id: int, det_xywh: Tuple[float, float, float, float]) -> Track:
    cx, cy = bbox_center_xy(det_xywh)
    _, _, w, h = det_xywh

    kf = KalmanCV()
    kf.init_from_measurement(cx, cy, vx=0.0, vy=0.0)

    tr = Track(
        track_id=next_id,
        kf=kf,
        w=w,
        h=h,
        hits=1,
        missed=0,
        confirmed=False,
        last_update_frame=frame_id
    )
    return tr


# =============================
# GNN 单步
# =============================
def gnn_step(tracks: List[Track],
             dets_xywh: List[Tuple[float, float, float, float]],
             frame_id: int,
             next_id: int):
    # 1) 所有轨迹先预测
    for tr in tracks:
        tr.kf.predict()
        tr.missed += 1
        tr.total_predictions += 1

    # 2) GNN：全局一次性最优匹配（Hungarian）
    cost = build_cost_matrix_gnn(tracks, dets_xywh)
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

    # 3) 更新已匹配轨迹
    for r, c in matches:
        det = dets_xywh[c]
        z = bbox_center_xy(det)
        tracks[r].kf.update(z)

        _, _, w_meas, h_meas = det
        tracks[r].w = (1.0 - WH_SMOOTH) * tracks[r].w + WH_SMOOTH * w_meas
        tracks[r].h = (1.0 - WH_SMOOTH) * tracks[r].h + WH_SMOOTH * h_meas

        tracks[r].hits += 1
        tracks[r].missed = 0
        tracks[r].last_update_frame = frame_id
        tracks[r].total_updates += 1

        if (not tracks[r].confirmed) and tracks[r].hits >= MIN_HITS_TO_CONFIRM:
            tracks[r].confirmed = True

    # 4) 未匹配检测生成新轨迹
    for d_idx, det in enumerate(dets_xywh):
        if d_idx in matched_d:
            continue
        tr = create_track(next_id, frame_id, det)
        tracks.append(tr)
        next_id += 1

    # 5) 删除失配过久轨迹
    tracks = [tr for tr in tracks if tr.missed <= MAX_MISSED]

    # 6) 限制轨迹数量
    if len(tracks) > MAX_TRACKS_KEEP:
        tracks.sort(key=lambda t: (t.confirmed, t.hits, -t.missed), reverse=True)
        tracks = tracks[:MAX_TRACKS_KEEP]

    return tracks, next_id, matches


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


def run_tracking_kalman_gnn(detection_data: Dict[int, List[Tuple[float, float, float, float]]]):
    tracks: List[Track] = []
    next_id = 0

    frame_tracks: Dict[int, List[Tuple[int, Tuple[float, float, float, float], str]]] = {}
    total_matches = 0
    total_dets = 0

    for frame_id in sorted(detection_data.keys()):
        dets = detection_data[frame_id]
        total_dets += len(dets)

        tracks, next_id, matches = gnn_step(tracks, dets, frame_id, next_id)
        total_matches += len(matches)

        cur = []
        for tr in tracks:
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
        "total_matches": total_matches,
        "final_active_tracks": len(tracks),
        "association": "GNN (Global Nearest Neighbor)"
    }
    return results, frame_tracks, stats


# =============================
# 可视化 / 输出
# =============================
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

def plot_tracks(out_png: str, frame_tracks, title="UAV MOT (Kalman+GNN)"):
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

    # 显示坐标系选择
    if DISPLAY_CARTESIAN_Y:
        ax.set_ylim(0, IMG_H)      # 下方为0，上方为IMG_H
    else:
        ax.set_ylim(IMG_H, 0)      # 图像坐标：上方为0，下方为IMG_H

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

    # 画bbox
    for tid in selected:
        c = tid2color[tid]
        for fr, bb, state in bboxes.get(tid, []):
            x, y, w, h = bb

            if DISPLAY_CARTESIAN_Y:
                # 图像坐标(top-left) -> 笛卡尔显示坐标(bottom-left)
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


# =============================
# 无GT指标
# =============================
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
    active_counts = [len(frame_tracks.get(fr, [])) for fr in range(total_frames)]

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


# =============================
# GT 读取与评估
# =============================
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
            raise ValueError(f"GT CSV 缺少必要列。当前列: {reader.fieldnames}")

        gt_by_frame: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
        for row in reader:
            fr = int(float(row[c_frame]))
            gid = int(float(row[c_id]))
            x = float(row[c_x]); y = float(row[c_y])
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
    df_plot["RMSE_m"] = df_plot["RMSE_m"].ffill().bfill()

    plt.figure(figsize=(8, 5))
    plt.plot(df_plot["frame"].to_numpy(),
             df_plot["RMSE_m"].to_numpy(),
             marker='o', linewidth=2)
    plt.xlabel("Frame")
    plt.ylabel("RMSE (m)")
    plt.title("GNN Localization RMSE (m)")
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
        b = sum(1 for i in range(1, len(frames)) if frames[i] - frames[i - 1] > 1)
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
        else:
            f.write("\n[GT evaluation]\n")
            f.write("No GT file provided.\n")


def dump_json(path: str, obj: dict):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =============================
# 主程序
# =============================
def main():
    ensure_dir(OUTPUT_DIR)

    frame_files = list_frame_files(INPUT_DIR)
    if len(frame_files) < 2:
        raise RuntimeError(f"未找到足够的 frame_*.txt：{INPUT_DIR}")

    print(f"[info] found {len(frame_files)} frames, full_scan={os.path.basename(frame_files[0])}")
    print(f"[info] WINDOW_SIZE={WINDOW_SIZE}, GNN_GATING_DISTANCE={GNN_GATING_DISTANCE}, MAX_MISSED={MAX_MISSED}")
    if USE_PIXEL_TO_METER:
        print(f"[info] meter_per_pixel={METER_PER_PIXEL:.6f}, scene_width_m={SCENE_WIDTH_M:.6f}")

    detection_data = run_detection_windowed(frame_files)
    total_dets = sum(len(v) for v in detection_data.values())
    print(f"[info] total detections across all frames: {total_dets}")
    if total_dets == 0:
        raise RuntimeError("所有帧检测结果均为0。请先调整检测器参数。")

    results, frame_tracks, stats = run_tracking_kalman_gnn(detection_data)

    out_png = os.path.join(OUTPUT_DIR, "gnn_plot.png")
    out_tracks = os.path.join(OUTPUT_DIR, "tracks.txt")
    out_report = os.path.join(OUTPUT_DIR, "report.txt")

    plot_tracks(out_png, frame_tracks, title=f"UAV MOT (Kalman+GNN) | window={WINDOW_SIZE}")
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

    eval_no_gt_evaluate(out_tracks,
                        os.path.join(OUTPUT_DIR, "metrics_no_gt.json"),
                        os.path.join(OUTPUT_DIR, "report_no_gt.txt"))

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

        df_summary.to_csv(os.path.join(OUTPUT_DIR, "gt_summary_metrics.csv"), index=False, encoding="utf-8-sig")
        df_frame.to_csv(os.path.join(OUTPUT_DIR, "gt_frame_metrics.csv"), index=False, encoding="utf-8-sig")
        plot_rmse_with_gt(df_frame, os.path.join(OUTPUT_DIR, "rmse_with_gt_m.png"))
        dump_json(os.path.join(OUTPUT_DIR, "gt_metrics.json"), gt_metrics)

        print("\n================= GNN Overall Metrics =================")
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
        print("=======================================================\n")

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


if __name__ == "__main__":
    main()