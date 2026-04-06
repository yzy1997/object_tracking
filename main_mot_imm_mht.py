# main_mot_imm_mht.py
# -*- coding: utf-8 -*-

import os
import re
import glob
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.pixel_shifting_correction import RadarImageProcessor
from src.object_detection import SpatialDroneDetector
from src.imm_mht_tracker import IMMMHTTracker, IMMTrack


# -----------------------------
# 固定路径（按你真实目录）
# -----------------------------
INPUT_DIR = r"D:\codes\object_tracking\data\out_uav_4_edge_to_edge_bidir\txt"
OUTPUT_DIR = r"D:\codes\object_tracking\results\mot_imm_mht_out"


# -----------------------------
# 图像尺寸
# -----------------------------
IMG_W, IMG_H = 132, 132


# -----------------------------
# 检测：窗口累积参数（关键）
# -----------------------------
WINDOW_SIZE = 6
SHIFT_PIXEL = 4
DETECTOR_DEBUG = False
DETECTOR_MIN_Y = 0


# -----------------------------
# IMM-MHT 参数（优化版 - 降低ID Switch和误跟率）
# -----------------------------
DT = 1.0
MAX_MISSED = 6              # 更严格：减少最大丢失帧数
MIN_HITS_TO_CONFIRM = 3     # 增加确认所需命中次数

# 统计门控：2D 卡方 95% (更严格，减少误跟)
GATING_CHI2 = 4.0           # 更严格的门控

# 假设剪枝
MAX_HYPOTHESES = 30         # 增加假设数量
N_SCAN = 5                  # 增加N-scan深度

# 4 UAV 先验（避免轨迹爆炸）
MAX_CONFIRMED = 4
MAX_TRACKS_KEEP = 10

# 打分（优化：大幅增加惩罚减少虚假轨迹）
MISS_PENALTY = 10.0         # 更高惩罚
BIRTH_PENALTY = 20.0        # 更高出生惩罚

# cost 中速度/尺寸权重（增加速度一致性权重减少ID切换）
LAMBDA_V = 0.8              # 更高的速度一致性权重
LAMBDA_WH = 0.1             # 增加尺寸一致性权重

# KF 测量噪声（降低，更信任测量）
R_POS = 4.0                 # 更信任测量值

# IMM 两个模型的过程噪声（更平滑）
IMM_Q0 = (0.3, 0.08)
IMM_Q1 = (1.0, 0.5)

WH_SMOOTH = 0.8             # 更高的尺寸平滑


# -----------------------------
# 可视化
# -----------------------------
NUM_DRONES_TO_SHOW = 4
BBOX_ALPHA = 0.65
BBOX_LINEWIDTH = 1.3
PALETTE = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]  # 蓝 绿 橙 紫


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


# -----------------------------
# 检测：窗口累积（保持与你 main_mot.py 一致）
# -----------------------------
def run_detection_windowed(frame_files: List[str]) -> Dict[int, List[Tuple[float, float, float, float]]]:
    if len(frame_files) < 2:
        raise RuntimeError("至少需要 frame_0000 + 一个 update 帧")

    full_scan = frame_files[0]
    updates = frame_files[1:]  # frame 1..N

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
        window_updates = updates[start:end]  # updates[0] corresponds to frame_0001

        detector.update_files = window_updates
        boxes_xyxy = detector.detect()
        detection_data[frame_id] = xyxy_to_xywh_list(boxes_xyxy)

        if (frame_id % 50 == 0) or (frame_id < 5):
            print(
                f"[detect] frame={frame_id:04d} window=[{start+1:04d}..{end:04d}] "
                f"files={len(window_updates)} dets={len(detection_data[frame_id])}"
            )

    return detection_data


# -----------------------------
# IMM-MHT 运行
# -----------------------------
def run_tracking_imm_mht(detection_data: Dict[int, List[Tuple[float, float, float, float]]]):
    tracker = IMMMHTTracker(
        dt=DT,
        gating_chi2=GATING_CHI2,
        max_hypotheses=MAX_HYPOTHESES,
        n_scan=N_SCAN,
        max_missed=MAX_MISSED,
        min_hits_to_confirm=MIN_HITS_TO_CONFIRM,
        max_confirmed=MAX_CONFIRMED,
        max_tracks_keep=MAX_TRACKS_KEEP,
        miss_penalty=MISS_PENALTY,
        birth_penalty=BIRTH_PENALTY,
        lambda_v=LAMBDA_V,
        lambda_wh=LAMBDA_WH,
        r_pos=R_POS,
        imm_q0=IMM_Q0,
        imm_q1=IMM_Q1,
        wh_smooth=WH_SMOOTH,
    )

    tracks: List[IMMTrack] = []
    next_id = 0
    frame_tracks: Dict[int, List[Tuple[int, Tuple[float, float, float, float], str]]] = {}

    for frame_id in sorted(detection_data.keys()):
        dets = detection_data[frame_id]

        if frame_id == 0:
            # init from detections at frame 0
            for det in dets:
                tr = tracker.create_track(next_id, frame_id, det)
                tracks.append(tr)
                next_id += 1
            tracker.initialize(tracks)
        else:
            tracks, next_id = tracker.step(dets, frame_id=frame_id, next_id=next_id)

        # save per-frame
        cur = []
        for tr in tracks:
            cx = float(tr.imm.x[0, 0])
            cy = float(tr.imm.x[1, 0])
            w = float(tr.w)
            h = float(tr.h)

            x = clamp(cx - w / 2.0, 0.0, IMG_W - 1.0)
            y = clamp(cy - h / 2.0, 0.0, IMG_H - 1.0)
            w = clamp(w, 1.0, IMG_W - x)
            h = clamp(h, 1.0, IMG_H - y)

            state = "det" if tr.last_update_frame == frame_id else "pred"
            cur.append((tr.track_id, (x, y, w, h), state))

        frame_tracks[frame_id] = cur

    # flatten results
    results = []
    for fr, items in frame_tracks.items():
        for tid, bb, state in items:
            x, y, w, h = bb
            results.append((fr, tid, x, y, w, h, state))

    stats = {
        "total_frames": len(detection_data),
        "final_tracks": len(tracks),
        "association": "IMM-MHT",
        "window": WINDOW_SIZE,
        "gating_chi2": GATING_CHI2,
        "lambda_v": LAMBDA_V,
        "lambda_wh": LAMBDA_WH,
        "max_confirmed": MAX_CONFIRMED,
    }

    return results, frame_tracks, stats


# -----------------------------
# 输出：tracks.txt
# -----------------------------
def save_tracks_txt(results, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# frame tid x y w h state(det|pred)\n")
        for fr, tid, x, y, w, h, state in results:
            f.write(f"{fr} {tid} {x:.3f} {y:.3f} {w:.3f} {h:.3f} {state}\n")


# -----------------------------
# 轨迹序列与 TopK
# -----------------------------
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


# -----------------------------
# 绘图（保持你的风格）
# 注意：y 轴反向用于和你之前一致的显示习惯
# -----------------------------
def plot_mot(out_png: str, frame_tracks, title: str):
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
    ax.set_ylim(IMG_H, 0)  # y 轴反向（和你之前一致）
    ax.set_aspect("equal", adjustable="box")

    for tid in selected:
        c = tid2color[tid]
        seq = points[tid]
        xs = [p[0] for p in seq]
        ys = [p[1] for p in seq]
        ax.plot(xs, ys, linewidth=2.2, color=c, label=f"UAV {selected.index(tid)+1} (tid={tid})")
        ax.scatter([xs[0]], [ys[0]], s=24, color=c, marker="o",
                   edgecolors="black", linewidths=0.5, zorder=6)
        ax.scatter([xs[-1]], [ys[-1]], s=34, color=c, marker="s",
                   edgecolors="black", linewidths=0.5, zorder=7)

    for tid in selected:
        c = tid2color[tid]
        for fr, bb, state in bboxes.get(tid, []):
            x, y, w, h = bb
            ls = "-" if state == "det" else "--"
            alpha = BBOX_ALPHA if state == "det" else 0.35
            ax.add_patch(
                Rectangle(
                    (x, y), w, h,
                    fill=False,
                    edgecolor=c,
                    linewidth=BBOX_LINEWIDTH,
                    alpha=alpha,
                    linestyle=ls,
                    zorder=3,
                )
            )

    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


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
    print(f"[info] WINDOW_SIZE={WINDOW_SIZE}, GATING_CHI2={GATING_CHI2}, MAX_MISSED={MAX_MISSED}")

    detection_data = run_detection_windowed(frame_files)
    total_dets = sum(len(v) for v in detection_data.values())
    print(f"[info] total detections across all frames: {total_dets}")
    if total_dets == 0:
        raise RuntimeError(
            "所有帧检测结果均为0：请先让 SpatialDroneDetector 在每帧能输出bbox。\n"
            "优先调整 WINDOW_SIZE，或在 SpatialDroneDetector 内调整阈值/连通域面积过滤。"
        )

    results, frame_tracks, stats = run_tracking_imm_mht(detection_data)

    out_png = os.path.join(OUTPUT_DIR, "mot_plot_imm_mht.png")
    out_tracks = os.path.join(OUTPUT_DIR, "tracks_imm_mht.txt")
    out_stats = os.path.join(OUTPUT_DIR, "run_stats_imm_mht.json")

    plot_mot(out_png, frame_tracks, title=f"UAV MOT (IMM-MHT) | window={WINDOW_SIZE}")
    save_tracks_txt(results, out_tracks)
    dump_json(out_stats, stats)

    # Top4 版本（用于你只看 4 条主轨迹是否稳定）
    top4_ids = select_topk_track_ids(frame_tracks, k=NUM_DRONES_TO_SHOW)
    frame_tracks_top4 = filter_frame_tracks_by_ids(frame_tracks, top4_ids)
    out_png_top4 = os.path.join(OUTPUT_DIR, "mot_plot_imm_mht_top4.png")
    plot_mot(out_png_top4, frame_tracks_top4, title=f"UAV MOT (IMM-MHT) TOP4 | window={WINDOW_SIZE}")

    print("[done] outputs:")
    print(" ", out_png)
    print(" ", out_png_top4)
    print(" ", out_tracks)
    print(" ", out_stats)
    print("[info] top4 tids:", top4_ids)


if __name__ == "__main__":
    main()
