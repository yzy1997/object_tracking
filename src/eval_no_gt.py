# -*- coding: utf-8 -*-
"""
No-GT 轨迹评价（不依赖真值，不改动关联逻辑）
读取 tracks.txt: frame tid x y w h state(det|pred)
输出: metrics_no_gt.json, report_no_gt.txt

用法（直接运行）：
  python eval_no_gt.py
或在 main_mot.py 结束后 import 调用 evaluate(...)
"""

from __future__ import annotations

import os
import json
import math
import statistics
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# -----------------------------
# 固定路径（按你工程改）
# -----------------------------
RESULTS_DIR = r"D:\codes\object_tracking\results\mot_compare_out"
TRACKS_TXT = os.path.join(RESULTS_DIR, "tracks.txt")
OUT_JSON = os.path.join(RESULTS_DIR, "metrics_no_gt.json")
OUT_REPORT = os.path.join(RESULTS_DIR, "report_no_gt.txt")

# 图像边界（可选；你是 132x132）
IMG_W, IMG_H = 132, 132

# 近距离判定阈值：同一帧两条轨迹中心过近 -> 可能重复/误检（像素）
DUP_NEAR_DIST = 2.0

# gate 距离阈值：用于 EDR/UTR 的简单匹配（像素）
GATE_DIST = 4.0


@dataclass
class TrackPoint:
    frame: int
    tid: int
    x: float
    y: float
    w: float
    h: float
    state: str  # "det" or "pred"


def read_tracks_txt(path: str) -> List[TrackPoint]:
    pts: List[TrackPoint] = []
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            frame = int(parts[0])
            tid = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            state = parts[6].strip()
            pts.append(TrackPoint(frame, tid, x, y, w, h, state))
    return pts


def group_by_frame(points: List[TrackPoint]) -> Dict[int, List[TrackPoint]]:
    by: Dict[int, List[TrackPoint]] = {}
    for p in points:
        by.setdefault(p.frame, []).append(p)
    return by


def group_by_tid(points: List[TrackPoint]) -> Dict[int, List[TrackPoint]]:
    by: Dict[int, List[TrackPoint]] = {}
    for p in points:
        by.setdefault(p.tid, []).append(p)
    for tid, lst in by.items():
        lst.sort(key=lambda z: z.frame)
    return by


def euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def greedy_match_cost(
    A: List[Tuple[float, float]],
    B: List[Tuple[float, float]],
    gate: float
) -> Tuple[int, int, int, float]:
    """
    简单贪心匹配：用于无GT的 EDR/UTR proxy
    返回: (matched, unmatched_A, unmatched_B, mean_dist_of_matched)
    """
    if not A and not B:
        return 0, 0, 0, 0.0
    if not A:
        return 0, 0, len(B), 0.0
    if not B:
        return 0, len(A), 0, 0.0

    used_b = set()
    pairs: List[Tuple[float, int, int]] = []
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            d = euclid(a, b)
            if d <= gate:
                pairs.append((d, i, j))
    pairs.sort(key=lambda t: t[0])

    used_a = set()
    matched_dists = []
    for d, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched_dists.append(d)

    matched = len(matched_dists)
    unmatched_a = len(A) - matched
    unmatched_b = len(B) - matched
    mean_d = float(sum(matched_dists) / matched) if matched > 0 else 0.0
    return matched, unmatched_a, unmatched_b, mean_d


def compute_metrics(points: List[TrackPoint]) -> Dict:
    by_frame = group_by_frame(points)
    by_tid = group_by_tid(points)

    frames = sorted(by_frame.keys())
    if not frames:
        return {"error": "no frames"}

    # -----------------------------
    # 每帧统计
    # -----------------------------
    tracks_per_frame = []
    dets_per_frame = []
    preds_per_frame = []
    out_of_bounds = 0
    total_points = 0

    dup_pairs = 0
    possible_pairs = 0

    # EDR/UTR（以 state=det 的 track 点当做“观测检测”proxy）
    total_obs = 0
    total_obs_matched = 0
    total_trk = 0
    total_trk_matched = 0
    matched_dist_sum = 0.0
    matched_dist_cnt = 0

    for fr in frames:
        pts_f = by_frame[fr]
        total_points += len(pts_f)

        det_pts = [p for p in pts_f if p.state == "det"]
        pred_pts = [p for p in pts_f if p.state != "det"]

        tracks_per_frame.append(len(pts_f))
        dets_per_frame.append(len(det_pts))
        preds_per_frame.append(len(pred_pts))

        # 越界统计
        for p in pts_f:
            if p.x < 0 or p.x >= IMG_W or p.y < 0 or p.y >= IMG_H:
                out_of_bounds += 1

        # 同帧重复/过近轨迹统计（不需要GT）
        centers = [(p.x, p.y) for p in pts_f]
        n = len(centers)
        possible_pairs += n * (n - 1) // 2
        for i in range(n):
            for j in range(i + 1, n):
                if euclid(centers[i], centers[j]) <= DUP_NEAR_DIST:
                    dup_pairs += 1

        # EDR/UTR proxy：轨迹 vs “观测(det点)”
        trk_centers = [(p.x, p.y) for p in pts_f]
        obs_centers = [(p.x, p.y) for p in det_pts]

        matched, un_trk, un_obs, mean_d = greedy_match_cost(trk_centers, obs_centers, gate=GATE_DIST)
        # matched 在这里是“轨迹-观测”匹配数
        total_trk += len(trk_centers)
        total_obs += len(obs_centers)
        total_trk_matched += matched
        total_obs_matched += matched
        if matched > 0:
            matched_dist_sum += mean_d * matched
            matched_dist_cnt += matched

    # -----------------------------
    # 每轨迹统计：寿命、断裂、平滑
    # -----------------------------
    track_lengths = []
    det_ratios_per_track = []
    frag_per_track = []  # det连续段数
    vel_stds = []

    for tid, seq in by_tid.items():
        if not seq:
            continue
        track_lengths.append(len(seq))

        det_flags = [1 if p.state == "det" else 0 for p in seq]
        det_ratio = sum(det_flags) / len(det_flags)
        det_ratios_per_track.append(det_ratio)

        # fragmentation: det 的连续段数
        segments = 0
        in_seg = False
        for f in det_flags:
            if f == 1 and not in_seg:
                segments += 1
                in_seg = True
            elif f == 0:
                in_seg = False
        frag_per_track.append(segments)

        # 速度抖动
        if len(seq) >= 3:
            v = []
            for i in range(1, len(seq)):
                v.append(euclid((seq[i].x, seq[i].y), (seq[i-1].x, seq[i-1].y)))
            if len(v) >= 2:
                vel_stds.append(statistics.pstdev(v))
        # len<3 的轨迹不算抖动

    def safe_mean(x: List[float]) -> float:
        return float(sum(x) / len(x)) if x else 0.0

    def safe_median(x: List[float]) -> float:
        return float(statistics.median(x)) if x else 0.0

    det_ratio_all = safe_mean([d / t if t > 0 else 0 for d, t in zip(dets_per_frame, tracks_per_frame)])
    avg_tracks = safe_mean(tracks_per_frame)
    std_tracks = float(statistics.pstdev(tracks_per_frame)) if len(tracks_per_frame) >= 2 else 0.0

    metrics = {
        "Frames": {
            "count": len(frames),
            "first": frames[0],
            "last": frames[-1],
        },
        "Counts": {
            "AvgTracksPerFrame": avg_tracks,
            "TrackCountStd": std_tracks,
            "AvgDetPerFrame": safe_mean(dets_per_frame),
            "AvgPredPerFrame": safe_mean(preds_per_frame),
            "DetRatio_over_frames": det_ratio_all,  # det/total per frame 平均
        },
        "Tracks": {
            "NumUniqueTracks": len(by_tid),
            "MeanTrackLen": safe_mean(track_lengths),
            "MedianTrackLen": safe_median(track_lengths),
            "MeanDetRatio_per_track": safe_mean(det_ratios_per_track),
            "MedianDetRatio_per_track": safe_median(det_ratios_per_track),
            "MeanFragSegments_per_track": safe_mean(frag_per_track),
            "MedianFragSegments_per_track": safe_median(frag_per_track),
        },
        "Geometry": {
            "OutOfBoundsRatio": (out_of_bounds / total_points) if total_points else 0.0,
            "DuplicationScore_near_pairs_ratio": (dup_pairs / possible_pairs) if possible_pairs else 0.0,
            "DupNearDist": DUP_NEAR_DIST,
        },
        "Consistency_NoGT": {
            # 这些是“轨迹解释观测”的 proxy（观测=det 点）
            "GateDist": GATE_DIST,
            "EDR_matched_obs_ratio": (total_obs_matched / total_obs) if total_obs else 0.0,
            "UTR_unmatched_track_ratio": (1.0 - (total_trk_matched / total_trk)) if total_trk else 0.0,
            "MeanMatchedDist": (matched_dist_sum / matched_dist_cnt) if matched_dist_cnt else 0.0,
        },
        "Smoothness": {
            "MeanVelStd": safe_mean(vel_stds),
            "MedianVelStd": safe_median(vel_stds),
        },
        "Notes": [
            "No-GT metrics: these do NOT represent absolute tracking accuracy (MOTA/IDF1).",
            "EDR/UTR here use track points with state=det as observation proxy, so they can be optimistic.",
        ]
    }
    return metrics


def format_report(metrics: Dict) -> str:
    if "error" in metrics:
        return f"[error] {metrics['error']}\n"

    F = metrics["Frames"]
    C = metrics["Counts"]
    T = metrics["Tracks"]
    G = metrics["Geometry"]
    K = metrics["Consistency_NoGT"]
    S = metrics["Smoothness"]

    lines = []
    lines.append("==== No-GT Tracking Report ====")
    lines.append(f"Frames: {F['count']}  (range {F['first']}..{F['last']})")
    lines.append("")
    lines.append("[Counts]")
    lines.append(f"  AvgTracksPerFrame: {C['AvgTracksPerFrame']:.3f}   (expect ~4 if only 4 UAVs)")
    lines.append(f"  TrackCountStd:     {C['TrackCountStd']:.3f}   (lower is more stable)")
    lines.append(f"  AvgDetPerFrame:    {C['AvgDetPerFrame']:.3f}")
    lines.append(f"  AvgPredPerFrame:   {C['AvgPredPerFrame']:.3f}")
    lines.append(f"  DetRatio(frames):  {C['DetRatio_over_frames']:.3f}  (higher => fewer pure predictions)")
    lines.append("")
    lines.append("[Tracks]")
    lines.append(f"  NumUniqueTracks:         {T['NumUniqueTracks']}")
    lines.append(f"  MeanTrackLen:            {T['MeanTrackLen']:.3f}")
    lines.append(f"  MedianTrackLen:          {T['MedianTrackLen']:.3f}")
    lines.append(f"  MeanDetRatio_per_track:  {T['MeanDetRatio_per_track']:.3f}")
    lines.append(f"  MeanFragSegments/track:  {T['MeanFragSegments_per_track']:.3f} (1 is best; >1 indicates breaks)")
    lines.append("")
    lines.append("[Consistency (No-GT proxy)]")
    lines.append(f"  GateDist:           {K['GateDist']:.3f}")
    lines.append(f"  EDR (matched obs):  {K['EDR_matched_obs_ratio']:.3f}")
    lines.append(f"  UTR (unmatched trk):{K['UTR_unmatched_track_ratio']:.3f}")
    lines.append(f"  MeanMatchedDist:    {K['MeanMatchedDist']:.3f}")
    lines.append("")
    lines.append("[Geometry]")
    lines.append(f"  OutOfBoundsRatio:   {G['OutOfBoundsRatio']:.6f}")
    lines.append(f"  DuplicationScore:   {G['DuplicationScore_near_pairs_ratio']:.6f} (near-dist={G['DupNearDist']})")
    lines.append("")
    lines.append("[Smoothness]")
    lines.append(f"  MeanVelStd:         {S['MeanVelStd']:.3f} (lower is smoother)")
    lines.append("")
    lines.append("Notes:")
    for n in metrics.get("Notes", []):
        lines.append(f"  - {n}")
    lines.append("")
    return "\n".join(lines)


def evaluate(tracks_txt: str = TRACKS_TXT, out_json: str = OUT_JSON, out_report: str = OUT_REPORT) -> Dict:
    points = read_tracks_txt(tracks_txt)
    metrics = compute_metrics(points)

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(out_report, "w", encoding="utf-8") as f:
        f.write(format_report(metrics))

    return metrics


if __name__ == "__main__":
    m = evaluate()
    print("[done] wrote:")
    print(" ", OUT_JSON)
    print(" ", OUT_REPORT)