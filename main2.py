# main2.py
# -*- coding: utf-8 -*-

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

from src.denoise import RadarDenoiser
from src.pixel_shifting_correction import RadarImageProcessor
from src.object_detection import SpatialDroneDetector
from src.CA_EKF import KalmanPredictor


def natural_sort_key(s):
    parts = re.split(r'(\d+)', s)
    return [int(p) if p.isdigit() else p for p in parts]


# ---- 辅助函数 ----
def box_center(b):
    x0, x1, y0, y1 = b
    return np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=np.float32)

def filter_boxes_roi(boxes, roi):
    if not boxes:
        return []
    xmin, xmax, ymin, ymax = roi
    out = []
    for b in boxes:
        c = box_center(b)
        if xmin <= c[0] <= xmax and ymin <= c[1] <= ymax:
            out.append(b)
    return out

def brightest_peak_box_in_roi(img, roi, half=2):
    """在 ROI 内取 img(TopHat 或 diff) 的最亮像素，生成小框。"""
    if not isinstance(img, np.ndarray):
        return None
    H, W = img.shape
    xmin, xmax, ymin, ymax = roi
    xmin = max(0, int(xmin)); xmax = min(W - 1, int(xmax))
    ymin = max(0, int(ymin)); ymax = min(H - 1, int(ymax))
    if xmin >= xmax or ymin >= ymax:
        return None
    sub = img[ymin:ymax + 1, xmin:xmax + 1]
    if sub.size == 0 or float(sub.max()) <= 0:
        return None
    r_rel, c_rel = np.unravel_index(np.argmax(sub), sub.shape)
    r = ymin + r_rel; c = xmin + c_rel
    x0 = max(c - half, 0); x1 = min(c + half, W - 1) + 1
    y0 = max(r - half, 0); y1 = min(r + half, H - 1) + 1
    return (x0, x1, y0, y1)


def main():
    data_folder = "data/1600"
    out_folder  = "pics/1600"
    os.makedirs(out_folder, exist_ok=True)

    # 1) 文件列表
    all_files = sorted(glob.glob(os.path.join(data_folder, "*.txt")),
                       key=natural_sort_key)
    if len(all_files) < 2:
        print("[ERROR] 至少需要背景帧 + 1 帧更新数据")
        return
    full_scan   = all_files[0]
    frame_files = all_files[1:]

    # 2) 去噪器（远距小目标：不要强制邻居）
    denoiser = RadarDenoiser(
        kernel_size=3, min_neighbors=1,
        filter_size=3, thr_percentile=95, rel_floor=0.12,
        morph_open=1, morph_close=1
    )

    # 3) 处理器 + EKF
    proc = RadarImageProcessor(shift_pixel=4, denoiser=denoiser)
    predictor = KalmanPredictor(process_var=0.5, meas_var=25.0)

    actual_centers, predicted_centers = [], []
    Vmax = 15.0

    # 4) 检测器（单实例 + 静态背景）
    detector = SpatialDroneDetector(
        processor      = proc,
        full_scan_file = full_scan,
        update_files   = [],
        use_otsu       = False,
        thr_percentile = 45,
        thr_rel_floor  = 0.06,
        median_size    = 1,
        tophat_radius  = 1,
        cluster_dist   = 3,
        topk           = 8,
        min_area       = 1,  max_area = 20,
        max_width      = 10, max_height = 8,
        min_y          = 90,
        bg_mode        = 'static',
        temporal_window= 1,
        debug          = True
    )

    # 5) ROI & 自适应门控（仅用于内部选择，不再画出来）
    ROI    = (5, 120, 80, 131)   # 覆盖上方黄点带（按需要微调）
    BASE_GATE_R = 14.0
    miss_count = 0

    # 6) 逐帧
    for idx, fp in enumerate(frame_files, start=1):
        print(f"[Frame {idx:03d}] {os.path.basename(fp)}")
        detector.update_files = [fp]

        # 预测点 + 自适应门控
        if getattr(predictor, "inited", False):
            gate_center = predictor.predict(future_frames=1)[0]
        else:
            gate_center = None
        gate_r = BASE_GATE_R * (1.0 + 0.6 * miss_count)

        # 检测
        boxes = detector.detect()

        # 从 ROI 内挑候选
        roi_boxes = filter_boxes_roi(boxes, ROI)

        sel_box = None
        if roi_boxes:
            if gate_center is None:
                roi_boxes.sort(key=lambda b: -box_center(b)[1])
                sel_box = roi_boxes[0]
            else:
                roi_boxes.sort(key=lambda b: np.linalg.norm(box_center(b) - gate_center))
                if np.linalg.norm(box_center(roi_boxes[0]) - gate_center) <= gate_r:
                    sel_box = roi_boxes[0]
        else:
            # ROI 双回退：TopHat → Diff
            topht = getattr(detector, "tophat_img", None)
            cand  = brightest_peak_box_in_roi(topht, ROI, half=2)
            if cand is None:
                diff = getattr(detector, "diff_img", None)
                cand = brightest_peak_box_in_roi(diff, ROI, half=2)
            if cand is not None:
                if (gate_center is None) or (np.linalg.norm(box_center(cand) - gate_center) <= gate_r):
                    sel_box = cand

        # —— 可视化：不画 ROI/门控，只画“红色小方框” —— #
        img = detector.build_accumulated_image()
        plt.figure(figsize=(5,5))
        plt.imshow(img, cmap='jet', origin='lower')
        ax = plt.gca()

        # 画所有检测到的目标框（红色小方框）
        for b in boxes:
            x0, x1, y0, y1 = b
            ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0,
                                       edgecolor='red', facecolor='none',
                                       lw=1.2, alpha=0.85))
        # 高亮本帧用于更新的目标框（更粗的红框）
        if sel_box is not None:
            x0, x1, y0, y1 = sel_box
            ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0,
                                       edgecolor='red', facecolor='none',
                                       lw=2.5))

        # 历史“观测”轨迹
        pts = [c for c in actual_centers if c is not None]
        if len(pts) >= 2:
            arr = np.vstack(pts)
            ax.plot(arr[:,0], arr[:,1], '-o', color='white',
                    markersize=3.5, linewidth=1.2)

        # 最新预测连线
        if getattr(predictor, "inited", False) and pts:
            last = pts[-1]
            p_raw = predictor.predict(future_frames=1)[0]
            ax.plot([last[0], p_raw[0]], [last[1], p_raw[1]], 'r--', lw=1.2)
            ax.plot(p_raw[0], p_raw[1], 'rx', mew=2, markersize=7)

        ax.set_title(f"Frame {idx:03d}")
        ax.axis('off')
        plt.savefig(os.path.join(out_folder, f"frame_{idx:03d}.png"),
                    bbox_inches='tight', pad_inches=0)
        plt.close()

        # —— EKF：更新/预测 + 自适应门控记账 —— #
        if sel_box is not None:
            center = predictor.update(idx, sel_box)
            actual_centers.append(center)
            miss_count = 0
        else:
            if getattr(predictor, "inited", False):
                predictor.predict(future_frames=1)
            actual_centers.append(None)
            miss_count = min(miss_count + 1, 3)

        if getattr(predictor, "inited", False):
            next_pred = predictor.predict(future_frames=1)[0]
            last_obs  = next((c for c in reversed(actual_centers) if c is not None), None)
            if last_obs is not None:
                delta = next_pred - last_obs
                dist  = np.linalg.norm(delta)
                if dist > Vmax:
                    delta = delta / max(dist, 1e-6) * Vmax
                predicted_centers.append(last_obs + delta)
            else:
                predicted_centers.append(next_pred)
        else:
            predicted_centers.append(None)

    # 7) 评估（跳过空帧）
    errs = []
    for i in range(len(predicted_centers)-1):
        p = predicted_centers[i]
        a = actual_centers[i+1]
        if p is None or a is None:
            continue
        errs.append(np.linalg.norm(p - a))
    errs = np.array(errs, dtype=np.float32)
    mae  = errs.mean() if errs.size>0 else np.nan
    rmse = np.sqrt((errs**2).mean()) if errs.size>0 else np.nan
    print(f"\nOverall MAE = {mae:.2f}px, RMSE = {rmse:.2f}px")

    # 8) 汇总图（白点+白实线=真实；红框+红虚线=预测）
    from matplotlib.patches import Rectangle

    bg = detector.build_accumulated_image()
    plt.figure(figsize=(6,6))
    plt.imshow(bg, cmap='jet', origin='lower')
    ax = plt.gca()

    # 收集轨迹点（跳过 None）
    obs_list  = [c for c in actual_centers if c is not None]
    pred_list = [p for p in predicted_centers[:-1] if p is not None]

    # ===== 真实观测：白色点 + 白色实线 =====
    if len(obs_list) > 0:
        obs = np.vstack(obs_list)
        ax.plot(
            obs[:,0], obs[:,1],
            '-o', color='white', linewidth=1.4, markersize=4,
            markeredgecolor='black', markeredgewidth=0.3,
            label='Obs Track'
        )

    # ===== 预测：红色小方框 + 红色虚线 =====
    PRED_BOX_HALF = 2   # 方框半边长（像素），想更小可改为 1
    pred_proxy = None
    if len(pred_list) > 0:
        preds = np.vstack(pred_list)

        # 连线
        ax.plot(preds[:,0], preds[:,1], '--', color='red', linewidth=1.2, label='Pred Track')

        # 小方框（以预测中心为正方形中心）
        H, W = bg.shape
        for px, py in preds:
            px, py = float(px), float(py)
            x0 = max(px - PRED_BOX_HALF, 0)
            y0 = max(py - PRED_BOX_HALF, 0)
            w  = min(PRED_BOX_HALF*2, W - x0)
            h  = min(PRED_BOX_HALF*2, H - y0)
            ax.add_patch(Rectangle((x0, y0), w, h, edgecolor='red', facecolor='none', lw=1.6))

    # 评估文本
    plt.text(0.02, 0.02, f"MAE={mae:.2f}px  RMSE={rmse:.2f}px",
             color='yellow', transform=plt.gca().transAxes,
             fontsize=10, bbox=dict(facecolor='black', alpha=0.5))

    ax.axis('off')
    plt.savefig(os.path.join(out_folder, "CA_EKF_compare.png"),
                bbox_inches='tight', pad_inches=0)
    plt.close()



if __name__ == "__main__":
    main()
