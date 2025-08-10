# main.py

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

from src.pixel_shifting_correction import RadarImageProcessor
from src.object_detection      import SpatialDroneDetector
from src.CA_UKF               import UKFPredictor   # 换成 UKF


def natural_sort_key(s):
    parts = re.split(r'(\d+)', s)
    return [int(p) if p.isdigit() else p for p in parts]


def main():
    data_folder = "data/600m2"
    out_folder  = "pics_ukf"
    os.makedirs(out_folder, exist_ok=True)

    # 1. 文件列表
    all_files = sorted(glob.glob(os.path.join(data_folder, "*.txt")),
                       key=natural_sort_key)
    full_scan   = all_files[0]
    frame_files = all_files[1:]

    # 2. 初始化
    proc      = RadarImageProcessor(shift_pixel=4)
    predictor = UKFPredictor(
        process_var=0.5,
        meas_var   =10.0
    )

    actual_centers    = []
    predicted_centers = []
    Vmax = 20.0  # px/frame

    # 3. 逐帧检测 + 更新 + 预测
    for idx, fp in enumerate(frame_files, start=1):
        print(f"[Frame {idx:03d}] {os.path.basename(fp)}")
        detector = SpatialDroneDetector(
            processor      = proc,
            full_scan_file = full_scan,
            update_files   = [fp],
            use_otsu       = False,
            thr_percentile = 15,
            median_size    = 3,
            tophat_radius  = 5,
            min_area       = 3,
            max_area       = 50
        )
        boxes = detector.detect()

        if boxes:
            # 单目标：面积最大的框
            areas = [(x1-x0)*(y1-y0) for x0,x1,y0,y1 in boxes]
            i_best = int(np.argmax(areas))
            box    = boxes[i_best]

            # UKF update
            center = predictor.update(idx, box)
            actual_centers.append(center)

            # UKF predict 下一帧
            p_raw = predictor.predict(future_frames=1)[0]
            delta = p_raw - center
            dist  = np.linalg.norm(delta)
            if dist > Vmax:
                delta = delta / dist * Vmax
            predicted_centers.append(center + delta)
        else:
            actual_centers.append(None)
            predicted_centers.append(None)

        # 可视化单帧
        img = detector.build_accumulated_image()
        plt.figure(figsize=(5,5))
        plt.imshow(img, cmap='jet', origin='lower')
        ax = plt.gca()

        # 检测框
        if boxes:
            x0,x1,y0,y1 = box
            ax.add_patch(plt.Rectangle(
                (x0, y0), x1-x0, y1-y0,
                edgecolor='yellow', facecolor='none', lw=2
            ))

        # 历史轨迹
        pts = [c for c in actual_centers if c is not None]
        if len(pts)>=2:
            arr = np.vstack(pts)
            ax.plot(arr[:,0], arr[:,1],
                    '-o', color='white', markersize=4, linewidth=1.2)

        # 最新预测
        if actual_centers[-1] is not None and predicted_centers[-1] is not None:
            lt = actual_centers[-1]
            nt = predicted_centers[-1]
            ax.plot([lt[0], nt[0]], [lt[1], nt[1]],
                    'r--', lw=1.5)
            ax.plot(nt[0], nt[1],
                    'rx', mew=2, markersize=8)

        ax.axis('off')
        ax.set_title(f"UKF Frame {idx:03d}")
        plt.savefig(os.path.join(out_folder, f"ukf_frame_{idx:03d}.png"),
                    bbox_inches='tight', pad_inches=0)
        plt.close()

    # 4. 计算 MAE / RMSE
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
    print(f"\nUKF Overall MAE = {mae:.3f}px, RMSE = {rmse:.3f}px")

    # 5. 汇总图：只画点
    bg = detector.build_accumulated_image()
    plt.figure(figsize=(6,6))
    plt.imshow(bg, cmap='jet', origin='lower')
    pts_all  = np.vstack([c for c in actual_centers    if c is not None])
    preds_at = np.vstack([p for p in predicted_centers[:-1] if p is not None])

    plt.plot(pts_all[:,0], pts_all[:,1],
             '-o', color='white', markersize=4, linewidth=1.2,
             label='True Track')
    plt.plot(preds_at[:,0], preds_at[:,1],
             'rx', markersize=6, label='Pred Points')

    plt.text(0.02, 0.02,
             f"MAE={mae:.2f}px\nRMSE={rmse:.2f}px",
             color='yellow', transform=plt.gca().transAxes,
             fontsize=10, bbox=dict(facecolor='black', alpha=0.5))
    plt.legend(loc='lower right', fontsize=8)
    plt.axis('off')
    plt.savefig(os.path.join(out_folder, "ukf_summary.png"),
                bbox_inches='tight', pad_inches=0)
    plt.close()

    # 6. 对比图：折线形式
    plt.figure(figsize=(6,6))
    plt.imshow(bg, cmap='jet', origin='lower')
    plt.plot(pts_all[:,0], pts_all[:,1],
             '-o', color='white', markersize=4, linewidth=1.2,
             label='True Track')
    plt.plot(preds_at[:,0], preds_at[:,1],
             '--o', color='red', markersize=4, linewidth=1.2,
             label='Pred Track')

    plt.text(0.02, 0.02,
             f"MAE={mae:.2f}px  RMSE={rmse:.2f}px",
             color='yellow', transform=plt.gca().transAxes,
             fontsize=10, bbox=dict(facecolor='black', alpha=0.5))
    plt.legend(loc='lower right', fontsize=8)
    plt.axis('off')
    plt.savefig(os.path.join(out_folder, "ukf_compare.png"),
                bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == "__main__":
    main()
