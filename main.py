# main.py

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

from src.pixel_shifting_correction import RadarImageProcessor
from src.object_detection      import SpatialDroneDetector
# from src.UKF import KalmanPredictor
# from src.EKF import KalmanPredictor
from src.CA_EKF import KalmanPredictor
# from src.CA_UKF import KalmanPredictorUKF as KalmanPredictor
# from src.IMM_UKF import IMMUKFTracker as KalmanPredictor
# from src.data_association import associate_detections_to_tracks


def natural_sort_key(s):
    parts = re.split(r'(\d+)', s)
    # 数字部分转 int，其它部分保持 str
    return [int(p) if p.isdigit() else p for p in parts]

def main():
    data_folder = "data/600m2"
    out_folder  = "pics"
    os.makedirs(out_folder, exist_ok=True)

    # 读取目录下所有 txt，第一帧当背景，全都放到 frame_files
    all_files   = sorted(glob.glob(os.path.join(data_folder, "*.txt")),key=natural_sort_key)
    full_scan   = all_files[0]
    frame_files = all_files[1:]

    # 初始化预处理器 + 滤波预测器
    proc      = RadarImageProcessor(shift_pixel=4)
    predictor = KalmanPredictor(
        process_var=2.0,    # 过程噪声方差，可调
        meas_var=15.0       # 测量噪声方差，可调
    )

    actual_centers    = []   # 存储每帧的真实中心
    predicted_centers = []   # 存储每帧的预测中心

    for idx, fp in enumerate(frame_files, start=1):
        print(f"[Frame {idx:03d}] {os.path.basename(fp)}")

        # 新建检测器，单帧更新
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

        box = None
        # 如果检测到目标，就取最大面积那一个
        if boxes:
            areas = [(x1-x0)*(y1-y0) for x0,x1,y0,y1 in boxes]
            best  = int(np.argmax(areas))
            box   = boxes[best]

            # 卡尔曼滤波更新，返回滤波后中心
            center = predictor.update(idx, box)
            actual_centers.append(center)

            # 外推下一帧位置
            p = predictor.predict(future_frames=1)[0]
            predicted_centers.append(p)
        else:
            # 没检测到就填 None
            actual_centers.append(None)
            predicted_centers.append(None)

        # 可视化当前帧
        img = detector.build_accumulated_image()
        plt.figure(figsize=(5,5))
        plt.imshow(img, cmap='jet', origin='lower')
        ax = plt.gca()

        # 画检测框
        if box is not None:
            x0,x1,y0,y1 = box
            ax.add_patch(plt.Rectangle(
                (x0,y0), x1-x0, y1-y0,
                edgecolor='yellow', facecolor='none', lw=2, label='Detection'
            ))

        # 画历史轨迹
        pts = [c for c in actual_centers if c is not None]
        if len(pts) >= 2:
            pts = np.vstack(pts)
            ax.plot(pts[:,0], pts[:,1],
                    '-o', color='white',
                    markersize=4, linewidth=1.2,
                    label='History')

        # 画最新的预测
        if actual_centers[-1] is not None and predicted_centers[-1] is not None:
            last_true = actual_centers[-1]
            nxt       = predicted_centers[-1]
            ax.plot([last_true[0], nxt[0]],
                    [last_true[1], nxt[1]],
                    'r--', lw=1.5, label='Prediction')
            ax.plot(nxt[0], nxt[1],
                    'rx', mew=2, markersize=8)

        ax.legend(loc='lower right', fontsize=8)
        ax.set_title(f"Frame {idx:03d}")
        ax.axis('off')

        # 保存成 augment_pre_frame_xxx.png
        out_png = os.path.join(out_folder,
                               f"augment_pre_frame_{idx:03d}.png")
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0)
        plt.close()

    # 计算整体 MAE / RMSE
    dists = []
    for i in range(len(predicted_centers)-1):
        p = predicted_centers[i]
        a = actual_centers[i+1]
        if p is None or a is None:
            continue
        dists.append(np.linalg.norm(p - a))
    dists = np.array(dists, dtype=np.float32)
    mae  = dists.mean() if len(dists)>0 else np.nan
    rmse = np.sqrt((dists**2).mean()) if len(dists)>0 else np.nan
    print(f"\nOverall MAE = {mae:.3f}px, RMSE = {rmse:.3f}px")

    # （可选）汇总图：所有历史轨迹 + 所有预测点 + 误差卡片
    plt.figure(figsize=(6,6))
    # 用最后一帧累积图当背景，也可重新 build_accumulated_image()
    bg = detector.build_accumulated_image()
    plt.imshow(bg, cmap='jet', origin='lower')

    # 整条历史轨迹
    pts = np.vstack([c for c in actual_centers if c is not None])
    plt.plot(pts[:,0], pts[:,1],
             '-o', color='white', markersize=4, linewidth=1.2,
             label='History')

    # 所有预测点
    preds = np.vstack([p for p in predicted_centers[:-1] if p is not None])
    plt.plot(preds[:,0], preds[:,1],
             'rx', markersize=6, label='Prediction')

    # 右下误差卡片
    plt.text(0.02, 0.02,
             f"MAE = {mae:.2f}px\nRMSE = {rmse:.2f}px",
             color='yellow', transform=plt.gca().transAxes,
             fontsize=10, bbox=dict(facecolor='black', alpha=0.5))

    plt.legend(loc='lower right', fontsize=8)
    plt.axis('off')
    plt.savefig(os.path.join(out_folder, "augment_pre_summary.png"),
                bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    main()
