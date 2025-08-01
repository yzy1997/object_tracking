# main.py

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from src.pixel_shifting_correction import RadarImageProcessor
from src.object_detection      import SpatialDroneDetector
from src.trajectory_prediction import SingleDronePredictor

def main():
    data_folder = "data/600m"
    out_folder  = "pics"
    os.makedirs(out_folder, exist_ok=True)

    # 1) 准备数据文件列表
    all_files   = sorted(glob.glob(os.path.join(data_folder, "*.txt")))
    full_scan   = all_files[0]       # 背景全景
    frame_files = all_files[1:]      # 每帧增量

    # 2) 初始化处理器和预测器
    proc      = RadarImageProcessor(shift_pixel=4)
    predictor = SingleDronePredictor()

    # 用于统计真实与预测
    actual_centers    = []   # 第 i 帧检测到的中心
    predicted_centers = []   # 对第 i+1 帧的预测中心

    # 3) 逐帧检测 & 预测 & 单帧可视化
    for idx, fp in enumerate(frame_files, start=1):
        print(f"[Frame {idx:03d}] 读取 {os.path.basename(fp)}")

        # 每次只喂当前帧给 detector
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
        boxes = detector.detect()   # 当帧所有候选框

        if len(boxes) > 0:
            # 选面积最大的框作为无人机
            areas = [(x1-x0)*(y1-y0) for x0,x1,y0,y1 in boxes]
            best  = int(np.argmax(areas))
            box   = boxes[best]

            # 更新真实中心
            c = predictor.update(idx, box)
            actual_centers.append(c)

            # 预测下一帧
            p = predictor.predict(future_frames=1)[0]
            predicted_centers.append(p)
        else:
            # 漏检则用 None 占位
            box = None
            actual_centers.append(None)
            predicted_centers.append(None)

        # 构建累积后的雷达图
        img = detector.build_accumulated_image()

        # 单帧可视化
        plt.figure(figsize=(5,5))
        plt.imshow(img, cmap='jet', origin='lower')
        ax = plt.gca()

        # (1) 当前检测框，黄色
        if box is not None:
            x0,x1,y0,y1 = box
            ax.add_patch(plt.Rectangle(
                (x0,y0), x1-x0, y1-y0,
                edgecolor='yellow', facecolor='none', lw=2,
                label='Detection'
            ))

        # (2) 历史轨迹，白折线+圆点
        if len(predictor.centers) >= 2:
            hist = np.vstack(predictor.centers)
            ax.plot(hist[:,0], hist[:,1],
                    '-o', color='white', markersize=4,
                    linewidth=1.2, label='History')

        # (3) 预测虚线+红叉
        if predicted_centers[-1] is not None:
            last = predictor.centers[-1]
            nxt  = predicted_centers[-1]
            ax.plot([last[0], nxt[0]], [last[1], nxt[1]],
                    'r--', linewidth=1.5, label='Prediction')
            ax.plot(nxt[0], nxt[1],
                    'rx', markersize=8, mew=2)

        # 把单帧图例放右下
        ax.legend(loc='lower right', fontsize=8)
        ax.set_title(f"Frame {idx:03d}")
        ax.axis('off')

        out_png = os.path.join(out_folder, f"predict_frame_{idx:03d}.png")
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0)
        plt.close()
        print("  -> 已保存", out_png)

    # 4) 计算预测误差（第 i 帧预测 vs 第 i+1 帧真实）
    dists = []
    for i in range(len(predicted_centers)-1):
        p = predicted_centers[i]
        a = actual_centers[i+1]
        if p is None or a is None:
            continue
        dists.append(np.linalg.norm(p - a))
    dists = np.array(dists, dtype=np.float32)

    if len(dists)>0:
        mae  = dists.mean()
        rmse = np.sqrt((dists**2).mean())
    else:
        mae = rmse = np.nan
    print(f"\n整体预测误差：MAE = {mae:.3f}px, RMSE = {rmse:.3f}px")

    # 5) 绘制一张“汇总图”：全部真实轨迹 + 全部预测点 + 误差连线 + 文字 + 图例
    plt.figure(figsize=(6,6))
    bg = detector.build_accumulated_image()  # 用最后一帧的累积图当背景
    plt.imshow(bg, cmap='jet', origin='lower')
    ax = plt.gca()

    # 收集有效的真实点和预测点
    real_pts = np.array([c for c in actual_centers if c is not None])
    pred_pts = np.array([p for p in predicted_centers if p is not None])

    # (1) 真实轨迹：白折线+圆点
    if len(real_pts) > 0:
        ax.plot(real_pts[:,0], real_pts[:,1],
                '-o', color='white', markersize=5,
                linewidth=1.5, label='Real Traj')

    # (2) 预测点：红叉
    if len(pred_pts) > 0:
        ax.plot(pred_pts[:,0], pred_pts[:,1],
                'rx', markersize=8, mew=2, label='Predicted')

    # (3) 每次真实→预测连虚线
    for i in range(len(predicted_centers)-1):
        p = predicted_centers[i]
        a = actual_centers[i]
        if p is None or a is None:
            continue
        ax.plot([a[0], p[0]], [a[1], p[1]],
                color='gray', linestyle='--', linewidth=1)

    # (4) 误差文字框，右下对齐
    txt = f"MAE = {mae:.2f}px\nRMSE = {rmse:.2f}px"
    ax.text(
        0.98, 0.02, txt,
        transform=ax.transAxes,
        color='white', fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(facecolor='black', alpha=0.5, pad=4)
    )

    # (5) 图例放右下
    ax.legend(loc='lower right', fontsize=8)
    ax.set_title("Overall Trajectory & Predictions")
    ax.axis('off')

    out_all = os.path.join(out_folder, "summary.png")
    plt.savefig(out_all, bbox_inches='tight', pad_inches=0)
    plt.close()
    print("汇总图已保存到", out_all)


if __name__ == "__main__":
    main()
