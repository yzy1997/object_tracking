# -*- coding: utf-8 -*-
"""
模拟无人机轨迹生成器
基于背景图片生成10条模拟无人机轨迹，每条25-30个点
用于测试GNN, MHT, IMM-MHT三种关联算法

特点：从左到右飞行，上下起伏，轨迹可交叉，模拟单光子雷达效果（黄色1-2像素点）
"""
import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# =============================
# 路径配置
# =============================
OUTPUT_DIR = r"D:\codes\object_tracking\results\simulated_tracks"
GT_CSV_PATH = os.path.join(OUTPUT_DIR, "gt_tracks.csv")
DETECTION_JSON_PATH = os.path.join(OUTPUT_DIR, "detections.json")

# 图像参数 - 雷达图像132x132
IMG_W, IMG_H = 132, 132

# 轨迹参数
NUM_TRAJECTORIES = 10
POINTS_PER_TRAJECTORY_MIN = 25
POINTS_PER_TRAJECTORY_MAX = 30

# =============================
# 轨迹生成函数 - 从左到右、上下起伏
# =============================
def generate_left_to_right_trajectory(
    start_x: float,
    start_y: float,
    num_points: int,
    wave_amplitude: float = 15.0,
    wave_freq: float = 0.3,
    base_speed: float = 4.0
) -> List[Tuple[float, float]]:
    """
    生成从左到右、上下起伏的无人机轨迹
    类似单光子雷达的稀疏点效果
    """
    points = []
    x = start_x
    y = start_y

    # 随机相位偏移，使每条轨迹起伏不同步
    phase_offset = np.random.uniform(0, 2 * np.pi)

    for i in range(num_points):
        points.append((x, y))

        # x方向匀速增加（从左到右）
        x += base_speed + np.random.uniform(-0.5, 0.5)

        # y方向正弦波动（上下起伏）
        y = start_y + wave_amplitude * np.sin(wave_freq * i + phase_offset)
        y += np.random.uniform(-2.0, 2.0)  # 添加小幅度随机抖动

        # 边界约束
        x = np.clip(x, 5, IMG_W - 5)
        y = np.clip(y, 5, IMG_H - 5)

    return points


def generate_all_trajectories() -> Dict[int, Dict]:
    """生成所有无人机轨迹 - 从左到右飞行，有3-4条轨迹明显交叉"""
    np.random.seed(42)

    trajectories = {}

    # 定义10条轨迹 - 分散但在中间区域有几条明显交叉
    start_configs = [
        # (start_x, start_y, wave_amplitude, wave_freq, base_speed)
        (10, 10, 4, 0.15, 4.0),    # 底部1
        (10, 25, 5, 0.18, 4.1),    # 底部2 - 与底部1接近
        (10, 40, 3, 0.12, 4.2),   # 中下
        (10, 55, 8, 0.22, 4.0),   # 中间1 - 交叉区域开始
        (10, 70, 10, 0.25, 3.9),  # 中间2 - 与中间1,3,4交叉
        (10, 85, 7, 0.20, 4.1),   # 中间3 - 与中间2,4交叉
        (10, 100, 6, 0.17, 4.2),  # 中间4 - 与中间2,3交叉
        (10, 115, 5, 0.15, 4.0),  # 中上
        (10, 125, 4, 0.13, 4.1),  # 顶部
        (10, 60, 12, 0.28, 3.8),  # 中间5 - 与中间1-4都交叉
    ]

    for i in range(NUM_TRAJECTORIES):
        start_x, start_y, wave_amp, wave_freq, speed = start_configs[i]
        num_points = np.random.randint(POINTS_PER_TRAJECTORY_MIN, POINTS_PER_TRAJECTORY_MAX + 1)

        points = generate_left_to_right_trajectory(
            start_x=start_x,
            start_y=start_y,
            num_points=num_points,
            wave_amplitude=wave_amp,
            wave_freq=wave_freq,
            base_speed=speed
        )

        trajectories[i] = {
            "track_id": i,
            "points": points,
            "num_points": len(points)
        }

    return trajectories


def save_gt_tracks(trajectories: Dict[int, Dict], output_path: str):
    """保存为GT格式 (frame, gt_id, x, y, w, h)"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "gt_id", "x", "y", "w", "h"])

        for track_id, data in trajectories.items():
            points = data["points"]
            for frame_idx, (x, y) in enumerate(points):
                # 单光子雷达效果：1-2像素的稀疏点
                w, h = 1.5, 1.5
                writer.writerow([frame_idx, track_id, f"{x:.2f}", f"{y:.2f}", f"{w:.2f}", f"{h:.2f}"])

    print(f"[GT] Saved to {output_path}")


def save_detections(trajectories: Dict[int, Dict], output_path: str, noise_std: float = 1.0):
    """
    保存为检测格式 (frame -> list of detections)
    添加测量噪声模拟真实检测（单光子雷达噪声）
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    max_frame = max(len(data["points"]) for data in trajectories.values())

    detections = {}
    np.random.seed(123)

    for frame_idx in range(max_frame):
        frame_dets = []
        for track_id, data in trajectories.items():
            points = data["points"]
            if frame_idx < len(points):
                x, y = points[frame_idx]
                # 添加测量噪声（单光子雷达噪声）
                x_noisy = x + np.random.normal(0, noise_std)
                y_noisy = y + np.random.normal(0, noise_std)
                # 1-2像素大小的检测框
                w = 1.5 + np.random.normal(0, 0.3)
                h = 1.5 + np.random.normal(0, 0.3)
                w, h = max(1.0, w), max(1.0, h)
                frame_dets.append({
                    "track_id": track_id,
                    "x": float(x_noisy),
                    "y": float(y_noisy),
                    "w": float(w),
                    "h": float(h),
                    "true_x": float(x),
                    "true_y": float(y)
                })
        detections[frame_idx] = frame_dets

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2)

    print(f"[Detections] Saved to {output_path}")
    return detections


def visualize_trajectories(trajectories: Dict[int, Dict], output_path: str, bg_image_path: str = None):
    """可视化生成的轨迹 - 黄色单光子点，带背景图"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # 加载背景图
    bg_path = r"D:\codes\object_tracking\pics\1900\frame_002.png"
    if os.path.exists(bg_path):
        img = plt.imread(bg_path)
        ax.imshow(img, extent=[0, IMG_W, 0, IMG_H], cmap='gray', alpha=0.6)
    else:
        ax.set_facecolor("#0d0d1a")

    # 黄色单光子点效果 - 只显示点，不连线
    for track_id, data in trajectories.items():
        points = data["points"]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        # 只绘制单个光子点（黄色亮点）- 不画连线
        ax.scatter(xs, ys, c='yellow', s=15, marker='o',
                  edgecolors='none', zorder=5, alpha=0.9)

        # 起点和终点标记
        ax.scatter([xs[0]], [ys[0]], c='lime', s=40, marker='o',
                   edgecolors='white', linewidths=0.5, zorder=6)
        ax.scatter([xs[-1]], [ys[-1]], c='red', s=40, marker='s',
                   edgecolors='white', linewidths=0.5, zorder=6)

    ax.set_xlim(0, IMG_W)
    ax.set_ylim(0, IMG_H)
    ax.set_xlabel("X (pixel)", fontsize=12)
    ax.set_ylabel("Y (pixel)", fontsize=12)
    ax.set_title(f"Simulated UAV Trajectories (10 tracks, 25-30 yellow pixels)\nLeft-to-Right with Wave Motion", fontsize=12)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')
    ax.grid(True, alpha=0.3, linestyle='--', color='white')

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
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"[Visualization] Saved to {output_path}")


# =============================
# 主程序
# =============================
def main():
    print("=" * 60)
    print("Generating simulated UAV trajectories...")
    print("=" * 60)

    # 生成轨迹
    trajectories = generate_all_trajectories()

    # 打印统计信息
    print(f"\nGenerated {len(trajectories)} trajectories:")
    for track_id, data in trajectories.items():
        print(f"  Track {track_id}: {data['num_points']} points")

    # 保存GT
    save_gt_tracks(trajectories, GT_CSV_PATH)

    # 保存检测
    detections = save_detections(trajectories, DETECTION_JSON_PATH)

    # 可视化
    bg_path = r"D:\codes\object_tracking\pics\1900\frame_002.png"
    vis_path = os.path.join(OUTPUT_DIR, "simulated_trajectories.png")
    visualize_trajectories(trajectories, vis_path, bg_path)

    print("\n" + "=" * 60)
    print("Trajectory generation complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    return trajectories, detections


if __name__ == "__main__":
    main()