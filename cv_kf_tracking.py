# main2.py
# -*- coding: utf-8 -*-

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
# 设置字体以支持中文
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from src.denoise import RadarDenoiser
from src.pixel_shifting_correction import RadarImageProcessor
from src.object_detection import SpatialDroneDetector
from src.CA_EKF import KalmanPredictor


def natural_sort_key(s):
    parts = re.split(r'(\d+)', s)
    return [int(p) if p.isdigit() else p for p in parts]


def box_center(b):
    x0, x1, y0, y1 = b
    return np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=np.float32)


def calculate_metrics(actual_centers, predicted_centers, pixel_to_meter=5.0):
    """
    计算追踪性能指标
    pixel_to_meter: 像素到米的转换系数（600m距离下估计）
    """
    # 检测概率
    valid_detections = sum(1 for c in actual_centers if c is not None)
    detection_prob = valid_detections / len(actual_centers) if actual_centers else 0
    
    # 定位误差RMSE
    errors_px = []
    frame_errors = []  # 存储每帧的误差
    for i in range(len(predicted_centers)-1):
        if predicted_centers[i] is not None and actual_centers[i+1] is not None:
            error = np.linalg.norm(predicted_centers[i] - actual_centers[i+1])
            errors_px.append(error)
            frame_errors.append((i+1, error * pixel_to_meter))  # 帧号和误差(米)
        else:
            frame_errors.append((i+1, None))  # 没有误差的帧
    
    rmse_px = np.sqrt(np.mean(np.array(errors_px)**2)) if errors_px else float('nan')
    rmse_m = rmse_px * pixel_to_meter
    mae_px = np.mean(errors_px) if errors_px else float('nan')
    mae_m = mae_px * pixel_to_meter
    
    # ID switches (追踪中断次数)
    id_switches = 0
    prev_valid = False
    for c in actual_centers:
        if c is not None:
            if not prev_valid and id_switches > 0:
                id_switches += 1
            prev_valid = True
        else:
            if prev_valid:
                prev_valid = False
    
    return {
        'detection_prob': detection_prob,
        'rmse_px': rmse_px,
        'rmse_m': rmse_m,
        'mae_px': mae_px,
        'mae_m': mae_m,
        'id_switches': max(0, id_switches - 1),
        'frame_errors': frame_errors
    }


def plot_rmse_curve(frame_errors, out_folder, pixel_to_meter=5.0):
    """绘制逐帧RMSE曲线"""
    frames = []
    errors_m = []
    cumulative_rmse = []
    
    all_errors_so_far = []
    
    for frame_num, error_m in frame_errors:
        frames.append(frame_num)
        
        if error_m is not None:
            errors_m.append(error_m)
            all_errors_so_far.append(error_m)
            current_rmse = np.sqrt(np.mean(np.array(all_errors_so_far)**2))
            cumulative_rmse.append(current_rmse)
        else:
            errors_m.append(None)
            if all_errors_so_far:
                cumulative_rmse.append(np.sqrt(np.mean(np.array(all_errors_so_far)**2)))
            else:
                cumulative_rmse.append(None)
    
    # 只有有效数据时才绘图
    valid_errors = [e for e in errors_m if e is not None]
    if not valid_errors:
        print("No valid errors to plot RMSE curve")
        return
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 上图：逐帧误差
    valid_frames = [f for f, e in zip(frames, errors_m) if e is not None]
    valid_errors = [e for e in errors_m if e is not None]
    
    if valid_errors:
        ax1.plot(valid_frames, valid_errors, 'b-o', markersize=4, linewidth=1.5, label='Frame Error')
        mean_error = np.mean(valid_errors)
        ax1.axhline(y=mean_error, color='r', linestyle='--', alpha=0.7, 
                    label=f'Mean Error = {mean_error:.2f}m')
    
    ax1.set_xlabel('Frame Number', fontsize=11)
    ax1.set_ylabel('Localization Error (m)', fontsize=11)
    ax1.set_title('Frame-by-Frame Localization Error', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # 下图：累积RMSE
    valid_cumulative = [(f, r) for f, r in zip(frames, cumulative_rmse) if r is not None]
    if valid_cumulative:
        cum_frames, cum_rmse = zip(*valid_cumulative)
        ax2.plot(cum_frames, cum_rmse, 'g-', linewidth=2, label='Cumulative RMSE')
        ax2.fill_between(cum_frames, 0, cum_rmse, alpha=0.3, color='green')
        
        final_rmse = cum_rmse[-1]
        ax2.axhline(y=final_rmse, color='r', linestyle='--', alpha=0.7, 
                    label=f'Final RMSE = {final_rmse:.2f}m')
    
    ax2.set_xlabel('Frame Number', fontsize=11)
    ax2.set_ylabel('Cumulative RMSE (m)', fontsize=11)
    ax2.set_title('Cumulative RMSE Over Time', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, 'rmse_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 打印统计信息
    if valid_errors:
        print(f"\nError Statistics:")
        print(f"  Min Error: {min(valid_errors):.2f} m (Frame {valid_frames[valid_errors.index(min(valid_errors))]})")
        print(f"  Max Error: {max(valid_errors):.2f} m (Frame {valid_frames[valid_errors.index(max(valid_errors))]})")
        print(f"  Mean Error: {np.mean(valid_errors):.2f} m")
        print(f"  Std Error: {np.std(valid_errors):.2f} m")


def main():
    # ========== 配置参数 ==========
    data_folder = r"D:\codes\object_tracking\data\600m"
    out_folder = "pics/600m_final"
    os.makedirs(out_folder, exist_ok=True)
    
    # 像素到米的转换（600m距离，估计值）
    PIXEL_TO_METER = 5.0
    
    # 检查数据文件夹
    if not os.path.exists(data_folder):
        print(f"[ERROR] Data folder not found: {data_folder}")
        return
    
    # 1) 获取文件列表
    all_files = sorted(glob.glob(os.path.join(data_folder, "*.txt")), 
                      key=natural_sort_key)
    
    print(f"Found {len(all_files)} files")
    if len(all_files) < 2:
        print("[ERROR] Need at least background frame + 1 update frame")
        return
    
    full_scan = all_files[0]
    frame_files = all_files[1:]
    print(f"Background frame: {os.path.basename(full_scan)}")
    print(f"Frames to process: {len(frame_files)}\n")
    
    # 2) 初始化去噪器（放宽参数以检测小目标）
    denoiser = RadarDenoiser(
        kernel_size=3, 
        min_neighbors=0,  # 允许完全孤立的点
        filter_size=3, 
        thr_percentile=85,  # 降低阈值
        rel_floor=0.08,
        morph_open=0,
        morph_close=0  # 不做形态学操作
    )
    
    # 3) 初始化处理器
    proc = RadarImageProcessor(shift_pixel=4, denoiser=denoiser)
    
    # 4) 初始化检测器（大幅放宽参数）
    detector = SpatialDroneDetector(
        processor=proc,
        full_scan_file=full_scan,
        update_files=[],
        use_otsu=False,
        thr_percentile=25,  # 更低的阈值
        thr_rel_floor=0.02,
        median_size=1,
        tophat_radius=1,
        cluster_dist=5,
        topk=20,
        min_area=1,
        max_area=40,
        max_width=15,
        max_height=15,
        min_y=100,  # 扩大搜索范围
        bg_mode='static',
        temporal_window=1,
        debug=True  # 打开调试信息
    )
    
    # 5) 追踪参数
    DRONE_Y_RANGE = (100, 127)  # 扩大Y范围
    EXPECTED_VX = -3.0  # 预期水平速度
    MAX_SPEED = 10.0
    
    # 初始化Kalman滤波器
    predictor = KalmanPredictor(process_var=0.5, meas_var=20.0)  # 增加不确定性
    
    # 存储轨迹
    actual_centers = []
    predicted_centers = []
    
    # 追踪状态
    gate_radius = 10.0  # 增大初始门控半径
    miss_count = 0
    tracking_started = False
    last_center = None
    
    print("=== Start Tracking ===\n")
    
    # ========== 主循环 ==========
    for idx, fp in enumerate(frame_files, start=1):
        print(f"\n--- Frame {idx:03d} ---")
        detector.update_files = [fp]
        
        # 执行检测
        all_boxes = detector.detect()
        print(f"Total detections: {len(all_boxes)}")
        
        # 过滤：只保留上方区域的检测
        candidate_boxes = []
        for b in all_boxes:
            cx, cy = box_center(b)
            if DRONE_Y_RANGE[0] <= cy <= DRONE_Y_RANGE[1]:
                candidate_boxes.append(b)
                print(f"  Candidate at ({cx:.1f}, {cy:.1f})")
        
        print(f"Candidates in ROI: {len(candidate_boxes)}")
        
        # 选择策略
        sel_box = None
        
        if not tracking_started:
            # 初始化阶段：优先右侧，但接受任何合理的检测
            if candidate_boxes:
                # 按X坐标排序，选择最右边的
                candidate_boxes.sort(key=lambda b: -box_center(b)[0])
                sel_box = candidate_boxes[0]
                tracking_started = True
                cx, cy = box_center(sel_box)
                print(f"[INIT] Start tracking at ({cx:.1f}, {cy:.1f})")
            elif hasattr(detector, 'tophat_img') and detector.tophat_img is not None:
                # 在整个上方区域找最亮点
                tophat = detector.tophat_img
                y0, y1 = DRONE_Y_RANGE
                y0 = max(0, min(y0, tophat.shape[0]-1))
                y1 = max(0, min(y1, tophat.shape[0]-1))
                
                if y1 > y0:
                    roi_tophat = tophat[y0:y1+1, :]
                    
                    if roi_tophat.size > 0:
                        max_val = roi_tophat.max()
                        print(f"  TopHat max in ROI: {max_val}")
                        if max_val > 20:  # 降低阈值
                            ry, rx = np.unravel_index(np.argmax(roi_tophat), roi_tophat.shape)
                            peak_x, peak_y = rx, y0 + ry
                            
                            sel_box = (max(0, peak_x-3), min(127, peak_x+3),
                                      max(0, peak_y-3), min(127, peak_y+3))
                            tracking_started = True
                            print(f"[INIT] Start tracking (brightest) at ({peak_x}, {peak_y})")
        
        else:
            # 已经开始追踪，使用预测引导
            pred_center = predictor.predict(future_frames=1)[0]
            current_gate = gate_radius * (1.0 + 0.5 * miss_count)
            print(f"  Predicted: ({pred_center[0]:.1f}, {pred_center[1]:.1f}), gate={current_gate:.1f}")
            
            # 找最接近预测的框
            best_box = None
            best_dist = float('inf')
            
            for b in candidate_boxes:
                c = box_center(b)
                dist = np.linalg.norm(c - pred_center)
                
                # 考虑运动方向
                if last_center is not None:
                    dx = c[0] - last_center[0]
                    if dx > 5:  # 惩罚大幅向右的运动
                        dist += dx * 3
                
                if dist < best_dist:
                    best_dist = dist
                    best_box = b
            
            if best_box and best_dist < current_gate:
                sel_box = best_box
                print(f"  Selected box at distance {best_dist:.1f}")
            else:
                # 扩大搜索：在预测位置附近找最亮点
                if hasattr(detector, 'tophat_img') and detector.tophat_img is not None:
                    tophat = detector.tophat_img
                    H, W = tophat.shape
                    search_r = int(current_gate)
                    
                    x0 = max(0, int(pred_center[0] - search_r))
                    x1 = min(W-1, int(pred_center[0] + search_r))
                    y0 = max(0, int(pred_center[1] - search_r))
                    y1 = min(H-1, int(pred_center[1] + search_r))
                    
                    if x1 > x0 and y1 > y0:
                        roi = tophat[y0:y1+1, x0:x1+1]
                        if roi.size > 0:
                            max_val = roi.max()
                            print(f"  TopHat max near prediction: {max_val}")
                            if max_val > 15:  # 很低的阈值
                                ry, rx = np.unravel_index(np.argmax(roi), roi.shape)
                                peak_x, peak_y = x0 + rx, y0 + ry
                                sel_box = (max(0, peak_x-3), min(W-1, peak_x+3),
                                          max(0, peak_y-3), min(H-1, peak_y+3))
                                print(f"  Using brightest point at ({peak_x}, {peak_y})")
        
        # ========== 生成每帧图像 ==========
        img = detector.build_accumulated_image()
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 左图：雷达图像和追踪
        axes[0].imshow(img, cmap='jet', origin='lower')
        axes[0].set_title(f'Frame {idx:03d} - Radar Image', fontsize=12)
        
        # 画所有候选框（黄色）
        for b in candidate_boxes:
            x0, x1, y0, y1 = b
            rect = Rectangle((x0, y0), x1-x0, y1-y0,
                           edgecolor='yellow', facecolor='none', lw=1, alpha=0.5)
            axes[0].add_patch(rect)
        
        # 画选中的追踪框（红色）
        if sel_box:
            x0, x1, y0, y1 = sel_box
            rect = Rectangle((x0, y0), x1-x0, y1-y0,
                           edgecolor='red', facecolor='none', lw=2.5)
            axes[0].add_patch(rect)
            cx, cy = box_center(sel_box)
            axes[0].plot(cx, cy, 'r+', markersize=10, markeredgewidth=2)
        
        # 画历史轨迹
        valid_centers = [c for c in actual_centers if c is not None]
        if len(valid_centers) >= 2:
            arr = np.array(valid_centers)
            axes[0].plot(arr[:,0], arr[:,1], 'w-', linewidth=2.5, alpha=0.9)
            axes[0].plot(arr[:,0], arr[:,1], 'wo', markersize=5, alpha=0.8)
        
        # 画预测位置（绿色）
        if getattr(predictor, "inited", False):
            pred = predictor.predict(future_frames=1)[0]
            axes[0].plot(pred[0], pred[1], 'gx', markersize=10, markeredgewidth=2)
        
        axes[0].axis('off')
        
        # 右图：TopHat图像
        if hasattr(detector, 'tophat_img') and detector.tophat_img is not None:
            axes[1].imshow(detector.tophat_img, cmap='hot', origin='lower')
            axes[1].set_title(f'TopHat Enhancement', fontsize=12)
            if sel_box:
                x0, x1, y0, y1 = sel_box
                rect = Rectangle((x0, y0), x1-x0, y1-y0,
                               edgecolor='red', facecolor='none', lw=2)
                axes[1].add_patch(rect)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, f'frame_{idx:03d}.png'), dpi=100)
        plt.close()
        
        # ========== 更新状态 ==========
        if sel_box:
            center = predictor.update(idx, sel_box)
            actual_centers.append(center)
            last_center = center
            miss_count = 0
            
            # 初次检测时设置速度先验
            if len([c for c in actual_centers if c is not None]) == 1:
                predictor.x[2] = EXPECTED_VX
                predictor.x[3] = 0.0
                print(f"  Set initial velocity: vx={EXPECTED_VX}")
        else:
            actual_centers.append(None)
            miss_count = min(miss_count + 1, 5)
            print(f"  MISSED (count={miss_count})")
            
            if getattr(predictor, "inited", False):
                predictor.predict(future_frames=1)
        
        # 记录预测
        if getattr(predictor, "inited", False):
            next_pred = predictor.predict(future_frames=1)[0]
            predicted_centers.append(next_pred)
        else:
            predicted_centers.append(None)
    
    # ========== 计算性能指标 ==========
    metrics = calculate_metrics(actual_centers, predicted_centers, PIXEL_TO_METER)
    
    print("\n=== Tracking Performance ===")
    print(f"Detection Probability: {metrics['detection_prob']:.2%}")
    print(f"RMSE: {metrics['rmse_m']:.2f} m ({metrics['rmse_px']:.2f} pixels)")
    print(f"MAE:  {metrics['mae_m']:.2f} m ({metrics['mae_px']:.2f} pixels)")
    print(f"ID Switches: {metrics['id_switches']}")
    
    # 分析运动
    valid_centers = [c for c in actual_centers if c is not None]
    if len(valid_centers) >= 2:
        centers_arr = np.array(valid_centers)
        distances = np.sqrt(np.sum(np.diff(centers_arr, axis=0)**2, axis=1))
        avg_speed_px = np.mean(distances)
        avg_speed_m = avg_speed_px * PIXEL_TO_METER
        
        total_dist_px = np.sum(distances)
        total_dist_m = total_dist_px * PIXEL_TO_METER
        
        print(f"\nMotion Analysis:")
        print(f"Average Speed: {avg_speed_m:.1f} m/frame ({avg_speed_px:.1f} px/frame)")
        print(f"Total Distance: {total_dist_m:.1f} m ({total_dist_px:.1f} pixels)")
    
    # ========== 绘制RMSE曲线 ==========
    if metrics['rmse_px'] != float('nan'):
        plot_rmse_curve(metrics['frame_errors'], out_folder, PIXEL_TO_METER)
    
    # ========== 生成最终汇总图 ==========
    bg = detector.build_accumulated_image()
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(bg, cmap='jet', origin='lower')
    ax = plt.gca()
    
    # 画真实轨迹
    if valid_centers:
        centers_arr = np.array(valid_centers)
        ax.plot(centers_arr[:,0], centers_arr[:,1], 'w-', 
               linewidth=3, label='Actual Track', alpha=0.9)
        ax.plot(centers_arr[:,0], centers_arr[:,1], 'wo', 
               markersize=6, markeredgecolor='black', markeredgewidth=0.5)
        
        # 标记起点和终点
        ax.plot(centers_arr[0,0], centers_arr[0,1], 'go', 
               markersize=12, label='Start', markeredgewidth=2)
        ax.plot(centers_arr[-1,0], centers_arr[-1,1], 'ro', 
               markersize=12, label='End', markeredgewidth=2)
    
    # 画预测轨迹
    pred_valid = [p for p in predicted_centers if p is not None]
    if pred_valid:
        preds_arr = np.array(pred_valid)
        ax.plot(preds_arr[:,0], preds_arr[:,1], 'r--', 
               linewidth=2, alpha=0.7, label='Predicted Track')
    
    # 添加图例和标题
    if valid_centers:  # 只有有轨迹时才添加图例
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    ax.set_title(f'Single Photon Radar Drone Tracking @ 600m\n'
                f'Detection Rate:{metrics["detection_prob"]:.1%} | '
                f'RMSE:{metrics["rmse_m"]:.1f}m | '
                f'ID Switches:{metrics["id_switches"]}',
                fontsize=13, fontweight='bold')
    ax.set_xlabel('X (pixels)', fontsize=11)
    ax.set_ylabel('Y (pixels)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, 'tracking_result.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nAll results saved to: {out_folder}")


if __name__ == "__main__":
    main()