# -*- coding: utf-8 -*-
"""
多假设跟踪 (MHT) 最小可运行示例 - 4台无人机场景
包含航迹可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
from typing import List, Dict
import copy
import os


# ==================== 数据结构 ====================

@dataclass
class Detection:
    """检测目标"""
    x: float
    y: float
    frame_id: int
    det_id: int
    
@dataclass 
class Track:
    """航迹"""
    track_id: int
    positions: List[tuple] = field(default_factory=list)
    state: np.ndarray = None
    
    def __hash__(self):
        return hash(self.track_id)
    
    def __eq__(self, other):
        return self.track_id == other.track_id
    
    def predict(self, dt=1.0):
        if self.state is not None:
            self.state[0] += self.state[2] * dt
            self.state[1] += self.state[3] * dt
        return self.state[:2] if self.state is not None else None
    
    def update(self, det: Detection):
        if self.state is None:
            self.state = np.array([det.x, det.y, 0.0, 0.0])
        else:
            alpha = 0.7
            vx = det.x - self.state[0]
            vy = det.y - self.state[1]
            self.state[0] = det.x
            self.state[1] = det.y
            self.state[2] = alpha * vx + (1-alpha) * self.state[2]
            self.state[3] = alpha * vy + (1-alpha) * self.state[3]
        self.positions.append((det.x, det.y, det.frame_id))


@dataclass
class Hypothesis:
    """假设：一种可能的关联方案"""
    score: float
    tracks: Dict[int, Track]
    associations: List[tuple]
    
    def __lt__(self, other):
        return self.score > other.score


# ==================== MHT 跟踪器 ====================

class MHTTracker:
    def __init__(self, max_hypotheses=10, gate_threshold=50.0):
        self.max_hypotheses = max_hypotheses
        self.gate_threshold = gate_threshold
        self.hypotheses: List[Hypothesis] = []
        self.next_track_id = 0
        self.best_tracks: Dict[int, Track] = {}
        
    def _distance(self, track: Track, det: Detection) -> float:
        if track.state is None:
            return float('inf')
        pred = track.state[:2]
        return np.sqrt((pred[0] - det.x)**2 + (pred[1] - det.y)**2)
    
    def _generate_associations(self, tracks: Dict[int, Track], detections: List[Detection]) -> List[List[tuple]]:
        if not tracks or not detections:
            return [[]]
        
        track_list = list(tracks.values())
        n_tracks = len(track_list)
        n_dets = len(detections)
        
        cost_matrix = np.zeros((n_tracks, n_dets))
        for i, track in enumerate(track_list):
            for j, det in enumerate(detections):
                dist = self._distance(track, det)
                cost_matrix[i, j] = dist if dist < self.gate_threshold else 1e6
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        associations = []
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < self.gate_threshold:
                associations.append((track_list[i].track_id, detections[j].det_id))
        
        return [associations]
    
    def process_frame(self, detections: List[Detection], frame_id: int):
        if not self.hypotheses:
            initial_tracks = {}
            for det in detections:
                track = Track(track_id=self.next_track_id)
                track.update(det)
                initial_tracks[self.next_track_id] = track
                self.next_track_id += 1
            self.hypotheses = [Hypothesis(score=1.0, tracks=initial_tracks, associations=[])]
            self.best_tracks = copy.deepcopy(initial_tracks)
            return
        
        new_hypotheses = []
        
        for hyp in self.hypotheses:
            predicted_tracks = copy.deepcopy(hyp.tracks)
            for track in predicted_tracks.values():
                track.predict()
            
            assoc_options = self._generate_associations(predicted_tracks, detections)
            
            for associations in assoc_options:
                new_tracks = copy.deepcopy(predicted_tracks)
                new_score = hyp.score
                
                associated_dets = set()
                for track_id, det_id in associations:
                    det = next(d for d in detections if d.det_id == det_id)
                    new_tracks[track_id].update(det)
                    associated_dets.add(det_id)
                    new_score *= 0.95
                
                for det in detections:
                    if det.det_id not in associated_dets:
                        new_track = Track(track_id=self.next_track_id)
                        new_track.update(det)
                        new_tracks[self.next_track_id] = new_track
                        self.next_track_id += 1
                        new_score *= 0.5
                
                new_hypotheses.append(Hypothesis(
                    score=new_score,
                    tracks=new_tracks,
                    associations=associations
                ))
        
        new_hypotheses.sort(key=lambda h: h.score, reverse=True)
        self.hypotheses = new_hypotheses[:self.max_hypotheses]
        
        if self.hypotheses:
            self.best_tracks = copy.deepcopy(self.hypotheses[0].tracks)
    
    def get_tracks(self) -> Dict[int, Track]:
        return self.best_tracks


# ==================== 数据生成 ====================

def generate_uav_data(n_uavs=4, n_frames=50, noise_std=2.0):
    np.random.seed(42)
    
    uav_configs = [
        {'start': (10, 10), 'velocity': (3, 2)},
        {'start': (90, 10), 'velocity': (-2, 3)},
        {'start': (10, 90), 'velocity': (2, -2)},
        {'start': (90, 90), 'velocity': (-3, -1)},
    ]
    
    ground_truth = {i: [] for i in range(n_uavs)}
    detections_by_frame = {f: [] for f in range(n_frames)}
    det_id = 0
    
    for frame in range(n_frames):
        for uav_id in range(n_uavs):
            cfg = uav_configs[uav_id]
            true_x = cfg['start'][0] + cfg['velocity'][0] * frame
            true_y = cfg['start'][1] + cfg['velocity'][1] * frame
            ground_truth[uav_id].append((true_x, true_y, frame))
            
            if np.random.rand() < 0.95:
                det_x = true_x + np.random.randn() * noise_std
                det_y = true_y + np.random.randn() * noise_std
                detections_by_frame[frame].append(
                    Detection(x=det_x, y=det_y, frame_id=frame, det_id=det_id)
                )
                det_id += 1
    
    return ground_truth, detections_by_frame


# ==================== 可视化 ====================

def visualize_tracking(ground_truth, tracked_results, save_path="mht_tracking.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    
    # 左图：真实轨迹
    ax1 = axes[0]
    ax1.set_title('Ground Truth Trajectories', fontsize=12)
    for uav_id, positions in ground_truth.items():
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        ax1.plot(xs, ys, '-', color=colors[uav_id % len(colors)], 
                linewidth=2, label=f'UAV {uav_id}')
        ax1.scatter(xs[0], ys[0], color=colors[uav_id % len(colors)], 
                   s=100, marker='o', edgecolors='black', zorder=5)
        ax1.scatter(xs[-1], ys[-1], color=colors[uav_id % len(colors)], 
                   s=100, marker='s', edgecolors='black', zorder=5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 右图：跟踪结果
    ax2 = axes[1]
    ax2.set_title('MHT Tracking Results', fontsize=12)
    
    sorted_tracks = sorted(tracked_results.values(), 
                          key=lambda t: len(t.positions), reverse=True)
    
    for i, track in enumerate(sorted_tracks[:8]):
        if len(track.positions) < 3:
            continue
        xs = [p[0] for p in track.positions]
        ys = [p[1] for p in track.positions]
        color = colors[i % len(colors)]
        ax2.plot(xs, ys, '-', color=color, linewidth=2, 
                label=f'Track {track.track_id}', alpha=0.8)
        ax2.scatter(xs[0], ys[0], color=color, s=100, marker='o', 
                   edgecolors='black', zorder=5)
        ax2.scatter(xs[-1], ys[-1], color=color, s=100, marker='s', 
                   edgecolors='black', zorder=5)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    
    # 保存到当前目录
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图像已保存: {save_path}")


# ==================== 主程序 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("多假设跟踪 (MHT) 演示 - 4台无人机场景")
    print("=" * 60)
    
    # 生成数据
    print("\n[1] 生成模拟数据...")
    ground_truth, detections_by_frame = generate_uav_data(n_uavs=4, n_frames=50)
    total_dets = sum(len(d) for d in detections_by_frame.values())
    print(f"    - 无人机数量: 4")
    print(f"    - 总帧数: 50")
    print(f"    - 总检测数: {total_dets}")
    
    # 运行MHT
    print("\n[2] 运行MHT跟踪器...")
    tracker = MHTTracker(max_hypotheses=10, gate_threshold=30.0)
    
    for frame_id in range(50):
        dets = detections_by_frame[frame_id]
        tracker.process_frame(dets, frame_id)
    
    tracks = tracker.get_tracks()
    print(f"    - 生成航迹数: {len(tracks)}")
    
    valid_tracks = [t for t in tracks.values() if len(t.positions) >= 10]
    print(f"    - 有效航迹数 (>=10帧): {len(valid_tracks)}")
    
    # 可视化 - 保存到当前目录
    print("\n[3] 生成可视化结果...")
    save_path = os.path.join(os.path.dirname(__file__), "mht_tracking.png")
    visualize_tracking(ground_truth, tracks, save_path)
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
