# src/data_association.py
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict, Optional
import json
import os
import time


class Track:
    """单个目标轨迹"""
    def __init__(self, track_id: int, init_box: Tuple, init_frame: int):
        self.track_id = track_id
        self.history = [init_box]  # 历史检测框
        self.centers = [self._box_center(init_box)]  # 历史中心点
        self.frames = [init_frame]  # 出现帧号
        self.predicted_center = None  # 预测的下一个位置
        self.missed_count = 0  # 连续丢失计数
        self.active = True  # 是否活跃
        
    @staticmethod
    def _box_center(box):
        x0, x1, y0, y1 = box
        return np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
    
    def update(self, box: Tuple, frame_idx: int):
        """更新轨迹"""
        self.history.append(box)
        self.centers.append(self._box_center(box))
        self.frames.append(frame_idx)
        self.missed_count = 0
        
    def predict(self, predictor, frame_idx: int):
        """预测下一帧位置"""
        if len(self.centers) >= 2:
            # 简单线性预测
            last_center = self.centers[-1]
            if len(self.centers) >= 3:
                # 使用最近两帧的速度
                v1 = self.centers[-1] - self.centers[-2]
                v2 = self.centers[-2] - self.centers[-3]
                velocity = (v1 + v2) / 2.0
            else:
                velocity = self.centers[-1] - self.centers[-2]
            
            self.predicted_center = last_center + velocity
        else:
            self.predicted_center = self.centers[-1] if self.centers else None
            
    def mark_missed(self):
        """标记丢失"""
        self.missed_count += 1
        if self.missed_count > 5:  # 连续5帧丢失则终止
            self.active = False
    
    def get_last_center(self):
        """获取最后一个中心点"""
        return self.centers[-1] if self.centers else None
    
    def get_last_box(self):
        """获取最后一个检测框"""
        return self.history[-1] if self.history else None


class NearestNeighborAssociator:
    """最近邻数据关联器"""
    
    def __init__(self, max_distance: float = 20.0, 
                 use_prediction: bool = True,
                 iou_threshold: float = 0.1):
        """
        参数:
            max_distance: 最大关联距离(像素)
            use_prediction: 是否使用预测位置进行关联
            iou_threshold: IoU阈值，用于关联验证
        """
        self.max_distance = max_distance
        self.use_prediction = use_prediction
        self.iou_threshold = iou_threshold
        
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 0
        self.frame_count = 0
        
        # 性能统计
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_tracks': 0,
            'id_switches': 0,
            'fragments': 0,
            'motp': [],  # 多目标跟踪精度
            'mota': [],  # 多目标跟踪准确度
            'mostly_tracked': 0,
            'mostly_lost': 0,
            'partially_tracked': 0
        }
        
    def _calculate_iou(self, box1, box2):
        """计算两个框的IoU"""
        x1_min, x1_max, y1_min, y1_max = box1
        x2_min, x2_max, y2_min, y2_max = box2
        
        # 计算交集
        inter_x_min = max(x1_min, x2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_min = max(y1_min, y2_min)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
            
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # 计算并集
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _calculate_distance(self, center1, center2):
        """计算两个中心点的欧氏距离"""
        return np.linalg.norm(center1 - center2)
    
    def associate(self, detections: List[Tuple], frame_idx: int) -> Dict[int, Tuple]:
        """
        执行数据关联
        
        参数:
            detections: 当前帧的检测框列表 [(x0,x1,y0,y1), ...]
            frame_idx: 当前帧索引
            
        返回:
            assignments: 字典 {track_id: detection_box}
        """
        self.frame_count = frame_idx
        self.stats['total_frames'] += 1
        self.stats['total_detections'] += len(detections)
        
        # 更新现有轨迹的预测
        active_tracks = {tid: track for tid, track in self.tracks.items() if track.active}
        for track in active_tracks.values():
            track.predict(None, frame_idx)
        
        # 如果没有检测或没有轨迹，直接返回
        if not detections or not active_tracks:
            # 处理新检测或丢失轨迹
            assignments = self._handle_no_associations(detections, active_tracks, frame_idx)
            return assignments
        
        # 构建成本矩阵
        n_tracks = len(active_tracks)
        n_dets = len(detections)
        cost_matrix = np.full((n_tracks, n_dets), self.max_distance * 10, dtype=np.float32)
        
        track_ids = list(active_tracks.keys())
        for i, track_id in enumerate(track_ids):
            track = active_tracks[track_id]
            
            # 使用预测位置或最后位置作为参考
            if self.use_prediction and track.predicted_center is not None:
                ref_center = track.predicted_center
            else:
                ref_center = track.get_last_center()
                
            if ref_center is None:
                continue
                
            for j, det_box in enumerate(detections):
                det_center = Track._box_center(det_box)
                distance = self._calculate_distance(ref_center, det_center)
                
                # 如果距离太大，不关联
                if distance <= self.max_distance:
                    # 添加IoU惩罚
                    iou = self._calculate_iou(track.get_last_box(), det_box) if track.get_last_box() else 0
                    if iou < self.iou_threshold:
                        distance *= (1.0 + (1.0 - iou))
                    
                    cost_matrix[i, j] = distance
        
        # 使用匈牙利算法进行最优匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        assignments = {}
        matched_dets = set()
        matched_tracks = set()
        
        # 处理匹配结果
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < self.max_distance:
                track_id = track_ids[i]
                assignments[track_id] = detections[j]
                matched_dets.add(j)
                matched_tracks.add(i)
                
                # 更新轨迹
                self.tracks[track_id].update(detections[j], frame_idx)
        
        # 处理未匹配的检测（新目标）
        for j in range(n_dets):
            if j not in matched_dets:
                new_track_id = self._create_new_track(detections[j], frame_idx)
                assignments[new_track_id] = detections[j]
        
        # 处理未匹配的轨迹（丢失目标）
        for i in range(n_tracks):
            if i not in matched_tracks:
                track_id = track_ids[i]
                self.tracks[track_id].mark_missed()
                if not self.tracks[track_id].active:
                    self.stats['fragments'] += 1
        
        return assignments
    
    def _handle_no_associations(self, detections, active_tracks, frame_idx):
        """处理没有关联的情况"""
        assignments = {}
        
        # 创建新轨迹
        for det in detections:
            new_track_id = self._create_new_track(det, frame_idx)
            assignments[new_track_id] = det
        
        # 标记丢失的轨迹
        for track_id, track in active_tracks.items():
            track.mark_missed()
            if not track.active:
                self.stats['fragments'] += 1
                
        return assignments
    
    def _create_new_track(self, detection_box, frame_idx):
        """创建新轨迹"""
        track_id = self.next_track_id
        self.next_track_id += 1
        self.tracks[track_id] = Track(track_id, detection_box, frame_idx)
        self.stats['total_tracks'] += 1
        return track_id
    
    def get_active_tracks(self):
        """获取活跃轨迹"""
        return {tid: track for tid, track in self.tracks.items() if track.active}
    
    def get_all_tracks(self):
        """获取所有轨迹"""
        return self.tracks
    
    def save_statistics(self, output_dir: str, algorithm_name: str = "NN"):
        """保存性能统计"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 计算MOT指标
        if self.stats['total_frames'] > 0:
            # 这里可以添加更复杂的MOT指标计算
            stats_summary = {
                'algorithm': algorithm_name,
                'total_frames': self.stats['total_frames'],
                'total_detections': self.stats['total_detections'],
                'total_tracks': self.stats['total_tracks'],
                'id_switches': self.stats['id_switches'],
                'fragments': self.stats['fragments'],
                'avg_tracks_per_frame': self.stats['total_tracks'] / self.stats['total_frames'] if self.stats['total_frames'] > 0 else 0,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 保存为JSON
            stats_file = os.path.join(output_dir, f"{algorithm_name}_stats.json")
            with open(stats_file, 'w') as f:
                json.dump(stats_summary, f, indent=2)
            
            # 保存轨迹数据
            tracks_data = {}
            for track_id, track in self.tracks.items():
                tracks_data[track_id] = {
                    'frames': track.frames,
                    'centers': [c.tolist() for c in track.centers],
                    'boxes': track.history,
                    'active': track.active,
                    'missed_count': track.missed_count
                }
            
            tracks_file = os.path.join(output_dir, f"{algorithm_name}_tracks.json")
            with open(tracks_file, 'w') as f:
                json.dump(tracks_data, f, indent=2)
            
            return stats_summary
        return None


class GNNAssociator(NearestNeighborAssociator):
    """全局最近邻关联器（使用匈牙利算法）"""
    
    def __init__(self, max_distance: float = 25.0, 
                 use_prediction: bool = True,
                 iou_threshold: float = 0.1):
        super().__init__(max_distance, use_prediction, iou_threshold)
        self.algorithm_name = "GNN"


class JPDAAssociator(NearestNeighborAssociator):
    """联合概率数据关联器（简化版）"""
    
    def __init__(self, max_distance: float = 30.0,
                 use_prediction: bool = True,
                 iou_threshold: float = 0.2,
                 gate_probability: float = 0.95):
        super().__init__(max_distance, use_prediction, iou_threshold)
        self.gate_probability = gate_probability
        self.algorithm_name = "JPDA"
        
    def associate(self, detections: List[Tuple], frame_idx: int) -> Dict[int, Tuple]:
        """JPDA关联（简化实现）"""
        # 这里可以实现更复杂的JPDA逻辑
        # 目前先使用父类的最近邻关联
        return super().associate(detections, frame_idx)