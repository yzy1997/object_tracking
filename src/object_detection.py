# src/object_detection.py
import os
import numpy as np
from collections import deque
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.feature import peak_local_max

class EnhancedDroneDetector:
    def __init__(self, processor, update_files, 
                 y_threshold=40, eps=8, min_samples=1,
                 temporal_window=5, speed_threshold=0.8,
                 forbidden_zones=None, signal_adaptation=0.2):
        """
        改进后的无人机检测器初始化
        """
        self.processor = processor
        self.update_files = update_files
        self.y_threshold = y_threshold
        self.eps = eps
        self.min_samples = min_samples
        self.temporal_window = temporal_window
        self.speed_threshold = speed_threshold
        self.forbidden_zones = forbidden_zones or []
        self.signal_adaptation = signal_adaptation
        self.history = deque(maxlen=temporal_window)
        self.resolution = processor.resolution
        self.dynamic_threshold = 100  # 初始信号阈值

    def _is_in_forbidden_zone(self, x, y):
        """优化后的禁止区域检查"""
        for (x1, x2, y1, y2) in self.forbidden_zones:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    def _collect_changes(self):
        """动态阈值信号采集"""
        all_points = []
        for file in self.update_files:
            data = self.processor.read_radar_data(file)
            if not data:
                continue
                
            # 动态阈值调整
            signal_strengths = [d for _, _, d in data]
            if signal_strengths:
                new_threshold = np.percentile(signal_strengths, 80)
                self.dynamic_threshold = (1 - self.signal_adaptation) * self.dynamic_threshold \
                                       + self.signal_adaptation * new_threshold

            valid_points = [
                (x, y) for y, x, d in data 
                if y >= self.y_threshold 
                and d > self.dynamic_threshold * 0.7
                and not self._is_in_forbidden_zone(x, y)
            ]
            all_points.extend(valid_points)
        return np.array(all_points) if all_points else None

    def _calculate_movement(self, current_points):
        """优化速度计算逻辑"""
        moving_features = []
        for pt in current_points:
            speed = 0
            if self.history:
                displacements = []
                for prev_points in self.history:
                    if prev_points.size > 0:
                        dists = np.linalg.norm(pt - prev_points, axis=1)
                        min_dist = np.min(dists) if len(dists) > 0 else 0
                        displacements.append(min_dist)
                speed = np.percentile(displacements, 75) if displacements else 0
            moving_features.append(np.append(pt, speed))
        return np.array(moving_features)

    def _temporal_accumulation(self):
        """时域信号累积检测"""
        accum_map = np.zeros(self.resolution)
        for frame_points in self.history:
            for x, y in frame_points:
                x_idx = min(int(x), self.resolution[0]-1)
                y_idx = min(int(y), self.resolution[1]-1)
                accum_map[x_idx, y_idx] += 1
        
        peaks = peak_local_max(
            accum_map, 
            min_distance=3,
            threshold_abs=self.temporal_window * 0.6,
            num_peaks=10
        )
        return peaks

    def _merge_detections(self, boxes, peaks):
        """融合聚类结果和时域结果"""
        combined = []
        # 转换时域峰值点为检测框
        peak_boxes = [(max(0, p[1]-2), max(0, p[0]-2), 4, 4) for p in peaks]
        
        # 合并并去重
        all_boxes = boxes + peak_boxes
        for box in all_boxes:
            if not self._is_duplicate(box, combined):
                combined.append(box)
        return combined

    def _is_duplicate(self, new_box, existing_boxes, threshold=5):
        """检测框去重逻辑"""
        x_new, y_new, w_new, h_new = new_box
        center_new = (x_new + w_new/2, y_new + h_new/2)
        
        for box in existing_boxes:
            x, y, w, h = box
            center = (x + w/2, y + h/2)
            distance = np.linalg.norm(np.array(center_new) - np.array(center))
            if distance < threshold:
                return True
        return False

    def detect(self):
        """改进后的检测流程"""
        points = self._collect_changes()
        if points is None or len(points) == 0:
            return []
            
        self.history.append(points.copy())
        enhanced_points = self._calculate_movement(points)
        
        # 动态参数调整
        if enhanced_points.size > 0:
            avg_movement = np.percentile(enhanced_points[:,2], 75)
            dynamic_eps = self.eps * (1 + avg_movement/3)
        else:
            dynamic_eps = self.eps
        
        # 标准化聚类
        scaler = StandardScaler()
        scaled_points = scaler.fit_transform(enhanced_points[:,:2])
        
        clustering = DBSCAN(
            eps=dynamic_eps*0.3,
            min_samples=self.min_samples,
            metric='euclidean'
        ).fit(scaled_points)
        
        # 生成检测框
        boxes = []
        for label in set(clustering.labels_):
            if label == -1:
                continue
                
            mask = clustering.labels_ == label
            cluster = enhanced_points[mask]
            
            # 速度过滤
            avg_speed = np.percentile(cluster[:,2], 75)
            if avg_speed < self.speed_threshold:
                continue
                
            # 生成检测框
            x_min, y_min = cluster[:,:2].min(axis=0)
            x_max, y_max = cluster[:,:2].max(axis=0)
            
            boxes.append((
                max(0, x_min-2), 
                max(0, y_min-2),
                min(self.resolution[0], x_max - x_min + 4),
                min(self.resolution[1], y_max - y_min + 4)
            ))
        
        # 时域累积检测
        temporal_peaks = self._temporal_accumulation()
        combined_boxes = self._merge_detections(boxes, temporal_peaks)
        
        return combined_boxes

    def plot_result(self, boxes, output_path):
        """增强可视化方法"""
        plt.figure(figsize=(12, 12))
        plt.imshow(self.processor.radar_image, cmap='hot', origin='lower')
        plt.title("Enhanced Drone Detection Result")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        
        # 绘制禁止区域
        for (x1, x2, y1, y2) in self.forbidden_zones:
            rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=1, edgecolor='white', 
                           facecolor='gray', alpha=0.3)
            plt.gca().add_patch(rect)
            
        # 绘制检测框
        for (x, y, w, h) in boxes:
            rect = Rectangle((x, y), w, h, 
                           linewidth=2, edgecolor='lime', 
                           facecolor='none', label='Detection')
            plt.gca().add_patch(rect)
        
        # 绘制时域累积点
        accum_map = self._temporal_accumulation()
        if accum_map.size > 0:
            plt.scatter(accum_map[:,1], accum_map[:,0], 
                        s=80, c='cyan', marker='x',
                        linewidths=2, label='Temporal Peaks')
        
        plt.legend(loc='upper right')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    @staticmethod
    def validate_detection(test_cases, processor_class, **detector_args):
        """更新验证方法"""
        results = []
        for case in test_cases:
            try:
                processor = processor_class(resolution=(132,132), shift_pixel=4)
                processor.process_files(case['file'], [])
                
                detector = EnhancedDroneDetector(processor, [], **detector_args)
                boxes = detector.detect()
                
                result = {
                    'file': os.path.basename(case['file']),
                    'expected': case['expected'],
                    'detected': len(boxes),
                    'status': 'PASS' if len(boxes)==case['expected'] else 'FAIL'
                }
            except Exception as e:
                result = {
                    'file': os.path.basename(case['file']),
                    'status': 'ERROR',
                    'error': str(e)
                }
            results.append(result)
        return results
