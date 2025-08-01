import numpy as np
from scipy.optimize import linear_sum_assignment

#################################################
#             bbox＋匈牙利算法                   #
#################################################
def iou(box1, box2):
    # box = (xmin, xmax, ymin, ymax)
    x1, x2 = max(box1[0], box2[0]), min(box1[1], box2[1])
    y1, y2 = max(box1[2], box2[2]), min(box1[3], box2[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    inter = (x2 - x1 + 1) * (y2 - y1 + 1)
    area1 = (box1[1] - box1[0] + 1) * (box1[3] - box1[2] + 1)
    area2 = (box2[1] - box2[0] + 1) * (box2[3] - box2[2] + 1)
    return inter / float(area1 + area2 - inter)

class Track:
    def __init__(self, bbox, track_id):
        self.bbox = bbox
        self.id = track_id
        self.hits = 1       # 连续被匹配到的次数
        self.misses = 0     # 连续漏检的次数

    def predict(self):
        # 如果要加卡尔曼滤波，这里返回预测位置
        return self.bbox

    def update(self, bbox):
        self.bbox = bbox
        self.hits += 1
        self.misses = 0

class DroneTracker:
    def __init__(self, max_misses=3, iou_thresh=0.3):
        self.tracks = []
        self.next_id = 0
        self.max_misses = max_misses
        self.iou_thresh = iou_thresh

    def update(self, detections):
        N = len(self.tracks)
        M = len(detections)
        if N == 0:
            # 没有现有轨迹，所有检测都生新 track
            for det in detections:
                self.tracks.append(Track(det, self.next_id))
                self.next_id += 1
            return self.tracks

        # 1) 构造 cost 矩阵：1 - IoU
        cost = np.ones((N, M), dtype=np.float32)
        for i, tr in enumerate(self.tracks):
            tb = tr.predict()
            for j, det in enumerate(detections):
                cost[i, j] = 1.0 - iou(tb, det)

        # 2) 匈牙利最优匹配
        row_idx, col_idx = linear_sum_assignment(cost)

        assigned_tracks = set()
        assigned_dets = set()

        # 3) 更新匹配到的
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < (1.0 - self.iou_thresh):
                self.tracks[r].update(detections[c])
                assigned_tracks.add(r)
                assigned_dets.add(c)

        # 4) 处理漏检的 track
        for i, tr in enumerate(self.tracks):
            if i not in assigned_tracks:
                tr.misses += 1

        # 5) 删除漏检超过阈值的 track
        self.tracks = [tr for tr in self.tracks if tr.misses <= self.max_misses]

        # 6) 未分配到任何 track 的检测 -> 新建 track
        for j, det in enumerate(detections):
            if j not in assigned_dets:
                self.tracks.append(Track(det, self.next_id))
                self.next_id += 1

        return self.tracks
