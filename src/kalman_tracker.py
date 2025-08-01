import numpy as np
from scipy.optimize import linear_sum_assignment

class KalmanTrack:
    """
    简单2D常速度卡尔曼滤波器
    状态： [x, y, vx, vy]
    测量： [x, y]
    """
    count = 0

    def __init__(self, bbox, dt=1.0, var_process=1e-2, var_measure=1e-1):
        """
        bbox: (xmin, xmax, ymin, ymax)
        dt: 帧间隔
        var_process: 过程噪声方差
        var_measure: 观测噪声方差
        """
        self.id = KalmanTrack.count
        KalmanTrack.count += 1

        # 初始测量中心
        x = 0.5 * (bbox[0] + bbox[1])
        y = 0.5 * (bbox[2] + bbox[3])
        self.bbox = bbox

        # 状态向量
        self.x = np.array([x, y, 0., 0.])   # 速度未知时可置0

        # 状态协方差
        self.P = np.eye(4)

        # 模型矩阵
        self.dt = dt
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]], dtype=float)
        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]], dtype=float)

        # 噪声
        self.Q = var_process * np.eye(4)
        self.R = var_measure * np.eye(2)

        # 管控生命周期
        self.hits = 1       # 更新次数
        self.misses = 0     # 连续漏检次数

    def predict(self):
        # x' = F x
        self.x = self.F.dot(self.x)
        # P' = F P F^T + Q
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        return self.x.copy()

    def update(self, bbox):
        # bbox -> 测量 z
        x = 0.5 * (bbox[0] + bbox[1])
        y = 0.5 * (bbox[2] + bbox[3])
        z = np.array([x, y])

        # 卡尔曼增益
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        # 更新状态
        y_res = z - self.H.dot(self.x)
        self.x = self.x + K.dot(y_res)
        self.P = (np.eye(4) - K.dot(self.H)).dot(self.P)

        # 更新外界 bbox 记录
        self.bbox = bbox
        self.hits += 1
        self.misses = 0

    def get_state(self):
        """
        返回当前估计的 bbox 位置
        """
        cx, cy = self.x[0], self.x[1]
        w = self.bbox[1] - self.bbox[0] + 1
        h = self.bbox[3] - self.bbox[2] + 1
        return (cx - w/2, cx + w/2, cy - h/2, cy + h/2)

    def speed(self):
        """
        返回当前速度幅值
        """
        return np.linalg.norm(self.x[2:4])


def iou(box1, box2):
    x1, x2 = max(box1[0], box2[0]), min(box1[1], box2[1])
    y1, y2 = max(box1[2], box2[2]), min(box1[3], box2[3])
    if x2<x1 or y2<y1:
        return 0.0
    inter = (x2-x1+1)*(y2-y1+1)
    a1 = (box1[1]-box1[0]+1)*(box1[3]-box1[2]+1)
    a2 = (box2[1]-box2[0]+1)*(box2[3]-box2[2]+1)
    return inter / (a1 + a2 - inter)


class DroneTracker:
    def __init__(self,
                 max_misses=3,
                 iou_thresh=0.3,
                 speed_thresh=0.5):
        """
        speed_thresh: 只有当 track 速度超过阈值，才认为是真正的飞行物。
        """
        self.tracks = []
        self.max_misses = max_misses
        self.iou_thresh = iou_thresh
        self.speed_thresh = speed_thresh

    def update(self, detections):
        """
        detections: [(xmin,xmax,ymin,ymax), ...]
        返回当前所有“有效”track 列表
        """
        # 1. 先用卡尔曼 predict
        for tr in self.tracks:
            tr.predict()

        N = len(self.tracks)
        M = len(detections)
        if N == 0:
            # 全部建新 track
            for det in detections:
                self.tracks.append(KalmanTrack(det))
            return self._valid_tracks()

        # 2. 构造 cost 矩阵 = 1 - IOU
        cost = np.ones((N, M), dtype=float)
        for i, tr in enumerate(self.tracks):
            pred_box = tr.get_state()
            for j, det in enumerate(detections):
                cost[i, j] = 1.0 - iou(pred_box, det)

        # 3. 最优匹配
        row_idx, col_idx = linear_sum_assignment(cost)

        assigned_tr = set()
        assigned_det= set()
        # 4. 匹配并 update
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < (1.0 - self.iou_thresh):
                self.tracks[r].update(detections[c])
                assigned_tr.add(r)
                assigned_det.add(c)

        # 5. 其余 tracks 漏检一次
        for i, tr in enumerate(self.tracks):
            if i not in assigned_tr:
                tr.misses += 1

        # 6. 删除漏检过多的
        self.tracks = [tr for tr in self.tracks if tr.misses <= self.max_misses]

        # 7. 对未匹配的 detections，新建 track
        for j, det in enumerate(detections):
            if j not in assigned_det:
                self.tracks.append(KalmanTrack(det))

        return self._valid_tracks()

    def _valid_tracks(self):
        """
        只返回那些'运动速度'超过阈值的 tracks
        """
        good = []
        for tr in self.tracks:
            if tr.hits>=2 and tr.speed() >= self.speed_thresh:
                # hits>=2 保证已经更新过一次速度
                good.append(tr)
        return good
