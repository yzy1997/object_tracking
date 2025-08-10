# src/CA_EKF.py
import numpy as np

class KalmanPredictor:
    """
    单目标匀加速运动的扩展卡尔曼滤波预测器（CA-EKF）。
    状态向量 x = [px, vx, ax, py, vy, ay]^T
    观测 z = [px, py]^T

    改进点：
    - 使用前两帧初始化速度（若可用）
    - Mahalanobis gating 拒绝异常观测
    - 提供 mark_missed(frame_idx) 以在漏检时推进状态
    """

    @staticmethod
    def get_center(box):
        x0, x1, y0, y1 = box
        return np.array([[(x0 + x1) / 2.0], [(y0 + y1) / 2.0]], dtype=np.float32)

    def __init__(self, process_var=2.0, meas_var=20.0, gate_thresh=9.0):
        self.dim_x, self.dim_z = 6, 2
        self.process_var = process_var
        self.R = np.eye(self.dim_z, dtype=np.float32) * meas_var
        # 初值 P 设较大（不确定性高），但对速度/加速度给更大不确定度
        self.P = np.diag([500.0, 500.0, 1000.0, 500.0, 500.0, 1000.0]).astype(np.float32)
        self.x = np.zeros((self.dim_x, 1), dtype=np.float32)
        self.inited = False
        self.last_frame = None

        # 用于两帧初始化速度的临时缓存
        self._tmp_prev_z = None
        self._tmp_prev_frame = None

        # gating 阈值（Mahalanobis 距离平方）
        self.gate_thresh = gate_thresh

    def fx(self, x, dt):
        px, vx, ax, py, vy, ay = x.flatten()
        return np.array([
            [px + vx * dt + 0.5 * ax * dt**2],
            [vx + ax * dt],
            [ax],
            [py + vy * dt + 0.5 * ay * dt**2],
            [vy + ay * dt],
            [ay]
        ], dtype=np.float32)

    def jac_F(self, dt):
        F = np.eye(self.dim_x, dtype=np.float32)
        F[0, 1] = F[3, 4] = dt
        F[1, 2] = F[4, 5] = dt
        F[0, 2] = F[3, 5] = 0.5 * dt**2
        return F

    def hx(self, x):
        return np.array([[x[0, 0]], [x[3, 0]]], dtype=np.float32)

    def jac_H(self):
        H = np.zeros((self.dim_z, self.dim_x), dtype=np.float32)
        H[0, 0] = 1.0
        H[1, 3] = 1.0
        return H

    def get_Q(self, dt):
        # 标准连续白噪声加速度模型的 Q（块对角）
        q = self.process_var
        t5 = dt**5 / 20.0
        t4 = dt**4 / 8.0
        t3 = dt**3 / 6.0
        t2 = dt**2 / 2.0
        Q_template = np.array([[t5, t4, t3], [t4, t3, t2], [t3, t2, dt]], dtype=np.float32)
        Q = np.zeros((self.dim_x, self.dim_x), dtype=np.float32)
        Q[0:3, 0:3] = Q_template
        Q[3:6, 3:6] = Q_template
        return Q * q

    def init_from_two_frames(self, frame_idx, box):
        """
        若已有之前一帧数据，则用两帧估计初速度并初始化。
        否则缓存当前观测作为第一帧，等待下一次来初始化两帧。
        """
        z = self.get_center(box)
        if self._tmp_prev_z is None:
            # 缓存第一帧
            self._tmp_prev_z = z.copy()
            self._tmp_prev_frame = frame_idx
            # 不真正初始化滤波器（等待第二帧以估计速度）
            self.last_frame = frame_idx
            return np.array([z[0,0], z[1,0]], dtype=np.float32)
        else:
            dt = frame_idx - self._tmp_prev_frame
            if dt <= 0:
                dt = 1.0
            vx = (z[0,0] - self._tmp_prev_z[0,0]) / dt
            vy = (z[1,0] - self._tmp_prev_z[1,0]) / dt
            # 用速度初始化，令加速度为 0
            self.x = np.zeros((self.dim_x,1), dtype=np.float32)
            self.x[0,0] = z[0,0]
            self.x[1,0] = vx
            self.x[2,0] = 0.0
            self.x[3,0] = z[1,0]
            self.x[4,0] = vy
            self.x[5,0] = 0.0
            # 重新设置 P：位置中等不确定，速度较高不确定，加速度不确定更高
            self.P = np.diag([25.0, 100.0, 500.0, 25.0, 100.0, 500.0]).astype(np.float32)
            self.last_frame = frame_idx
            self.inited = True
            # 清空临时缓存
            self._tmp_prev_z = None
            self._tmp_prev_frame = None
            return np.array([self.x[0,0], self.x[3,0]], dtype=np.float32)

    def predict_one(self, dt):
        if dt <= 0:
            return
        F = self.jac_F(dt)
        Q = self.get_Q(dt)
        self.x = self.fx(self.x, dt)
        self.P = F @ self.P @ F.T + Q

    def mark_missed(self, frame_idx):
        """
        当这一帧没有检测到目标时，调用该函数推进滤波器内部状态（predict）。
        这样下一次检测时 dt 计算会更合理。
        """
        if not self.inited:
            # 若尚未初始化，只更新 last_frame 和临时缓存时间（不改变 x）
            if self._tmp_prev_frame is not None:
                # 不改变 _tmp_prev_z，但更新帧号以保持 dt 正确性
                # 这里不做其他事
                self._tmp_prev_frame = frame_idx
            self.last_frame = frame_idx
            return np.array([self.x[0,0], self.x[3,0]], dtype=np.float32)

        dt = frame_idx - self.last_frame
        if dt > 0:
            self.predict_one(dt)
            self.last_frame = frame_idx
        return np.array([self.x[0,0], self.x[3,0]], dtype=np.float32)

    def update(self, frame_idx, box):
        """
        更新：若未初始化，则尝试用两帧初始化速度；若已初始化则按 CA-EKF 正常更新。
        添加 Mahalanobis gating，若观测异常则跳过更新（当作漏检）。
        """
        z = self.get_center(box)

        # 若未初始化，尝试两帧初始化
        if not self.inited:
            return self.init_from_two_frames(frame_idx, box)

        # 预测到当前时间
        dt = frame_idx - self.last_frame
        if dt > 0:
            self.predict_one(dt)

        H = self.jac_H()
        z_pred = self.hx(self.x)
        y = z - z_pred
        S = H @ self.P @ H.T + self.R

        # Mahalanobis 距离（基于观测维度 2）
        try:
            S_inv = np.linalg.inv(S)
            d2 = float((y.T @ S_inv @ y).squeeze())
        except np.linalg.LinAlgError:
            d2 = np.inf

        if d2 > self.gate_thresh:
            # 观测太离谱，视为虚警/异常，跳过更新（仅更新 last_frame）
            # 增大不确定度以表示未被观测约束（可选）
            self.P += np.eye(self.dim_x, dtype=np.float32) * 10.0
            self.last_frame = frame_idx
            # 返回当前预测位置
            return np.array([self.x[0,0], self.x[3,0]], dtype=np.float32)

        # 卡尔曼更新
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.dim_x, dtype=np.float32)
        self.P = (I - K @ H) @ self.P

        self.last_frame = frame_idx
        return np.array([self.x[0,0], self.x[3,0]], dtype=np.float32)

    def predict(self, future_frames=1):
        """
        外推 future_frames 帧，返回 future_frames x 2 的位置数组（不改变内部状态）。
        """
        if not self.inited:
            return np.zeros((future_frames, 2), dtype=np.float32)

        x0, P0, last = self.x.copy(), self.P.copy(), self.last_frame
        preds = []
        for _ in range(future_frames):
            # 固定 dt=1 做步进外推（可改为其它）
            self.predict_one(dt=1.0)
            preds.append([self.x[0,0], self.x[3,0]])
        # 恢复内部状态
        self.x, self.P, self.last_frame = x0, P0, last
        return np.array(preds, dtype=np.float32)
