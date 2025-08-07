# src/CA_EKF.py
import numpy as np

class KalmanPredictor:
    """
    单目标匀加速运动的扩展卡尔曼滤波预测器（CA-EKF）。
    状态向量 x = [px, vx, ax, py, vy, ay]^T
    观测 z = [px, py]^T
    """

    @staticmethod
    def get_center(box):
        x0, x1, y0, y1 = box
        return np.array([[(x0 + x1) / 2.0], [(y0 + y1) / 2.0]], dtype=np.float32)

    def __init__(self, process_var=1.0, meas_var=10.0):
        self.dim_x, self.dim_z = 6, 2
        self.process_var = process_var
        self.R = np.eye(self.dim_z, dtype=np.float32) * meas_var
        self.P = np.eye(self.dim_x, dtype=np.float32) * 500.0
        self.x = np.zeros((self.dim_x, 1), dtype=np.float32)
        self.inited = False
        self.last_frame = None

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
        q = self.process_var
        t5, t4, t3, t2 = dt**5/20, dt**4/8, dt**3/6, dt**2/2
        Q_template = np.array([[t5, t4, t3], [t4, t3, t2], [t3, t2, dt]], dtype=np.float32)
        Q = np.zeros((self.dim_x, self.dim_x), dtype=np.float32)
        Q[0:3, 0:3] = Q_template
        Q[3:6, 3:6] = Q_template
        return Q * q
    
    def init_filter(self, frame_idx, box):
        z = self.get_center(box)
        self.x[0,0], self.x[3,0] = z[0,0], z[1,0] # 位置用观测初始化，速/加-速度为0
        self.last_frame = frame_idx
        self.inited = True
        
    def predict_one(self, dt):
        F = self.jac_F(dt)
        Q = self.get_Q(dt)
        self.x = self.fx(self.x, dt)
        self.P = F @ self.P @ F.T + Q

    def update(self, frame_idx, box):
        if not self.inited:
            self.init_filter(frame_idx, box)
            return np.array([self.x[0,0], self.x[3,0]], dtype=np.float32)

        dt = frame_idx - self.last_frame
        if dt > 0:
            self.predict_one(dt)

        H = self.jac_H()
        z = self.get_center(box)
        z_pred = self.hx(self.x)
        y = z - z_pred
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.dim_x, dtype=np.float32)
        self.P = (I - K @ H) @ self.P
        self.last_frame = frame_idx
        return np.array([self.x[0,0], self.x[3,0]], dtype=np.float32)

    def predict(self, future_frames=1):
        if not self.inited: return np.zeros((future_frames, 2), dtype=np.float32)
        x0, P0, last = self.x.copy(), self.P.copy(), self.last_frame
        preds = []
        for _ in range(future_frames):
            self.predict_one(dt=1.0)
            preds.append([self.x[0,0], self.x[3,0]])
        self.x, self.P, self.last_frame = x0, P0, last
        return np.array(preds, dtype=np.float32)
