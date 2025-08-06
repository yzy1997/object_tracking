# UKF.py

import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

class KalmanPredictor:
    """
    单目标匀速运动的无迹卡尔曼滤波预测器，接口与原来的 KalmanPredictor 一致：
      - update(frame_idx, box)  返回去噪后的当前中心 (cx, cy)
      - predict(future_frames)   在当前状态基础上外推若干帧
    """

    @staticmethod
    def get_center(box):
        """
        box = [x0, x1, y0, y1]
        返回中心坐标 [cx, cy]
        """
        x0, x1, y0, y1 = box
        return np.array([ (x0 + x1) / 2.0, (y0 + y1) / 2.0 ], dtype=np.float32)

    def __init__(self,
                 process_var=1.0,    # 过程噪声方差
                 meas_var=10.0,      # 观测噪声方差
                 dt=1.0              # 默认帧间隔
                ):
        self.dim_x = 4
        self.dim_z = 2
        self.dt0   = dt
        self.process_var = process_var
        self.meas_var   = meas_var

        # 1) 定义状态转移函数 fx: x = [px, vx, py, vy]
        def fx(x, dt):
            px, vx, py, vy = x
            return np.array([
                px + vx * dt,
                vx,
                py + vy * dt,
                vy
            ], dtype=np.float32)

        # 2) 定义观测函数 hx: 只返回 px, py
        def hx(x):
            return np.array([ x[0], x[2] ], dtype=np.float32)

        # 3) 生成 MerweScaledSigmaPoints
        points = MerweScaledSigmaPoints(n=self.dim_x,
                                        alpha=0.1,
                                        beta=2.0,
                                        kappa=0.0)

        # 4) 构造 UKF
        self.ukf = UnscentedKalmanFilter(
            dim_x=self.dim_x,
            dim_z=self.dim_z,
            dt=self.dt0,
            fx=fx,
            hx=hx,
            points=points
        )

        # 5) 设置过程噪声 Q（匀速模型离散化）
        q  = process_var
        dt = self.dt0
        Q = np.array([
            [q*dt**3/3., q*dt**2/2.,      0.,          0.],
            [q*dt**2/2., q*dt,            0.,          0.],
            [0.,          0.,         q*dt**3/3.,  q*dt**2/2.],
            [0.,          0.,         q*dt**2/2.,   q*dt   ]
        ], dtype=np.float32)
        self.ukf.Q = Q

        # 6) 设置测量噪声 R
        self.ukf.R = np.eye(self.dim_z, dtype=np.float32) * meas_var

        # 7) 初始协方差 P（设大一点）
        self.ukf.P *= 500.0

        # 状态是否已初始化
        self.inited     = False
        # 上一次更新的帧号
        self.last_frame = None

    def init_filter(self, frame_idx, box):
        """
        第一次调用时用第一帧检测初始化状态。
        """
        z = self.get_center(box)
        # 位置取测量值，速度初始为 0
        self.ukf.x = np.array([ z[0], 0.0, z[1], 0.0 ], dtype=np.float32)
        # 保持初始 P
        self.ukf.P *= 1.0
        self.last_frame = frame_idx
        self.inited     = True

    def update(self, frame_idx, box):
        """
        收到新观测后先 predict 再 update，
        返回去噪后的中心 [cx, cy]
        """
        if not self.inited:
            self.init_filter(frame_idx, box)
            return np.array([ self.ukf.x[0], self.ukf.x[2] ], dtype=np.float32)

        # 1) 多步预测到当前帧 (帧号可能不连续)
        dt = frame_idx - self.last_frame
        if dt > 0:
            self.ukf.predict(dt=dt)

        # 2) UKF 更新
        z = self.get_center(box)
        self.ukf.update(z)

        # 3) 更新帧号
        self.last_frame = frame_idx

        return np.array([ self.ukf.x[0], self.ukf.x[2] ], dtype=np.float32)

    def predict(self, future_frames=1):
        """
        在当前状态基础上等间隔 dt=1 外推若干步，
        不修改原来的 ukf.x, ukf.P, last_frame，
        返回 shape=(future_frames, 2)
        """
        if not self.inited:
            return np.zeros((future_frames, 2), dtype=np.float32)

        # 备份当前状态
        x0 = self.ukf.x.copy()
        P0 = self.ukf.P.copy()
        last = self.last_frame

        preds = []
        for k in range(1, future_frames + 1):
            self.ukf.predict(dt=1.0)
            preds.append([ self.ukf.x[0], self.ukf.x[2] ])
            self.last_frame = last + k

        # 恢复原始状态
        self.ukf.x      = x0
        self.ukf.P      = P0
        self.last_frame = last

        return np.array(preds, dtype=np.float32)
