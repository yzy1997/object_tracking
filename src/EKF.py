import numpy as np

class KalmanPredictor:
    """
    单目标匀速运动的扩展卡尔曼滤波预测器（EKF）。
    状态向量 x = [px, vx, py, vy]^T，
    观测 z = [px, py]^T。
    接口同原 KalmanPredictor：
      - update(frame_idx, box)  返回去噪后的当前中心 (cx, cy)
      - predict(future_frames)   在当前状态基础上外推若干帧
    """

    @staticmethod
    def get_center(box):
        """
        box = [x0, x1, y0, y1]
        返回中心 [cx, cy]
        """
        x0, x1, y0, y1 = box
        return np.array([[ (x0 + x1)/2.0 ],
                         [ (y0 + y1)/2.0 ]], dtype=np.float32)

    def __init__(self,
                 process_var=1.0,   # 过程噪声方差
                 meas_var=10.0):    # 观测噪声方差
        # 状态维度、观测维度
        self.dim_x = 4
        self.dim_z = 2

        # 过程噪声方差、测量噪声协方差
        self.process_var = process_var
        self.R = np.eye(self.dim_z, dtype=np.float32) * meas_var

        # 初始 P、初始 x
        self.P = np.eye(self.dim_x, dtype=np.float32) * 500.0
        self.x = np.zeros((self.dim_x,1), dtype=np.float32)

        # 是否已初始化
        self.inited = False
        # 记录上次 update 的帧号
        self.last_frame = None

    def fx(self, x, dt):
        """
        非线性状态转移函数（匀速模型）：
          px' = px + vx*dt
          vx' = vx
          py' = py + vy*dt
          vy' = vy
        x: (4,1) 列向量
        返回 x_pred: (4,1)
        """
        px = x[0,0]; vx = x[1,0]
        py = x[2,0]; vy = x[3,0]
        return np.array([
            [px + vx*dt],
            [vx],
            [py + vy*dt],
            [vy]
        ], dtype=np.float32)

    def jac_F(self, dt):
        """
        状态转移函数关于 x 的雅可比矩阵 F (4x4)
        对匀速模型：
          F = [[1, dt, 0,  0],
               [0,  1, 0,  0],
               [0,  0, 1, dt],
               [0,  0, 0,  1]]
        """
        F = np.eye(self.dim_x, dtype=np.float32)
        F[0,1] = dt
        F[2,3] = dt
        return F

    def hx(self, x):
        """
        观测函数，只测 px, py
        x: (4,1)
        返回 z_pred: (2,1)
        """
        return np.array([
            [ x[0,0] ],
            [ x[2,0] ]
        ], dtype=np.float32)

    def jac_H(self):
        """
        观测函数关于 x 的雅可比 H (2x4)
          H = [[1, 0, 0, 0],
               [0, 0, 1, 0]]
        """
        H = np.zeros((self.dim_z, self.dim_x), dtype=np.float32)
        H[0,0] = 1.0
        H[1,2] = 1.0
        return H

    def init_filter(self, frame_idx, box):
        """
        第一次调用时初始化状态：位置用观测，速度初始化为 0
        """
        z = self.get_center(box)      # (2,1)
        self.x = np.array([
            [z[0,0]],
            [0.0],
            [z[1,0]],
            [0.0]
        ], dtype=np.float32)
        self.P = np.eye(self.dim_x, dtype=np.float32) * 500.0
        self.last_frame = frame_idx
        self.inited = True

    def predict_one(self, dt):
        """
        执行一次 EKF 预测：
          x = fx(x)
          P = F P F^T + Q
        其中 F=jac_F(dt)，Q 根据匀速模型离散化构造
        """
        F = self.jac_F(dt)
        # 构造 Q
        q = self.process_var
        Q = np.array([
            [q*dt**3/3, q*dt**2/2,      0,          0],
            [q*dt**2/2,    q*dt,        0,          0],
            [0,            0,      q*dt**3/3, q*dt**2/2],
            [0,            0,      q*dt**2/2,    q*dt]
        ], dtype=np.float32)

        # 状态预测
        self.x = self.fx(self.x, dt)
        # 协方差预测
        self.P = F.dot(self.P).dot(F.T) + Q

    def update(self, frame_idx, box):
        """
        收到新观测后，先多步 predict 再做一次 EKF 更新，返回当前去噪位置 [cx, cy]
        """
        if not self.inited:
            self.init_filter(frame_idx, box)
            return np.array([self.x[0,0], self.x[2,0]], dtype=np.float32)

        # 1) 多步预测到当前帧
        dt = frame_idx - self.last_frame
        if dt > 0:
            self.predict_one(dt)

        # 2) 计算雅可比 H
        H = self.jac_H()

        # 3) EKF 更新
        z = self.get_center(box)       # (2,1)
        z_pred = self.hx(self.x)       # (2,1)
        y = z - z_pred                 # 创新
        S = H.dot(self.P).dot(H.T) + self.R
        K = self.P.dot(H.T).dot(np.linalg.inv(S))

        self.x = self.x + K.dot(y)
        I = np.eye(self.dim_x, dtype=np.float32)
        self.P = (I - K.dot(H)).dot(self.P)

        # 4) 保存帧号
        self.last_frame = frame_idx

        return np.array([self.x[0,0], self.x[2,0]], dtype=np.float32)

    def predict(self, future_frames=1):
        """
        在当前状态基础上等间隔 1 帧外推若干步，不修改原状态，返回 shape=(future_frames,2)
        """
        if not self.inited:
            return np.zeros((future_frames, 2), dtype=np.float32)

        # 备份
        x0 = self.x.copy()
        P0 = self.P.copy()
        last = self.last_frame

        preds = []
        for k in range(1, future_frames+1):
            self.predict_one(dt=1.0)
            preds.append([ self.x[0,0], self.x[2,0] ])
            self.last_frame = last + k

        # 恢复
        self.x = x0
        self.P = P0
        self.last_frame = last

        return np.array(preds, dtype=np.float32)
