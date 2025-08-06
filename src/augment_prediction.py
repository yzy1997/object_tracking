import numpy as np

class KalmanPredictor:
    """
    单目标匀速运动卡尔曼滤波器预测器。
    状态向量 x = [px, vx, py, vy]^T，
    观测 z = [px, py]^T。
    """
    def __init__(self,
                 process_var=1.0,      # 过程噪声方差
                 meas_var=10.0):       # 观测噪声方差
        # 是否已初始化
        self.inited = False
        # 状态维度、测量维度
        self.dim_x = 4
        self.dim_z = 2

        # 状态转移矩阵 F, 后面会根据 dt 更新
        self.F = np.eye(self.dim_x)
        # 观测矩阵 H
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[0,0] = 1
        self.H[1,2] = 1

        # 过程噪声协方差 Q (根据 dt 动态更新)
        self.process_var = process_var
        # 测量噪声协方差 R
        self.R = np.eye(self.dim_z) * meas_var

        # 状态协方差 P
        self.P = np.eye(self.dim_x) * 500.0

        # 当前状态 x
        self.x = np.zeros((self.dim_x,1), dtype=np.float32)

        # 上一次更新的帧号
        self.last_frame = None

    @staticmethod
    def get_center(box):
        x0,x1,y0,y1 = box
        return np.array([[ (x0+x1)/2.0 ], [ (y0+y1)/2.0 ]], dtype=np.float32)

    def init_filter(self, frame_idx, box):
        """
        第一次调用时用第一帧检测初始化状态。
        """
        z = self.get_center(box)
        # 初始化位置，用观测值；速度初始化为 0
        self.x = np.array([[z[0,0]], [0.0], [z[1,0]], [0.0]], dtype=np.float32)
        self.P = np.eye(self.dim_x) * 500.0
        self.last_frame = frame_idx
        self.inited = True

    def predict_one(self, dt):
        """
        卡尔曼一次预测。根据 dt 更新 F, Q，然后做 x = F x, P = F P F^T + Q
        """
        # 状态转移 F
        self.F[0,1] = dt
        self.F[2,3] = dt
        # 过程噪声 Q（简化为常数模型）
        q = self.process_var
        # 简单构造 Q，位置和速度上的噪声
        Q = np.array([
            [q*dt**3/3, q*dt**2/2,       0,         0],
            [q*dt**2/2,    q*dt,         0,         0],
            [0,            0,      q*dt**3/3, q*dt**2/2],
            [0,            0,      q*dt**2/2,    q*dt]
        ], dtype=np.float32)
        # 预测
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + Q

    def update(self, frame_idx, box):
        """
        有新观测时调用。完成从上次帧到当前帧的多步 predict + 一次 update。
        返回去噪后的当前中心点 [cx, cy]。
        """
        if not self.inited:
            self.init_filter(frame_idx, box)
            return np.array([self.x[0,0], self.x[2,0]], dtype=np.float32)

        # 1) 先做多步预测到当前帧
        dt = frame_idx - self.last_frame
        if dt > 0:
            self.predict_one(dt)

        # 2) 卡尔曼更新
        z = self.get_center(box)         # (2,1)
        y = z - self.H.dot(self.x)       # 创新
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        self.x = self.x + K.dot(y)
        self.P = (np.eye(self.dim_x) - K.dot(self.H)).dot(self.P)

        # 3) 更新 last_frame
        self.last_frame = frame_idx
        # 返回估计后的位置
        return np.array([self.x[0,0], self.x[2,0]], dtype=np.float32)

    def predict(self, future_frames=1):
        """
        在当前状态基础上外推 future_frames 步，返回 shape=(future_frames,2) 的坐标数组。
        不修改原始状态与协方差。
        """
        if not self.inited:
            return np.zeros((future_frames,2), dtype=np.float32)

        # 拷贝当前 x,P
        x0 = self.x.copy()
        P0 = self.P.copy()
        last = self.last_frame

        preds = []
        for k in range(1, future_frames+1):
            self.predict_one(dt=1)   # 假设等间隔 1 帧
            preds.append([self.x[0,0], self.x[2,0]])
            self.last_frame = last + k

        # 恢复原始状态
        self.x = x0
        self.P = P0
        self.last_frame = last

        return np.array(preds, dtype=np.float32)
