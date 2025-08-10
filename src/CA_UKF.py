import numpy as np

class UKFPredictor:
    """
    常加速（Constant Acceleration）模型的 UKF 实现
    状态 x = [px, vx, ax, py, vy, ay], 测量 z = [px, py]
    """

    @staticmethod
    def get_center(box):
        x0, x1, y0, y1 = box
        return np.array([ (x0 + x1) / 2.0,
                          (y0 + y1) / 2.0 ], dtype=np.float32)

    def __init__(self,
                 process_var=1.0,    # 过程噪声强度 q
                 meas_var=10.0,      # 测量噪声方差
                 alpha=1e-3,
                 beta=2,
                 kappa=0):
        self.dim_x = 6
        self.dim_z = 2
        self.process_var = process_var
        self.R = np.eye(self.dim_z, dtype=np.float32) * meas_var

        # UKF 参数
        self.alpha = alpha
        self.beta  = beta
        self.kappa = kappa
        self.lambda_ = alpha**2 * (self.dim_x + kappa) - self.dim_x
        self.c = self.dim_x + self.lambda_

        # weights
        self.Wm = np.zeros(2*self.dim_x+1, dtype=np.float32)
        self.Wc = np.zeros(2*self.dim_x+1, dtype=np.float32)
        self.Wm[0] = self.lambda_ / self.c
        self.Wc[0] = self.lambda_ / self.c + (1 - alpha**2 + beta)
        self.Wm[1:] = 1.0 / (2*self.c)
        self.Wc[1:] = 1.0 / (2*self.c)

        # 初始状态与协方差
        self.x = np.zeros(self.dim_x, dtype=np.float32)
        self.P = np.eye(self.dim_x, dtype=np.float32) * 500.0

        self.inited = False
        self.last_frame = None

    def fx(self, x, dt):
        # 状态转移 f(x)
        px, vx, ax, py, vy, ay = x
        return np.array([
            px + vx*dt + 0.5*ax*dt**2,
            vx + ax*dt,
            ax,
            py + vy*dt + 0.5*ay*dt**2,
            vy + ay*dt,
            ay
        ], dtype=np.float32)

    def hx(self, x):
        # 观测函数，只观测位置
        return np.array([ x[0], x[3] ], dtype=np.float32)

    def get_Q(self, dt):
        # 与 CA_EKF 相同的 Q 计算
        t2 = dt**2 / 2.0
        t3 = dt**3 / 6.0
        t4 = dt**4 / 8.0
        t5 = dt**5 / 20.0
        Q3 = np.array([[t5, t4, t3],
                       [t4, t3, t2],
                       [t3, t2, dt]], dtype=np.float32)
        Q = np.zeros((6,6), dtype=np.float32)
        Q[0:3,0:3] = Q3
        Q[3:6,3:6] = Q3
        return Q * self.process_var

    def sigma_points(self):
        """
        生成 2n+1 个 sigma 点
        """
        X = np.zeros((2*self.dim_x+1, self.dim_x), dtype=np.float32)
        X[0] = self.x.copy()
        # c*P 的 Cholesky
        S = np.linalg.cholesky(self.c * self.P)
        for i in range(self.dim_x):
            X[i+1    ] = self.x + S[:, i]
            X[i+1+self.dim_x] = self.x - S[:, i]
        return X

    def predict_one(self, dt):
        # 1) 生成 sigma 点
        X = self.sigma_points()
        # 2) 通过 fx 传播
        X_pred = np.array([ self.fx(X[i], dt) for i in range(X.shape[0]) ],
                          dtype=np.float32)
        # 3) 计算预测均值
        x_pred = np.sum(self.Wm[:,None] * X_pred, axis=0)
        # 4) 计算预测协方差
        P_pred = np.zeros((self.dim_x, self.dim_x), dtype=np.float32)
        for i in range(2*self.dim_x+1):
            d = (X_pred[i] - x_pred).reshape(-1,1)
            P_pred += self.Wc[i] * (d @ d.T)
        P_pred += self.get_Q(dt)

        self.x = x_pred
        self.P = P_pred

    def predict(self, future_frames=1):
        if not self.inited:
            return np.zeros((future_frames, self.dim_z), dtype=np.float32)
        # 保存现场
        x0, P0, last0 = self.x.copy(), self.P.copy(), self.last_frame
        preds = []
        for _ in range(future_frames):
            self.predict_one(dt=1.0)
            preds.append(self.x[[0,3]].copy())
        # 恢复
        self.x, self.P, self.last_frame = x0, P0, last0
        return np.array(preds, dtype=np.float32)

    def init_filter(self, frame_idx, box):
        z = self.get_center(box)
        self.x[0], self.x[3] = z[0], z[1]
        # vx,ax,vy,ay 留 0
        self.last_frame = frame_idx
        self.inited = True

    def update(self, frame_idx, box):
        # 如果还没初始化，就先 init
        if not self.inited:
            self.init_filter(frame_idx, box)
            return np.array([self.x[0], self.x[3]], dtype=np.float32)

        dt = frame_idx - self.last_frame
        if dt > 0:
            self.predict_one(dt)

        # UKF 量测更新
        # 1) 生成 sigma 点
        X = self.sigma_points()
        # 2) 通过 hx 映射到测量空间
        Z = np.array([ self.hx(X[i]) for i in range(X.shape[0]) ],
                     dtype=np.float32)
        # 3) 计算测量预测均值
        z_pred = np.sum(self.Wm[:,None] * Z, axis=0)
        # 4) 计算 P_zz, P_xz
        P_zz = np.zeros((self.dim_z, self.dim_z), dtype=np.float32)
        P_xz = np.zeros((self.dim_x, self.dim_z), dtype=np.float32)
        for i in range(2*self.dim_x+1):
            dz = (Z[i] - z_pred).reshape(-1,1)
            dx = (X[i] - self.x).reshape(-1,1)
            P_zz += self.Wc[i] * (dz @ dz.T)
            P_xz += self.Wc[i] * (dx @ dz.T)
        P_zz += self.R

        # 5) 卡尔曼增益 K
        K = P_xz @ np.linalg.inv(P_zz)
        z = self.get_center(box)
        # 6) 更新状态与协方差
        self.x = self.x + (K @ (z - z_pred)).flatten()
        self.P = self.P - K @ P_zz @ K.T

        self.last_frame = frame_idx
        return np.array([self.x[0], self.x[3]], dtype=np.float32)
