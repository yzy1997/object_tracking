import numpy as np
from .CA_UKF import KalmanPredictorUKF as CAUKF

class CVUKF(CAUKF):
    """CV 模型 UKF（只关心位置和速度）"""
    def __init__(self, process_var=1.0, meas_var=10.0):
        super().__init__(process_var, meas_var)
        # 状态维度改成 4: [px,vx,py,vy]
        self.dim_x = 4
        self.dim_z = 2
        self.x = np.zeros((self.dim_x,1))
        self.P = np.eye(self.dim_x) * 500.0
        self.Q = np.eye(self.dim_x) * process_var
        self.R = np.eye(self.dim_z) * meas_var
        # 更新 UKF 参数
        self.lambda_ = self.alpha**2*(self.dim_x+self.kappa)-self.dim_x
        self.gamma = np.sqrt(self.dim_x+self.lambda_)
        self.Wm = np.full(2*self.dim_x+1, 0.5/(self.dim_x+self.lambda_))
        self.Wc = np.copy(self.Wm)
        self.Wm[0] = self.lambda_/(self.dim_x+self.lambda_)
        self.Wc[0] += (1 - self.alpha**2 + self.beta)

    def f(self, x, dt):
        px,vx,py,vy = x.flatten()
        return np.array([
            [px + vx*dt],
            [vx],
            [py + vy*dt],
            [vy]
        ])
    def h(self, x):
        return np.array([[x[0,0]],[x[2,0]]])


class IMMUKFTracker:
    def __init__(self, process_var=1.0, meas_var=10.0):
        # 模型列表
        self.models = [
            CVUKF(process_var, meas_var),  # 匀速
            CAUKF(process_var, meas_var)   # 匀加速
        ]
        self.mu = np.array([0.5, 0.5])  # 模型概率
        self.last_frame = None
        self.inited = False

        # 模型转移概率矩阵
        self.P_ij = np.array([[0.9, 0.1],
                              [0.1, 0.9]])

    def predict(self, dt=1.0):
        preds = []
        for m in self.models:
            m.predict(dt)
            preds.append([m.x[0,0], m.x[m.dim_x//2,0] if m.dim_x==4 else m.x[3,0]])
        # 模型概率加权融合
        px = np.sum([self.mu[i]*preds[i][0] for i in range(len(preds))])
        py = np.sum([self.mu[i]*preds[i][1] for i in range(len(preds))])
        return np.array([px,py])

    def update(self, frame_idx, box):
        z = (self.models[0].get_center(box)).flatten()

        # 如果整个 IMM 还没初始化，先初始化两个模型
        if not self.inited:
            for m in self.models:
                if hasattr(m, "dim_x") and m.dim_x == 4:
                    # CV-UKF：状态 [px,vx,py,vy]
                    m.x[0, 0] = z[0]
                    m.x[2, 0] = z[1]
                else:
                    # CA-UKF：状态 [px,vx,ax,py,vy,ay]
                    m.x[0, 0] = z[0]
                    m.x[3, 0] = z[1]
                m.last_frame = frame_idx
                m.inited = True
            self.last_frame = frame_idx
            self.inited = True
            return z

        # ======= 从这里开始是普通 update 流程 =======
        dt = frame_idx - self.last_frame
        if dt <= 0:
            dt = 1

        # 模型似然计算
        likelihoods = []
        for m in self.models:
            m.predict(dt)
            Hx = m.h(m.x)
            innov = z.reshape(-1, 1) - Hx
            S = m.R
            L = np.exp(-0.5 * innov.T @ np.linalg.inv(S) @ innov) / \
                np.sqrt((2*np.pi)**len(z) * np.linalg.det(S))
            likelihoods.append(L[0, 0])

        # IMM 模型概率更新
        self.mu = (self.P_ij.T @ self.mu) * likelihoods
        self.mu /= np.sum(self.mu)

        # 子滤波器更新
        for m in self.models:
            m.update(frame_idx, box)

        self.last_frame = frame_idx

        # 输出加权估计位置
        px = np.sum([self.mu[i] * (self.models[i].x[0, 0]) 
                        for i in range(len(self.models))])
        py = np.sum([self.mu[i] * (self.models[i].x[m.dim_x // 2, 0] 
                        if self.models[i].dim_x == 4 else self.models[i].x[3, 0]) 
                        for i in range(len(self.models))])
        return np.array([px, py])


    def predict_future(self, future_frames=1):
        preds = []
        for _ in range(future_frames):
            preds.append(self.predict(dt=1.0))
        return np.array(preds)
