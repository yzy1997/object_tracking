# IMM_EKF.py
import numpy as np
from collections import deque
from typing import Dict, List, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam


class BaseEKF:
    """EKF基类（TensorFlow兼容版）"""
    def __init__(self, dim_x=4, dim_z=2):
        self.dim_x = dim_x  # 状态维度 [x, y, vx, vy]
        self.dim_z = dim_z  # 观测维度 [x, y]
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x) * 0.1  # 过程噪声
        self.R = np.eye(dim_z) * 0.5  # 观测噪声
        self.H = np.zeros((dim_z, dim_x))
        self.H[:, :2] = np.eye(dim_z)

    def predict(self, dt=0.1):
        """预测步骤（需子类实现）"""
        raise NotImplementedError

    def update(self, z):
        """更新步骤"""
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P
        return self.x.copy()

class CV_EKF(BaseEKF):
    """匀速模型（Constant Velocity）"""
    def predict(self, dt=0.1):
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

class CA_EKF(BaseEKF):
    """匀加速模型（Constant Acceleration）"""
    def __init__(self, **kwargs):
        super().__init__(dim_x=6)  # [x, y, vx, vy, ax, ay]
        self.x = np.zeros(6)
        self.P = np.eye(6)
        self.Q = np.eye(6) * 0.3
        self.H = np.zeros((2, 6))
        self.H[:, :2] = np.eye(2)

    def predict(self, dt=0.1):
        self.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4]  # 返回与CV模型对齐的维度

class TrajectoryLSTM(tf.keras.Model):
    """基于TensorFlow的LSTM校正器"""
    def __init__(self, seq_length=10, feat_dim=2):
        super().__init__()
        self.lstm = LSTM(32, input_shape=(seq_length, feat_dim))
        self.dense = Dense(2)  # 输出位置修正量
        
    def call(self, inputs):
        x = self.lstm(inputs)
        return self.dense(x)

class IMM_EKF:
    """交互多模型EKF（TensorFlow实现版）"""
    def __init__(self, dt=0.1, process_noise=0.1, obs_noise=0.5):
        self.models = {
            'CV': CV_EKF(),
            'CA': CA_EKF()
        }
        # 设置过程噪声和观测噪声
        for model in self.models.values():
            if isinstance(model, CV_EKF):
                model.Q = np.eye(4) * process_noise
                model.R = np.eye(2) * obs_noise
            elif isinstance(model, CA_EKF):
                model.Q = np.eye(6) * process_noise
                model.R = np.eye(2) * obs_noise
        
        self.model_probs = {'CV': 0.7, 'CA': 0.3}
        self.dt = dt
        self.history = deque(maxlen=20)
        
        # 初始化TensorFlow LSTM
        self.lstm = TrajectoryLSTM()
        self.lstm.compile(optimizer=Adam(0.001), loss='mse')
        
        # 尝试加载预训练权重
        try:
            self.lstm.load_weights('lstm_corrector.h5')
            print("LSTM校正器权重加载成功")
        except:
            print("未找到预训练权重，将使用随机初始化")


    def update(self, z):
        """更新观测"""
        self.history.append(z)
        
        # 各模型独立更新
        for model in self.models.values():
            model.update(z)
            
        # 动态调整模型概率（基于预测误差）
        errors = {}
        for name, model in self.models.items():
            pred = model.H @ model.x
            errors[name] = np.linalg.norm(z - pred)
        
        total_error = sum(errors.values())
        for name in self.models:
            self.model_probs[name] = 0.9 * self.model_probs[name] + \
                                   0.1 * (1 - errors[name]/total_error)

    def predict(self, steps=5):
        """带LSTM校正的预测"""
        # 多模型预测
        predictions = {}
        for name, model in self.models.items():
            x_backup, P_backup = model.x.copy(), model.P.copy()
            
            pred_traj = []
            for _ in range(steps):
                pred = model.predict(self.dt)
                pred_traj.append(pred[:2])
            
            predictions[name] = np.mean(pred_traj[-3:], axis=0)
            model.x, model.P = x_backup, P_backup
        
        # 模型融合
        combined = np.zeros(2)
        for name, pred in predictions.items():
            combined += pred * self.model_probs[name]
        
        # LSTM校正（当有足够历史数据时）
        if len(self.history) >= 10:
            hist_array = np.array(self.history)[-10:]  # 取最近10个点
            hist_tensor = tf.expand_dims(hist_array, axis=0)  # 形状 (1, 10, 2)
            lstm_corr = self.lstm(hist_tensor).numpy()[0]
            combined += lstm_corr * 0.2  # 加权修正
        
        return combined, self.model_probs.copy()

    def train_lstm(self, X_train, y_train, epochs=50):
        """在线训练LSTM校正器"""
        if len(X_train) < 10:
            print("训练数据不足，至少需要10个样本")
            return
            
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
        
        self.lstm.fit(dataset, epochs=epochs, verbose=1)
        self.lstm.save_weights('lstm_corrector.h5')
        print("LSTM校正器训练完成")
