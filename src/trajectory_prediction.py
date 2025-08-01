# src/trajectory_prediction.py
import numpy as np

class SingleDronePredictor:
    """
    单目标：记录所有历史中心点，做一次线性拟合 x=vt+b, y=vt+b，
    然后可以外推 future_frames 帧。
    """
    def __init__(self):
        self.frames = []     # [t1, t2, ...]
        self.centers = []    # [[x1,y1], [x2,y2], ...]

    @staticmethod
    def get_center(box):
        # box = (x0, x1, y0, y1)
        x0, x1, y0, y1 = box
        return np.array([ (x0+x1)/2.0, (y0+y1)/2.0 ], dtype=np.float32)

    def update(self, frame_idx, box):
        """
        本函数在每帧被检测到 box 时调用，将中心点和帧号存下来。
        """
        c = self.get_center(box)
        self.frames.append(frame_idx)
        self.centers.append(c)
        return c

    def predict(self, future_frames=1):
        """
        对已有的历史点做一阶线性拟合，预测接下来 future_frames 帧的位置，
        返回 shape=(future_frames,2) 的 numpy 数组。
        """
        N = len(self.frames)
        if N < 2:
            # 点不足时，直接复制最后一个点
            if N == 1:
                return np.tile(self.centers[-1], (future_frames,1))
            else:
                return np.zeros((future_frames,2), dtype=np.float32)

        # 构造最小二乘： x = vx*t + bx
        ts = np.array(self.frames, dtype=np.float32)
        cs = np.vstack(self.centers)  # (N,2)
        vx, bx = np.polyfit(ts, cs[:,0], 1)
        vy, by = np.polyfit(ts, cs[:,1], 1)

        t_last = ts[-1]
        fut = []
        for k in range(1, future_frames+1):
            t = t_last + k
            fut.append([ vx*t + bx, vy*t + by ])
        return np.array(fut, dtype=np.float32)
