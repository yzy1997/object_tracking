# denoiser.py
# -*- coding: utf-8 -*-
"""
雷达去噪工具（稀疏增量友好 + 自适应阈值 + 邻域一致性）

用法 1（对“增量点”做去噪）：
    denoiser = RadarDenoiser(kernel_size=3, min_neighbors=2)
    clean_points = denoiser.filter_points(points, shape_hw=(H, W))

用法 2（对整幅雷达图做去噪）：
    denoiser = RadarDenoiser(filter_size=3, thr_percentile=95, rel_floor=0.12)
    denoised, mask = denoiser.denoise_image(radar_image, return_mask=True)
"""

import numpy as np
from scipy import ndimage

# -----------------------------
# 工具函数
# -----------------------------
def _ensure_float(img):
    if img is None:
        return None
    if img.dtype != np.float32 and img.dtype != np.float64:
        return img.astype(np.float32)
    return img


# -----------------------------
# 核心类
# -----------------------------
class RadarDenoiser:
    def __init__(self,
                 # —— 图像域去噪 —— #
                 filter_size: int = 3,        # 中值滤波核
                 distance_threshold: float = None,  # 绝对阈值（若为 None 则用分位阈值）
                 thr_percentile: float = 95.0,      # 分位阈值（仅在非零像素上）
                 rel_floor: float = 0.12,           # 相对地板：至少是 max 的一定比例
                 morph_open: int = 1,               # 开运算迭代次数（0 表示不用）
                 morph_close: int = 1,              # 闭运算迭代次数

                 # —— 稀疏点域去噪 —— #
                 kernel_size: int = 3,        # 邻域大小（奇数）
                 min_neighbors: int = 2,      # 邻域命中数阈值（含自身）
                 clip_percentile: float = 99.5 # 对点的幅值进行 winsorize 上裁剪
                 ):
        # 图像域
        self.filter_size = int(max(1, filter_size))
        self.distance_threshold = None if distance_threshold is None else float(distance_threshold)
        self.thr_percentile = float(thr_percentile)
        self.rel_floor = float(rel_floor)
        self.morph_open = int(max(0, morph_open))
        self.morph_close = int(max(0, morph_close))

        # 稀疏点域
        assert kernel_size % 2 == 1 and kernel_size >= 3, "kernel_size 需为 >=3 的奇数"
        self.kernel_size = int(kernel_size)
        self.min_neighbors = int(max(1, min_neighbors))
        self.clip_percentile = float(clip_percentile)

    # ------------- 图像域去噪 -------------
    def denoise_image(self, radar_image: np.ndarray, return_mask: bool = False):
        """
        对整幅雷达图做去噪（中值 → 自适应阈值成掩膜 → 掩膜上开闭 → 应用掩膜）。
        返回：
            denoised: 去噪后的灰度图（非掩膜区域置 0）
            mask:     二值掩膜（可选）
        """
        img = _ensure_float(radar_image)
        if img is None:
            raise ValueError("radar_image is None")

        out = img.copy()

        # 1) 中值滤波（轻度）
        if self.filter_size > 1:
            out = ndimage.median_filter(out, size=self.filter_size)

        # 2) 阈值（优先绝对阈值；否则仅在非零像素上取分位，并叠加相对地板）
        if self.distance_threshold is not None:
            thr = float(self.distance_threshold)
        else:
            nz = out[out > 0]
            if nz.size > 0:
                tmax = float(nz.max())
                thr_p = np.percentile(nz, self.thr_percentile)
                thr = max(thr_p, self.rel_floor * tmax)
            else:
                thr = np.inf  # 没有非零像素，掩膜全 0

        mask = out >= thr

        # 3) 二值形态学：开（去孤噪）→ 闭（填小孔）
        if self.morph_open > 0:
            mask = ndimage.binary_opening(mask, iterations=self.morph_open)
        if self.morph_close > 0:
            mask = ndimage.binary_closing(mask, iterations=self.morph_close)

        # 4) 应用掩膜（只保留显著区域）
        denoised = np.where(mask, out, 0.0)

        return (denoised, mask) if return_mask else denoised

    # ------------- 稀疏点域去噪 -------------
    def filter_points(self, points, shape_hw):
        """
        对“增量点列表”去噪：
        points: list[(y, x, val)]
        shape_hw: (H, W)
        返回: 过滤后的 list[(y,x,val)]
        """
        H, W = shape_hw
        if not points:
            return []

        hit = np.zeros((H, W), dtype=np.uint8)
        ys, xs, vs = [], [], []
        for (y, x, v) in points:
            if 0 <= y < H and 0 <= x < W:
                hit[y, x] = 1
                ys.append(y); xs.append(x); vs.append(float(v))

        # 邻域计数（含自身）
        k = np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)
        neigh = ndimage.convolve(hit, k, mode='constant', cval=0)

        # winsorize：裁掉极端大值，避免后续累积被少数 outlier 支配
        vs = np.asarray(vs, dtype=np.float32)
        if vs.size > 10:
            high = np.percentile(vs, self.clip_percentile)
        else:
            high = None

        filtered = []
        for y, x, v in zip(ys, xs, vs):
            if neigh[y, x] >= self.min_neighbors:
                if high is not None and v > high:
                    v = float(high)
                filtered.append((y, x, float(v)))
        return filtered


# -----------------------------
# 自测（可选）
# -----------------------------
def _synthetic_demo():
    """合成一个稀疏目标 + 噪声场，展示 denoise_image 的效果。"""
    import matplotlib.pyplot as plt
    H, W = 132, 132
    img = np.zeros((H, W), dtype=np.float32)

    rng = np.random.default_rng(0)
    # 随机噪声孤点
    ny = rng.integers(0, H, size=300)
    nx = rng.integers(0, W, size=300)
    img[ny, nx] = rng.uniform(10, 80, size=300)

    # 模拟一条上方黄点带（目标）
    for i in range(20):
        y = 100 + rng.integers(-1, 2)
        x = 60 + i*2 + rng.integers(-1, 2)
        if 0 <= y < H and 0 <= x < W:
            img[y, x] = 500 + rng.uniform(0, 100)

    denoiser = RadarDenoiser(filter_size=3, thr_percentile=95, rel_floor=0.12,
                             morph_open=1, morph_close=1)
    denoised, mask = denoiser.denoise_image(img, return_mask=True)

    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1); plt.imshow(img, cmap='hot', origin='lower'); plt.title('Raw'); plt.colorbar()
    plt.subplot(1,3,2); plt.imshow(mask, cmap='gray', origin='lower'); plt.title('Mask')
    plt.subplot(1,3,3); plt.imshow(denoised, cmap='hot', origin='lower'); plt.title('Denoised'); plt.colorbar()
    plt.tight_layout(); plt.show()


def test_denoise_with_processor(full_scan_file=None, update_file=None):
    """
    可选：用你的 RadarImageProcessor 读文件后在整幅图上做去噪示例
    full_scan_file: 首帧全景文件
    update_file:    一帧增量文件（可选）
    """
    import matplotlib.pyplot as plt
    from src.pixel_shifting_correction import RadarImageProcessor

    proc = RadarImageProcessor(shift_pixel=4)
    # 构建一幅图：全景 + （可选）一帧更新
    if full_scan_file is None:
        raise ValueError("请提供 full_scan_file 路径")
    proc.process_files(full_scan_file, [update_file] if update_file else None)
    img = proc.radar_image.astype(np.float32)

    denoiser = RadarDenoiser(filter_size=3, thr_percentile=95, rel_floor=0.12,
                             morph_open=1, morph_close=1)
    denoised, mask = denoiser.denoise_image(img, return_mask=True)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(img, cmap='hot', origin='lower'); plt.title('Accumulated'); plt.colorbar()
    plt.subplot(1,3,2); plt.imshow(mask, cmap='gray', origin='lower'); plt.title('Mask')
    plt.subplot(1,3,3); plt.imshow(denoised, cmap='hot', origin='lower'); plt.title('Denoised'); plt.colorbar()
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    # 默认跑合成演示；若想用真实文件，改成：
    # test_denoise_with_processor(full_scan_file="data/1600/valid_framedata_0115n_2.txt",
    #                             update_file="data/1600/valid_framedata_0115n_3.txt")
    _synthetic_demo()
