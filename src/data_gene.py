# -*- coding: utf-8 -*-
"""
噪声背景 + 不规则稀疏轨迹（含正弦/急弯/缓弯/多模式）
首帧：原始S扫描全景写出；后续：仅写增量（显示→逆映射→raw）
并导出每帧PNG预览
"""
import os, math, random
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# ========= 基本参数（与读取端一致） =========
W, H = 132, 132
SHIFT = 4
HEADER = 0xA5
RAND = random.Random(2025)
DIST_MIN, DIST_MAX = 0x0000, 0xFFFF

# ========= 可调：稀疏轨迹风格 =========
DOT_SPACING_PX = 7        # 弧长阈值（像素），越大越稀疏
DOT_STYLE = "cross"       # "cross" 或 "dot"
DOT_RADIUS = 1            # 十字臂长或圆点半径
TRAIL_INTENSITY_MIN = 0x1500
TRAIL_INTENSITY_MAX = 0x1B00


# ==== 更小、更稀疏的点状噪声参数 ====
NOISE_SEED = 2025
SPECKLE_DENSITY = 0.0009      # 每个像素成为噪声点的概率，越小越少（建议 0.0005~0.0015）
SPECKLE_VAL_RANGE = (320, 620) # 单点强度范围，背景之上不多不少
CLUSTER_POINTS = 10            # 额外“点簇”的中心个数（很少）
CLUSTER_RADIUS = 1             # 点簇半径(0或1，别太大)
CLUSTER_VAL_RANGE = (650, 900) # 点簇强度范围

# ========= raw <-> display 映射 =========
def raw_to_display(ry, rx):
    dy = H - 1 - ry
    if dy % 2 == 0:
        dx = (W - 1 - rx) + SHIFT
    else:
        dx = (W - 1 - rx) - SHIFT
    return dy, dx

def display_to_raw(dy, dx):
    ry = H - 1 - dy
    if dy % 2 == 0:
        rx = W - 1 - (dx - SHIFT)
    else:
        rx = W - 1 - (dx + SHIFT)
    return int(ry), int(rx)

# ========= I/O & 工具 =========
def clamp16(v: int) -> int:
    return max(DIST_MIN, min(DIST_MAX, int(v)))

def hex2(v: int) -> str:
    return f"{v & 0xFF:02X}"

def encode_line(ry: int, rx: int, dist: int) -> str:
    d = clamp16(dist)
    b4, b5 = (d >> 8) & 0xFF, d & 0xFF
    return f"{hex2(HEADER)} {hex2(ry)} {hex2(rx)} {hex2(b4)} {hex2(b5)}\n"

def s_scan_raw():
    for ry in range(H):
        xs = range(W) if ry % 2 == 0 else range(W-1, -1, -1)
        for rx in xs:
            yield ry, rx

# ========= 背景恢复（从你的txt）+ 噪声叠加 =========
def read_one_txt_to_display(filepath: str, display_frame: np.ndarray):
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            ps = line.strip().split()
            if len(ps) != 5: 
                continue
            try:
                ry = int(ps[1], 16); rx = int(ps[2], 16)
                b4 = int(ps[3], 16); b5 = int(ps[4], 16)
                dist = (b4 << 8) | b5
                dy, dx = raw_to_display(ry, rx)
                if 0 <= dy < H and 0 <= dx < W:
                    display_frame[dy, dx] = max(display_frame[dy, dx], dist)
            except:
                continue

def build_background_from_files(bg_files: List[str]) -> np.ndarray:
    frame = np.zeros((H, W), dtype=np.int32)
    for fp in bg_files:
        read_one_txt_to_display(fp, frame)
    return frame

def gaussian_blur_field(field: np.ndarray, sigma: float) -> np.ndarray:
    """简单 separable 高斯模糊（不依赖外部库）"""
    if sigma <= 0:
        return field
    radius = int(3*sigma)
    ax = np.arange(-radius, radius+1, dtype=np.float32)
    kernel = np.exp(-0.5*(ax/sigma)**2)
    kernel /= kernel.sum()
    # 横向
    tmp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), 1, field)
    # 纵向
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), 0, tmp)
    return out

def add_noise_to_display_micro(base: np.ndarray, seed=NOISE_SEED) -> np.ndarray:
    """
    只加稀疏点状噪声：
      1) 独立的椒盐小点（单像素）
      2) 极少量的小点簇（半径0~1）
    不做任何云雾/大尺度噪声
    """
    rng = np.random.default_rng(seed)
    disp = base.astype(np.int32).copy()

    H, W = disp.shape
    # (1) 独立椒盐点（单像素）
    mask = rng.random((H, W)) < SPECKLE_DENSITY
    if mask.any():
        vals = rng.integers(SPECKLE_VAL_RANGE[0], SPECKLE_VAL_RANGE[1] + 1, size=(H, W))
        disp[mask] = np.maximum(disp[mask], vals[mask])

    # (2) 极少量小点簇
    for _ in range(CLUSTER_POINTS):
        y = int(rng.integers(0, H)); x = int(rng.integers(0, W))
        val = int(rng.integers(CLUSTER_VAL_RANGE[0], CLUSTER_VAL_RANGE[1] + 1))
        r = int(CLUSTER_RADIUS)
        if r <= 0:
            # 单像素小亮点
            disp[y, x] = max(disp[y, x], val)
        else:
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < H and 0 <= xx < W and (dy*dy + dx*dx) <= r*r:
                        disp[yy, xx] = max(disp[yy, xx], val)

    return disp

# ========= 绘制稀疏“点” =========
def paint_dot(display_frame: np.ndarray, y: float, x: float, dist: int, r: int):
    cy, cx = int(round(y)), int(round(x))
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            yy, xx = cy + dy, cx + dx
            if 0 <= yy < H and 0 <= xx < W and (dy*dy + dx*dx) <= r*r:
                display_frame[yy, xx] = max(display_frame[yy, xx], clamp16(dist))

def paint_cross(display_frame: np.ndarray, y: float, x: float, dist: int, arm: int):
    cy, cx = int(round(y)), int(round(x))
    for d in range(-arm, arm+1):
        if 0 <= cy + d < H: display_frame[cy + d, cx] = max(display_frame[cy + d, cx], clamp16(dist))
        if 0 <= cx + d < W: display_frame[cy, cx + d] = max(display_frame[cy, cx + d], clamp16(dist))

def paint_sparse_point(display_frame, y, x, dist, style="cross", size=1):
    if style == "cross": paint_cross(display_frame, y, x, dist, arm=size)
    else:                 paint_dot(display_frame, y, x, dist, r=size)

# ========= 轨迹基类（弧长稀疏） =========
class SparseTrailBase:
    def __init__(self, y, x, yaw, v, dist_center, dot_spacing, style, size):
        self.y, self.x = float(y), float(x)
        self.yaw, self.v = float(yaw), float(v)
        self.dist_center = int(dist_center)
        self.dot_spacing = float(dot_spacing)
        self.style, self.size = style, int(size)
        self._acc = 0.0

    def _after_move(self, y_prev, x_prev):
        self._acc += math.hypot(self.y - y_prev, self.x - x_prev)
        # 软边界
        m = 1.0
        self.y = min(max(self.y, m), H-1-m)
        self.x = min(max(self.x, m), W-1-m)

    def maybe_paint(self, disp: np.ndarray):
        if self._acc >= self.dot_spacing:
            paint_sparse_point(disp, self.y, self.x, self.dist_center, self.style, self.size)
            self._acc = 0.0

# -------- 轨迹1：正弦型（沿 x 推进，y= y0 + A sin(2π f t + φ) + 小扰动）--------
class SineTrail(SparseTrailBase):
    def __init__(self, y, x, v_x=1.2, amp=14, freq=0.05, phase=None,
                 jitter=0.25, **kw):
        super().__init__(y, x, kw.get("yaw", 0.0), kw.get("v", 0.0),
                         kw["dist_center"], kw["dot_spacing"], kw["style"], kw["size"])
        self.vx = float(v_x)
        self.amp = float(amp)
        self.freq = float(freq)
        self.phase = RAND.uniform(0, 2*math.pi) if phase is None else float(phase)
        self.t = 0.0
        self.jitter = float(jitter)

    def step(self):
        y_prev, x_prev = self.y, self.x
        self.t += 1.0
        self.x += self.vx + RAND.uniform(-0.1, 0.1)
        self.y  = self.y + (self.amp * (2*math.pi*self.freq) * math.cos(2*math.pi*self.freq*self.t + self.phase))
        self.y += RAND.uniform(-self.jitter, self.jitter)  # 轻微扰动
        self._after_move(y_prev, x_prev)

# -------- 轨迹2：急弯（大角速度随机游走）--------
class AggressiveTurnTrail(SparseTrailBase):
    def __init__(self, y, x, yaw, v=1.6, omegamax=math.radians(28), amax=0.25, **kw):
        super().__init__(y, x, yaw, v, kw["dist_center"], kw["dot_spacing"], kw["style"], kw["size"])
        self.omegamax = float(omegamax); self.amax = float(amax)

    def step(self):
        y_prev, x_prev = self.y, self.x
        omega = RAND.uniform(-self.omegamax, self.omegamax)
        self.v = max(0.5, min(3.0, self.v + RAND.uniform(-self.amax, self.amax)))
        if abs(omega) < 1e-6:
            self.y += self.v * math.sin(self.yaw)
            self.x += self.v * math.cos(self.yaw)
        else:
            R = self.v / omega
            self.y += R * (math.sin(self.yaw + omega) - math.sin(self.yaw))
            self.x += R * (math.cos(self.yaw + omega) - math.cos(self.yaw))
            self.yaw += omega
        self._after_move(y_prev, x_prev)

# -------- 轨迹3：缓弯（小角速度+偶发转向）--------
class GentleTurnTrail(SparseTrailBase):
    def __init__(self, y, x, yaw, v=1.2, omegamax=math.radians(10), amax=0.18, **kw):
        super().__init__(y, x, yaw, v, kw["dist_center"], kw["dot_spacing"], kw["style"], kw["size"])
        self.omegamax = float(omegamax); self.amax = float(amax); self.k = 0

    def step(self):
        y_prev, x_prev = self.y, self.x
        self.k += 1
        omega = RAND.uniform(-self.omegamax, self.omegamax)
        if self.k % 20 == 0:  # 偶发较大转向
            omega += RAND.uniform(-self.omegamax*2.2, self.omegamax*2.2)
        self.v = max(0.5, min(2.5, self.v + RAND.uniform(-self.amax, self.amax)))
        if abs(omega) < 1e-6:
            self.y += self.v * math.sin(self.yaw)
            self.x += self.v * math.cos(self.yaw)
        else:
            R = self.v / omega
            self.y += R * (math.sin(self.yaw + omega) - math.sin(self.yaw))
            self.x += R * (math.cos(self.yaw + omega) - math.cos(self.yaw))
            self.yaw += omega
        self._after_move(y_prev, x_prev)

# -------- 轨迹4：多模式+OU噪声（不规则）--------
class OUNoise:
    def __init__(self, mu=0.0, theta=0.25, sigma=0.12):
        self.mu, self.theta, self.sigma = mu, theta, sigma
        self.state = 0.0
    def step(self):
        dx = self.theta*(self.mu - self.state) + self.sigma*RAND.gauss(0,1)
        self.state += dx
        return self.state

class VariedTrail(SparseTrailBase):
    MODES = ("CRUISE","ZIG","ZAG","BURST")
    def __init__(self, y, x, yaw, v=1.4, omegamax=math.radians(18), amax=0.22, **kw):
        super().__init__(y, x, yaw, v, kw["dist_center"], kw["dot_spacing"], kw["style"], kw["size"])
        self.omegamax, self.amax = omegamax, amax
        self.mode = RAND.choice(self.MODES); self.t = 0; self.T = RAND.randint(10, 24)
        self.noise_om = OUNoise(sigma=0.15); self.noise_a = OUNoise(sigma=0.10)

    def step(self):
        y_prev, x_prev = self.y, self.x
        self.t += 1
        if self.t >= self.T:
            self.mode = RAND.choice(self.MODES); self.t = 0; self.T = RAND.randint(10, 24)

        om_cmd = 0.0; a_cmd = 0.0
        if self.mode == "CRUISE": om_cmd = 0.0; a_cmd = 0.0
        elif self.mode == "ZIG":  om_cmd = +0.6*self.omegamax
        elif self.mode == "ZAG":  om_cmd = -0.6*self.omegamax
        elif self.mode == "BURST": om_cmd = RAND.uniform(-self.omegamax, self.omegamax); a_cmd = +0.7*self.amax

        omega = max(-self.omegamax, min(self.omegamax, om_cmd + 0.6*self.noise_om.step()*self.omegamax))
        acc   = max(-self.amax,     min(self.amax,     a_cmd + 0.6*self.noise_a.step()*self.amax))
        self.v = max(0.5, min(3.0, self.v + acc))

        if abs(omega) < 1e-6:
            self.y += self.v * math.sin(self.yaw)
            self.x += self.v * math.cos(self.yaw)
        else:
            R = self.v / omega
            self.y += R * (math.sin(self.yaw + omega) - math.sin(self.yaw))
            self.x += R * (math.cos(self.yaw + omega) - math.cos(self.yaw))
            self.yaw += omega

        self._after_move(y_prev, x_prev)

# ========= 差分/写文件/预览 =========
def diff_display(prev: np.ndarray, cur: np.ndarray):
    ys, xs = np.nonzero(prev != cur)
    return [(int(y), int(x), int(cur[y, x])) for y, x in zip(ys, xs)]

def write_full_raw(path: str, display_frame: np.ndarray):
    with open(path, "w", encoding="utf-8") as f:
        for ry, rx in s_scan_raw():
            dy, dx = raw_to_display(ry, rx)
            dist = int(display_frame[dy, dx]) if (0 <= dy < H and 0 <= dx < W) else 0
            f.write(encode_line(ry, rx, dist))

def write_delta_raw(path: str, changed_points_display):
    with open(path, "w", encoding="utf-8") as f:
        for dy, dx, dist in changed_points_display:
            ry, rx = display_to_raw(dy, dx)
            if 0 <= ry < H and 0 <= rx < W:
                f.write(encode_line(ry, rx, dist))

def save_png(display_frame: np.ndarray, out_png: str, title="Corrected Radar Image"):
    arr = display_frame.astype(float)
    nz = arr[arr > 0]
    vmin, vmax = (nz.min(), nz.max()*0.9) if nz.size else (0, 1)
    plt.figure(figsize=(8,8))
    im = plt.imshow(arr, cmap="hot", origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="Distance")
    plt.xlabel("X Coordinate"); plt.ylabel("Y Coordinate")
    plt.title(title); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

# ========= 主流程 =========
def main():
    # 背景：把你已有的 txt 作为底图（可多个）
    bg_files = [
        r"D:\codes\object_tracking\data\600m2\valid_framedata_0113_3.txt"   # ← 改成你的路径，支持多个
    ]
    out_dir = r"D:\codes\object_tracking\data\out_six_sparse_trails_varied_noisy"
    prefix = "valid_framedata_0113"
    num_frames = 40
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "preview_png"), exist_ok=True)

    base_bg = build_background_from_files(bg_files)
    noisy_bg = add_noise_to_display_micro(base_bg, seed=NOISE_SEED)

    # 轨迹集合：正弦×2、急弯×2、缓弯×1、不规则×1（共6条）
    trails = []
    # 正弦（幅度不同、频率不同）
    trails.append(SineTrail(y=RAND.uniform(20, H-20), x=RAND.uniform(5, 20),
                            v_x=1.1, amp=RAND.uniform(10,16), freq=RAND.uniform(0.035,0.06),
                            dist_center=int(RAND.uniform(TRAIL_INTENSITY_MIN, TRAIL_INTENSITY_MAX)),
                            dot_spacing=DOT_SPACING_PX, style=DOT_STYLE, size=DOT_RADIUS))
    trails.append(SineTrail(y=RAND.uniform(30, H-30), x=RAND.uniform(10, 25),
                            v_x=1.4, amp=RAND.uniform(6,12),  freq=RAND.uniform(0.05,0.08),
                            dist_center=int(RAND.uniform(TRAIL_INTENSITY_MIN, TRAIL_INTENSITY_MAX)),
                            dot_spacing=DOT_SPACING_PX, style=DOT_STYLE, size=DOT_RADIUS))
    # 急弯
    trails.append(AggressiveTurnTrail(y=RAND.uniform(10, H-10), x=RAND.uniform(10, W-10),
                            yaw=RAND.uniform(-math.pi, math.pi),
                            dist_center=int(RAND.uniform(TRAIL_INTENSITY_MIN, TRAIL_INTENSITY_MAX)),
                            dot_spacing=DOT_SPACING_PX, style=DOT_STYLE, size=DOT_RADIUS))
    trails.append(AggressiveTurnTrail(y=RAND.uniform(10, H-10), x=RAND.uniform(10, W-10),
                            yaw=RAND.uniform(-math.pi, math.pi),
                            dist_center=int(RAND.uniform(TRAIL_INTENSITY_MIN, TRAIL_INTENSITY_MAX)),
                            dot_spacing=DOT_SPACING_PX, style=DOT_STYLE, size=DOT_RADIUS))
    # 缓弯
    trails.append(GentleTurnTrail(y=RAND.uniform(10, H-10), x=RAND.uniform(10, W-10),
                            yaw=RAND.uniform(-math.pi, math.pi),
                            dist_center=int(RAND.uniform(TRAIL_INTENSITY_MIN, TRAIL_INTENSITY_MAX)),
                            dot_spacing=DOT_SPACING_PX, style=DOT_STYLE, size=DOT_RADIUS))
    # 不规则
    trails.append(VariedTrail(y=RAND.uniform(10, H-10), x=RAND.uniform(10, W-10),
                            yaw=RAND.uniform(-math.pi, math.pi),
                            dist_center=int(RAND.uniform(TRAIL_INTENSITY_MIN, TRAIL_INTENSITY_MAX)),
                            dot_spacing=DOT_SPACING_PX, style=DOT_STYLE, size=DOT_RADIUS))

    # 帧1：仅背景（全景写出）
    frame0 = noisy_bg.copy()
    write_full_raw(os.path.join(out_dir, f"{prefix}_1.txt"), frame0)
    save_png(frame0, os.path.join(out_dir, "preview_png", f"{prefix}_1.png"),
             title="Frame 1 (Noisy Background)")
    print("[OK] 写入首帧")

    prev = frame0.copy()
    cur  = frame0.copy()

    # 后续帧：推进 + 弧长稀疏落点 → 增量写出
    for k in range(2, num_frames + 1):
        for t in trails:
            t.step()
            t.maybe_paint(cur)
        changes = diff_display(prev, cur)
        write_delta_raw(os.path.join(out_dir, f"{prefix}_{k}.txt"), changes)
        save_png(cur, os.path.join(out_dir, "preview_png", f"{prefix}_{k}.png"),
                 title=f"Frame {k} (Varied Sparse Trails + Noise)")
        print(f"[OK] 增量帧{k} 新增像素={len(changes)}")
        prev = cur.copy()

if __name__ == "__main__":
    main()
