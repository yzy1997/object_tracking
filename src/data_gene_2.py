# -*- coding: utf-8 -*-
"""
4-UAV radar-like sparse echoes on a given background txt.
Target look:
- 4 distinct single-pass tracks crossing the middle area
- per-frame spacing ~3 px, total duration ~20-30 frames
- isolated hard points, occasionally 2~4 px tiny strip/block
- non-periodic, smooth up/down undulation (piecewise curved path), not "flat"
- avoid mixing with strong background by local clean-pixel search (do not create parallel bundles)

Update (2026-01-06):
- Each UAV can spawn from left edge and exit right edge, OR spawn from right and exit left.
- Per-UAV duration is automatically derived to match edge-to-edge travel with ~20-30 frames.

Outputs:
- out/txt/frame_0000.txt : full S-scan (background + hits)
- out/txt/frame_0001..   : incremental updates (curr > prev)
- summary_hits.png       : hit count overlay
- summary_max.png        : max amplitude overlay
"""

import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# =========================
# Basic params
# =========================
W, H = 132, 132
SHIFT = 4
HEADER = 0xA5
DIST_MIN, DIST_MAX = 0x0000, 0xFFFF

# >>> set these two paths <<<
BASE_BG_TXT = r"D:\codes\object_tracking\data\600m2\valid_framedata_0113_3.txt"
OUT_DIR = r"./data/out_uav_4_edge_to_edge_bidir"

TXT_DIR = os.path.join(OUT_DIR, "txt")
SUMMARY_HITS_PNG = os.path.join(OUT_DIR, "summary_hits.png")
SUMMARY_MAX_PNG = os.path.join(OUT_DIR, "summary_max.png")

SEED = 2026
RAND = random.Random(SEED)

# =========================
# Timing / spacing constraints (your stats)
# =========================
# You said: ~20-30 frames total life, spacing ~3 px between adjacent points.
# For edge-to-edge across width W, steps ~= W / vx. Therefore vx must be about 132/20..132/30 => 4.4..6.6
# We'll sample per-UAV vx in this range so each track naturally lasts 20-30 frames while crossing full image.
LIFE_MIN, LIFE_MAX = 20, 30
VX_RANGE_EDGE2EDGE = (W / LIFE_MAX, W / LIFE_MIN)  # ~ (4.4, 6.6)
SKIP = 1  # keep producing 1 output frame per sim step

# =========================
# Coordinate mapping (keep consistent with your old generator)
# =========================
MAPPING_MODE = 1
SHIFT_SIGN_FLIP = False

# =========================
# Echo appearance
# =========================
P_DETECTION = 0.97
P_MULTI_PIXEL = 0.22

UAV_INT_MIN = 450
UAV_INT_MAX = 2000

BG_BLOCK_TH = 1200
DROP_ON_STRONG_BG_PROB = 0.02
CLEAN_SEARCH_R = 14
MAX_OFFSET_FROM_PATH = 2.2

MIN_CONTRAST = 320

# =========================
# Motion: single-pass x + piecewise smooth undulation in y
# =========================
MARGIN_X = 2
MARGIN_Y = 10

K_CTRL = 6
UNDULATE_AMP_RANGE = (4.0, 12.0)
UNDULATE_SMOOTH_JITTER = 0.25
SLOPE_RANGE = (-0.10, 0.10)

JITTER_X_STD = 0.05
JITTER_Y_STD = 0.08

END_PAD = 2  # spawn slightly outside image and exit beyond

# =========================
# Coordinate mapping
# =========================
def raw_to_display(ry: int, rx: int) -> Tuple[int, int]:
    if MAPPING_MODE == 0:
        dy = H - 1 - ry
        base_x = W - 1 - rx
    elif MAPPING_MODE == 1:
        dy = ry
        base_x = rx
    elif MAPPING_MODE == 2:
        dy = H - 1 - ry
        base_x = rx
    elif MAPPING_MODE == 3:
        dy = ry
        base_x = W - 1 - rx
    else:
        raise ValueError("Unknown MAPPING_MODE")

    s = -SHIFT if SHIFT_SIGN_FLIP else SHIFT
    dx = base_x + s if (dy % 2 == 0) else base_x - s
    return dy, dx


def display_to_raw(dy: int, dx: int) -> Tuple[int, int]:
    s = -SHIFT if SHIFT_SIGN_FLIP else SHIFT
    base_x = dx - s if (dy % 2 == 0) else dx + s

    if MAPPING_MODE == 0:
        ry = H - 1 - dy
        rx = W - 1 - base_x
    elif MAPPING_MODE == 1:
        ry = dy
        rx = base_x
    elif MAPPING_MODE == 2:
        ry = H - 1 - dy
        rx = base_x
    elif MAPPING_MODE == 3:
        ry = dy
        rx = W - 1 - base_x
    else:
        raise ValueError("Unknown MAPPING_MODE")

    return int(ry), int(rx)

# =========================
# Encode / decode txt lines
# =========================
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
        xs = range(W) if ry % 2 == 0 else range(W - 1, -1, -1)
        for rx in xs:
            yield ry, rx


def parse_line_5hex(line: str) -> Optional[Tuple[int, int, int]]:
    ps = line.strip().split()
    if len(ps) != 5:
        return None
    try:
        ry = int(ps[1], 16)
        rx = int(ps[2], 16)
        b4 = int(ps[3], 16)
        b5 = int(ps[4], 16)
        dist = (b4 << 8) | b5
        return ry, rx, dist
    except Exception:
        return None

# =========================
# Load background as display matrix
# =========================
def load_background_from_txt(txt_path: str) -> np.ndarray:
    disp = np.zeros((H, W), dtype=np.int32)
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"背景txt不存在：{txt_path}")
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = parse_line_5hex(line)
            if parsed is None:
                continue
            ry, rx, dist = parsed
            dy, dx = raw_to_display(ry, rx)
            if 0 <= dy < H and 0 <= dx < W:
                if dist > disp[dy, dx]:
                    disp[dy, dx] = dist
    return disp

# =========================
# Helpers: smooth undulation (piecewise)
# =========================
def smoothstep(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


def piecewise_smooth(ctrl_t: np.ndarray, ctrl_y: np.ndarray, t: float) -> float:
    if t <= ctrl_t[0]:
        return float(ctrl_y[0])
    if t >= ctrl_t[-1]:
        return float(ctrl_y[-1])
    i = int(np.searchsorted(ctrl_t, t) - 1)
    t0, t1 = float(ctrl_t[i]), float(ctrl_t[i + 1])
    y0, y1 = float(ctrl_y[i]), float(ctrl_y[i + 1])
    u = (t - t0) / (t1 - t0)
    s = smoothstep(u)
    return y0 * (1 - s) + y1 * s

# =========================
# Avoid strong background: find clean nearby pixel, but stay near the path
# =========================
def find_clean_pixel_near(background: np.ndarray, y: float, x: float, r: int) -> Optional[Tuple[int, int]]:
    cy, cx = int(round(y)), int(round(x))
    y0, y1 = max(0, cy - r), min(H - 1, cy + r)
    x0, x1 = max(0, cx - r), min(W - 1, cx + r)

    best = None
    best_score = None

    for iy in range(y0, y1 + 1):
        for ix in range(x0, x1 + 1):
            d = ((iy - y) ** 2 + (ix - x) ** 2) ** 0.5
            if d > MAX_OFFSET_FROM_PATH:
                continue
            bgv = int(background[iy, ix])
            score = bgv + int(60 * d)
            if best is None or score < best_score:
                best = (iy, ix)
                best_score = score

    if best is None:
        best = (cy, cx) if (0 <= cy < H and 0 <= cx < W) else None
    return best


def choose_uav_strength(base_strength: int, bg_value: int) -> int:
    v = int(base_strength * RAND.uniform(0.85, 1.15))
    v = max(v, bg_value + MIN_CONTRAST)
    return clamp16(v)


def paint_uav_sparse(disp: np.ndarray, background: np.ndarray, y: float, x: float, base_strength: int):
    if RAND.random() > P_DETECTION:
        return

    p = find_clean_pixel_near(background, y, x, CLEAN_SEARCH_R)
    if p is None:
        return
    cy, cx = p

    bg_center = int(background[cy, cx])
    if bg_center > BG_BLOCK_TH and RAND.random() < DROP_ON_STRONG_BG_PROB:
        return

    v0 = choose_uav_strength(base_strength, bg_center)
    if v0 > disp[cy, cx]:
        disp[cy, cx] = v0

    if RAND.random() >= P_MULTI_PIXEL:
        return

    k = RAND.randint(1, 3)  # extra pixels => total 2~4
    chosen = {(cy, cx)}
    frontier = [(cy, cx)]

    for _ in range(k):
        sy, sx = RAND.choice(frontier)
        nb = [(sy + dy, sx + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dy == 0 and dx == 0)]
        RAND.shuffle(nb)

        for ny, nx in nb:
            if not (0 <= ny < H and 0 <= nx < W):
                continue
            if (ny, nx) in chosen:
                continue
            if ((ny - y) ** 2 + (nx - x) ** 2) ** 0.5 > MAX_OFFSET_FROM_PATH + 1.2:
                continue

            bgv = int(background[ny, nx])
            if bgv > BG_BLOCK_TH and RAND.random() < 0.35:
                continue

            v = int(v0 * RAND.uniform(0.55, 0.88))
            v = max(v, bgv + int(0.55 * MIN_CONTRAST))
            v = clamp16(v)
            if v > disp[ny, nx]:
                disp[ny, nx] = v

            chosen.add((ny, nx))
            frontier.append((ny, nx))
            break

# =========================
# Single-pass UAV with smooth undulation, bidirectional edge-to-edge
# =========================
@dataclass
class UAVPath:
    x: float
    y_mid: float
    vx: float            # signed vx (positive L->R, negative R->L)
    slope: float
    base_strength: int
    ctrl_t: np.ndarray
    ctrl_off: np.ndarray
    max_steps: int
    active: bool = True
    t: int = 0

    def step(self) -> Optional[Tuple[float, float]]:
        if not self.active:
            return None

        # x evolve
        self.x += self.vx + RAND.gauss(0.0, JITTER_X_STD)

        # normalized time for control curve
        tn = self.t / max(1, (self.max_steps - 1))
        off = piecewise_smooth(self.ctrl_t, self.ctrl_off, tn)

        # y: mid + slope*(x - center) + undulation + small jitter
        x_center = (W - 1) / 2.0
        y = self.y_mid + self.slope * (self.x - x_center) + off
        y += RAND.gauss(0.0, JITTER_Y_STD) + RAND.gauss(0.0, UNDULATE_SMOOTH_JITTER)

        y = max(MARGIN_Y, min(H - 1 - MARGIN_Y, y))

        self.t += 1

        # finish: edge out OR max_steps reached
        if (self.x < -END_PAD) or (self.x > (W - 1 + END_PAD)) or (self.t >= self.max_steps):
            self.active = False

        return y, self.x

    def paint(self, disp: np.ndarray, background: np.ndarray, yx: Tuple[float, float]):
        y, x = yx
        paint_uav_sparse(disp, background, y, x, self.base_strength)


def make_ctrl_curve() -> Tuple[np.ndarray, np.ndarray]:
    ctrl_t = np.linspace(0.0, 1.0, K_CTRL)
    amp = RAND.uniform(*UNDULATE_AMP_RANGE)
    ctrl_off = np.array([RAND.uniform(-amp, amp) for _ in range(K_CTRL)], dtype=np.float32)

    # start/end closer to 0 offset
    ctrl_off[0] *= 0.25
    ctrl_off[-1] *= 0.25

    # ensure at least one sign change in middle (avoid monotone drift)
    if np.all(ctrl_off[1:-1] >= 0) or np.all(ctrl_off[1:-1] <= 0):
        j = RAND.randint(1, K_CTRL - 2)
        ctrl_off[j] *= -1.0

    return ctrl_t, ctrl_off


def _pick_steps_from_v(v_abs: float) -> int:
    """Steps required to travel edge-to-edge (including END_PAD padding)."""
    distance = (W - 1 + 2 * END_PAD)  # from -END_PAD to W-1+END_PAD
    steps = int(np.ceil(distance / max(1e-6, v_abs)))
    return int(max(LIFE_MIN, min(LIFE_MAX, steps)))


def init_4_uavs_cross_middle_bidirectional() -> List[UAVPath]:
    """
    4 UAVs:
    - each starts slightly outside left OR right edge
    - each moves across to the opposite edge within ~20-30 steps by choosing |vx| accordingly
    - y_mids around center so tracks pass middle region
    """
    center = (H - 1) / 2.0

    # y mids: centered band, separated but not too high/low
    y_mids = [center - 14, center - 5, center + 5, center + 14]
    y_mids = [
        max(MARGIN_Y + 3, min(H - 1 - (MARGIN_Y + 3), y + RAND.uniform(-2.0, 2.0)))
        for y in y_mids
    ]
    RAND.shuffle(y_mids)

    base_strengths = [
        RAND.randint(800, 1300),
        RAND.randint(950, 1600),
        RAND.randint(1100, 1900),
        RAND.randint(850, 1700),
    ]
    RAND.shuffle(base_strengths)

    uavs: List[UAVPath] = []
    for i in range(4):
        # choose direction
        dir_lr = (RAND.random() < 0.5)  # True: L->R, False: R->L

        # choose speed magnitude so that steps in [20,30]
        v_abs = RAND.uniform(*VX_RANGE_EDGE2EDGE)
        max_steps = _pick_steps_from_v(v_abs)

        # after clamping steps, adjust v_abs slightly so the UAV can actually cross within max_steps
        # (avoid stopping early just because of ceil/clamp mismatch)
        distance = (W - 1 + 2 * END_PAD)
        v_abs = distance / max_steps * RAND.uniform(0.98, 1.02)

        vx = v_abs if dir_lr else -v_abs

        x0 = -END_PAD if dir_lr else (W - 1 + END_PAD)

        slope = RAND.uniform(*SLOPE_RANGE)
        ctrl_t, ctrl_off = make_ctrl_curve()

        uavs.append(UAVPath(
            x=float(x0),
            y_mid=float(y_mids[i]),
            vx=float(vx),
            slope=float(slope),
            base_strength=int(base_strengths[i]),
            ctrl_t=ctrl_t,
            ctrl_off=ctrl_off,
            max_steps=int(max_steps),
        ))
    return uavs

# =========================
# Diff lines and rendering
# =========================
def display_diff_to_raw_lines(prev_disp: np.ndarray, curr_disp: np.ndarray) -> List[str]:
    lines = []
    ys, xs = np.where(curr_disp > prev_disp)
    for dy, dx in zip(ys.tolist(), xs.tolist()):
        ry, rx = display_to_raw(int(dy), int(dx))
        if 0 <= ry < H and 0 <= rx < W:
            lines.append(encode_line(ry, rx, int(curr_disp[dy, dx])))
    return lines


def full_s_scan_lines_from_display(disp: np.ndarray) -> List[str]:
    lines = []
    for ry, rx in s_scan_raw():
        dy, dx = raw_to_display(ry, rx)
        if 0 <= dy < H and 0 <= dx < W:
            lines.append(encode_line(ry, rx, int(disp[dy, dx])))
        else:
            lines.append(encode_line(ry, rx, 0))
    return lines


def save_summary_hits(background_disp: np.ndarray, hit_count: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bg = background_disp.astype(np.float32)

    plt.figure(figsize=(6, 6), dpi=220)
    bg_vmax = np.percentile(bg, 99.7) if np.any(bg) else 1.0
    plt.imshow(bg, cmap="viridis", vmin=0, vmax=max(800.0, bg_vmax), interpolation="nearest")

    hc = hit_count.astype(np.float32)
    hc_vmax = max(1.0, np.percentile(hc, 99.5) if np.any(hc) else 1.0)
    alpha = np.clip(hc / hc_vmax, 0, 1) * 0.95

    plt.imshow(hc, cmap="autumn", vmin=0, vmax=hc_vmax, interpolation="nearest", alpha=alpha)
    plt.title("BG | hit-count overlay (4 UAV, edge-to-edge, bidir, undulating)", fontsize=9)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_summary_max(background_disp: np.ndarray, overlay_max: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bg = background_disp.astype(np.float32)
    ov = overlay_max.astype(np.float32)

    plt.figure(figsize=(6, 6), dpi=220)
    bg_vmax = np.percentile(bg, 99.7) if np.any(bg) else 1.0
    plt.imshow(bg, cmap="viridis", vmin=0, vmax=max(800.0, bg_vmax), interpolation="nearest")

    ov_vmax = max(1.0, np.percentile(ov, 99.5) if np.any(ov) else 1.0)
    alpha = np.clip(ov / ov_vmax, 0, 1) * 0.95
    plt.imshow(ov, cmap="autumn", vmin=0, vmax=ov_vmax, interpolation="nearest", alpha=alpha)
    plt.title("BG | max-amplitude overlay (4 UAV, edge-to-edge, bidir, undulating)", fontsize=9)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TXT_DIR, exist_ok=True)

    background = load_background_from_txt(BASE_BG_TXT)

    # init 4 uavs bidirectionally from edges
    uavs = init_4_uavs_cross_middle_bidirectional()

    # overall simulation steps: run until all inactive, but cap at a safe max
    # safe max uses the largest per-UAV max_steps plus small padding
    maxT = max(u.max_steps for u in uavs) + 3

    prev_disp = None
    overlay_max = np.zeros((H, W), dtype=np.int32)
    hit_count = np.zeros((H, W), dtype=np.int32)

    out_index = 0

    for t in range(maxT):
        disp = background.copy()

        do_paint = (t % SKIP) == 0
        if do_paint:
            for u in uavs:
                yx = u.step()
                if yx is None:
                    continue
                u.paint(disp, background, yx)

        # summaries relative to background
        delta = disp - background
        delta[delta < 0] = 0
        overlay_max = np.maximum(overlay_max, delta)
        hit_count += (delta > 0).astype(np.int32)

        if not do_paint:
            continue

        # write txt frames
        if prev_disp is None:
            lines = full_s_scan_lines_from_display(disp)
        else:
            lines = display_diff_to_raw_lines(prev_disp, disp)

        with open(os.path.join(TXT_DIR, f"frame_{out_index:04d}.txt"), "w", encoding="utf-8") as f:
            f.writelines(lines)

        prev_disp = disp
        out_index += 1

        if all(not u.active for u in uavs):
            break

    save_summary_hits(background, hit_count, SUMMARY_HITS_PNG)
    save_summary_max(background, overlay_max, SUMMARY_MAX_PNG)

    print("Done.")
    print(f"- background: {BASE_BG_TXT}")
    print(f"- out txt dir: {TXT_DIR}")
    print(f"- summary_hits: {SUMMARY_HITS_PNG}")
    print(f"- summary_max:  {SUMMARY_MAX_PNG}")
    print(f"- simulated steps: {t+1}, output frames: {out_index}, SKIP={SKIP}")
    for i, u in enumerate(uavs):
        direction = "L->R" if u.vx > 0 else "R->L"
        print(f"  UAV{i}: dir={direction}, |vx|={abs(u.vx):.2f} px/frame, max_steps={u.max_steps}")


if __name__ == "__main__":
    main()