# src/object_detection.py
# -*- coding: utf-8 -*-

import numpy as np
from scipy import ndimage
from scipy.cluster.hierarchy import fclusterdata
try:
    from skimage.filters import threshold_otsu
except ImportError:
    from skimage.filters.thresholding import threshold_otsu
from skimage.morphology import white_tophat, disk
from skimage.feature import peak_local_max


class SpatialDroneDetector:
    def __init__(self,
                 processor,
                 full_scan_file,
                 update_files,
                 # ----- 阈值 & 形态学 -----
                 use_otsu=False,
                 thr_percentile=45,      # ↓ 放低分位
                 median_size=1,
                 tophat_radius=1,        # ↓ 更小结构元，保留微点
                 # ----- 聚类 & 峰值fallback -----
                 cluster_dist=3,
                 topk=8,
                 # ----- 形状 & 位置过滤 -----
                 min_area=1, max_area=20,
                 max_width=10, max_height=8,
                 min_y=90,
                 # ----- 背景/时间域 -----
                 bg_mode='static',       # 本数据强烈建议 static
                 bg_alpha=0.10,          # ema 时才用
                 temporal_window=1,      # =1 等于关闭
                 # ----- 其他 -----
                 thr_rel_floor=0.06,     # ↓ 相对地板更宽松
                 debug=True):
        self.processor      = processor
        self.full_scan_file = full_scan_file
        self.update_files   = update_files

        self.use_otsu       = use_otsu
        self.thr_percentile = float(thr_percentile)
        self.median_size    = int(median_size)
        self.tophat_radius  = int(tophat_radius)

        self.cluster_dist   = int(cluster_dist)
        self.topk           = int(topk)

        self.min_area       = int(min_area)
        self.max_area       = int(max_area)
        self.max_width      = int(max_width)
        self.max_height     = int(max_height)
        self.min_y          = int(min_y)

        self.bg_mode        = bg_mode
        self.bg_alpha       = float(bg_alpha)
        self.temporal_window= int(temporal_window)
        self.thr_rel_floor  = float(thr_rel_floor)
        self.debug          = debug

        # 以 full_scan 作为“静态背景”
        self.processor.process_files(self.full_scan_file, [])
        self.bg_img = self.processor.radar_image.astype(np.float32).copy()

        # 调试 / 缓存
        self.recent_frames = []
        self.temporal_std  = None
        self.diff_img      = None
        self.tophat_img    = None
        self.mask_img      = None

    def build_accumulated_image(self):
        self.processor.process_files(self.full_scan_file, self.update_files)
        return self.processor.radar_image.astype(np.float32)

    def _update_temporal_buffer(self, frame):
        self.recent_frames.append(frame.copy())
        if len(self.recent_frames) > self.temporal_window:
            self.recent_frames.pop(0)
        if len(self.recent_frames) >= 3:
            stack = np.stack(self.recent_frames, axis=2)
            self.temporal_std = np.std(stack, axis=2)
        else:
            self.temporal_std = None

    def detect(self):
        """返回 [(x0,x1,y0,y1), …] 的列表"""
        # 1) 当前帧（full_scan + 本帧增量）
        curr = self.build_accumulated_image()

        # 2) 时间窗（默认关闭）
        self._update_temporal_buffer(curr)

        # 3) 静态背景差分
        diff = curr - self.bg_img
        diff[diff < 0] = 0.0
        if self.temporal_std is not None:
            std = self.temporal_std
            smax = np.percentile(std, 99)
            if smax > 0:
                w = np.clip(std / smax, 0.0, 1.0)
                diff = diff * w
        if self.debug:
            self.diff_img = diff.copy()

        # 4) 轻中值
        if self.median_size > 1:
            diff = ndimage.median_filter(diff, size=self.median_size)

        # 5) white-tophat（小结构元）
        selem = disk(max(1, self.tophat_radius))
        topht = white_tophat(diff, footprint=selem)
        if self.debug:
            self.tophat_img = topht.copy()

        tmax = float(topht.max())
        if tmax <= 0:
            if self.bg_mode == 'ema':
                a = self.bg_alpha
                self.bg_img = a * curr + (1 - a) * self.bg_img
            self.mask_img = np.zeros_like(topht, dtype=bool)
            if self.debug:
                print("[DEBUG] tophat thr=N/A, tophat max=0.0, non-zero count=0")
            return []

        # 6) 阈值：仅非零分位 + 相对地板
        nz = topht[topht > 0]
        if nz.size >= 20:
            if self.use_otsu:
                try:
                    thr_otsu = threshold_otsu(nz)
                except Exception:
                    thr_otsu = np.percentile(nz, self.thr_percentile)
                thr_pctl = np.percentile(nz, self.thr_percentile)
                thr = max(thr_otsu, thr_pctl)
            else:
                thr = np.percentile(nz, self.thr_percentile)
        else:
            thr = 0.0
        thr = max(thr, self.thr_rel_floor * tmax)
        if self.debug:
            print(f"[DEBUG] tophat thr={thr:.1f}, tophat max={tmax:.1f}, non-zero count={int(nz.size)}")

        mask = topht >= thr

        # 7) 形态学开闭 —— 对极小目标自动“关闭开闭”
        #    若最小面积<=2 或最大宽高<=3，使用 1x1 结构元（等价于不做开闭）
        struct_sz = 1 if (self.min_area <= 2 or (self.max_width <= 3 or self.max_height <= 3)) else 3
        structure = np.ones((struct_sz, struct_sz), dtype=bool)
        mask = ndimage.binary_opening(mask, structure=structure)
        mask = ndimage.binary_closing(mask, structure=structure)
        self.mask_img = mask.copy() if self.debug else None

        # 8) 聚类 → bbox → 形状/高度过滤
        coords = np.column_stack(np.nonzero(mask))
        boxes = []
        if coords.shape[0] > 0:
            try:
                labels = fclusterdata(coords, t=self.cluster_dist,
                                      criterion='distance', metric='euclidean')
                for lab in np.unique(labels):
                    pts = coords[labels == lab]
                    ys, xs = pts[:, 0], pts[:, 1]
                    y0, y1 = ys.min(), ys.max() + 1
                    x0, x1 = xs.min(), xs.max() + 1
                    w, h = x1 - x0, y1 - y0
                    area = w * h
                    if not (self.min_area <= area <= self.max_area):
                        continue
                    if w > self.max_width or h > self.max_height:
                        continue
                    if y0 < self.min_y:
                        continue
                    boxes.append((x0, x1, y0, y1))
            except ValueError:
                boxes = []

        # 9) 峰值回退（给上层 ROI/门控挑选）
        if len(boxes) < self.topk:
            peaks = peak_local_max(topht, num_peaks=max(self.topk * 2, 10),
                                   min_distance=1, footprint=np.ones((3, 3)))
            if peaks.size > 0:
                try:
                    labels2 = fclusterdata(peaks, t=self.cluster_dist,
                                           criterion='distance', metric='euclidean')
                    cand = []
                    for lab in np.unique(labels2):
                        pts = peaks[labels2 == lab]
                        intens = topht[pts[:, 0], pts[:, 1]]
                        idx = int(np.argmax(intens))
                        r, c = pts[idx]
                        half = 2
                        x0 = max(c - half, 0); x1 = min(c + half, diff.shape[1] - 1) + 1
                        y0 = max(r - half, 0); y1 = min(r + half, diff.shape[0] - 1) + 1
                        w, h = x1 - x0, y1 - y0
                        area = w * h
                        if not (self.min_area <= area <= self.max_area):
                            continue
                        if w > self.max_width or h > self.max_height:
                            continue
                        if y0 < self.min_y:
                            continue
                        cand.append((x0, x1, y0, y1))
                    if cand:
                        def center_intensity(b):
                            cy = (b[2] + b[3]) // 2; cx = (b[0] + b[1]) // 2
                            return topht[cy, cx]
                        cand = sorted(cand, key=lambda b: -center_intensity(b))
                        boxes = (boxes + cand)[:self.topk]
                except ValueError:
                    pass

        # static 模式不更新背景
        if self.bg_mode == 'ema':
            a = self.bg_alpha
            self.bg_img = a * curr + (1 - a) * self.bg_img

        return boxes
