# src/object_detection.py

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max

class SpatialDroneDetector:
    def __init__(self,
                 processor,
                 full_scan_file,
                 update_files,
                 median_size=3,
                 use_otsu=True,
                 thr_ratio=0.5,
                 thr_percentile=85,
                 min_area=2,
                 max_area=50,
                 max_width=10,
                 max_height=10,
                 min_y=70,
                 n_targets=4,
                 peak_min_distance=3,
                 peak_pad=1):
        # 把所有传入参数都保存为实例属性
        self.proc = processor
        self.full_scan_file = full_scan_file
        self.update_files = update_files
        self.median_size = median_size
        self.use_otsu = use_otsu
        self.thr_ratio = thr_ratio
        self.thr_percentile = thr_percentile
        self.min_area = min_area
        self.max_area = max_area
        self.max_width = max_width
        self.max_height = max_height
        self.min_y = min_y
        self.n_targets = n_targets
        self.peak_min_distance = peak_min_distance
        self.peak_pad = peak_pad

        # 用于存放累积图
        self.image = None

    def build_accumulated_image(self):
        """
        1) 读取全扫描文件，写入零图
        2) 读取增量帧文件，做最大值累积
        3) 中值滤波去孤立噪点
        """
        # 要求 processor 已经读取过一次 full_scan_file，
        # 并且有 radar_image 属性可以用于 np.zeros_like
        img = np.zeros_like(self.proc.radar_image, dtype=np.float32)

        # 1) 全扫描
        data_full = self.proc.read_radar_data(self.full_scan_file)
        for y, x, d in data_full:
            img[y, x] = d

        # 2) 增量帧累积最大值
        for f in self.update_files:
            data_upd = self.proc.read_radar_data(f)
            for y, x, d in data_upd:
                if d > img[y, x]:
                    img[y, x] = d

        # 3) 中值滤波
        img = ndimage.median_filter(img, size=self.median_size)

        self.image = img
        return img

    def detect(self):
        """
        基于累积图做阈值分割 + 开闭运算 + 连通域过滤，
        最后不够 n_targets 时用峰值补齐。
        返回列表：[(xmin, xmax, ymin, ymax), …]
        """
        if self.image is None:
            self.build_accumulated_image()
        img = self.image

        # 找非零点用于阈值计算
        nonzero = img[img > 0]
        if nonzero.size == 0:
            return []

        # —— 阈值分割 —— 
        if self.use_otsu:
            thr = threshold_otsu(nonzero)
        else:
            # 比如 85% 分位
            thr = np.percentile(nonzero, self.thr_percentile)
        bw = img > thr

        # —— 形态学开闭运算 —— 
        se = np.ones((3, 3), bool)
        bw = ndimage.binary_opening(bw, structure=se)
        bw = ndimage.binary_closing(bw, structure=se)

        # —— 连通域标记 + 面积/宽高/位置双向过滤 —— 
        lbl = label(bw, connectivity=2)
        props = regionprops(lbl)
        boxes = []
        for reg in props:
            area = reg.area
            minr, minc, maxr, maxc = reg.bbox
            h = maxr - minr
            w = maxc - minc

            # 1) 面积过滤
            if area < self.min_area or area > self.max_area:
                continue
            # 2) 宽高过滤
            if w > self.max_width or h > self.max_height:
                continue
            # 3) 地面／大楼噪声过滤
            if minr < self.min_y:
                continue

            # bbox 格式转换为 (xmin, xmax, ymin, ymax)
            boxes.append((minc, maxc - 1, minr, maxr - 1))

        # —— 峰值补齐 —— 
        if len(boxes) < self.n_targets:
            # peak_local_max 在新版 skimage 去掉了 indices 参数
            peaks = peak_local_max(
                img,
                min_distance=self.peak_min_distance,
                threshold_abs=thr,
            )
            # 兼容：如果返回的是布尔掩码，就用 argwhere
            if peaks.dtype == bool:
                coords = np.argwhere(peaks)
            else:
                coords = peaks  # 返回的就是 N×2 (row, col)

            # 按强度降序排序
            intensities = [img[y, x] for y, x in coords]
            order = np.argsort(intensities)[::-1]
            for idx in order:
                if len(boxes) >= self.n_targets:
                    break
                y, x = coords[idx]
                xmin = x - self.peak_pad
                xmax = x + self.peak_pad
                ymin = y - self.peak_pad
                ymax = y + self.peak_pad
                candidate = (xmin, xmax, ymin, ymax)
                if candidate not in boxes:
                    boxes.append(candidate)

        return boxes
