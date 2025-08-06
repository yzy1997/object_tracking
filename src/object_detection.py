# src/object_detection.py

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
                 use_otsu=True,
                 thr_percentile=80,
                 median_size=3,
                 # ----- 聚类 & 峰值fallback -----
                 tophat_radius=5,
                 cluster_dist=6,
                 topk=4,
                 # ----- 形状 & 位置过滤 -----
                 min_area=2, max_area=100,
                 max_width=15, max_height=15,
                 min_y=60):
        """
        processor:          RadarImageProcessor 实例
        full_scan_file:     静态全景文件，用于背景减法
        update_files:       后续帧文件列表
        use_otsu:           是否对 tophat 图用 Otsu
        thr_percentile:     若不启用 Otsu，就用百分位阈值
        median_size:        差分后做中值滤波去孤点
        tophat_radius:      white_tophat 的半径
        cluster_dist:       聚类时的距离阈值（像素）
        topk:               最终想要的无人机数量
        min_area...         连通域面积／框大小过滤
        min_y:              仅保留 y0>=min_y 的框
        """
        self.processor      = processor
        self.full_scan_file = full_scan_file
        self.update_files   = update_files

        self.use_otsu       = use_otsu
        self.thr_percentile = thr_percentile
        self.median_size    = median_size

        self.tophat_radius  = tophat_radius
        self.cluster_dist   = cluster_dist
        self.topk           = topk

        self.min_area       = min_area
        self.max_area       = max_area
        self.max_width      = max_width
        self.max_height     = max_height
        self.min_y          = min_y

        # 一次性读全景背景
        self.processor.process_files(self.full_scan_file, [])
        self.bg_img = self.processor.radar_image.astype(np.float32)

    def build_accumulated_image(self):
        self.processor.process_files(self.full_scan_file, self.update_files)
        return self.processor.radar_image.astype(np.float32)

    def detect(self):
        """返回 [(x0,x1,y0,y1), …] 的列表，数量动态自适应，至少 topk 个"""
        # 1) 背景减法 + 非负
        curr = self.build_accumulated_image()
        diff = curr - self.bg_img
        diff[diff < 0] = 0

        # 2) 中值滤波
        if self.median_size > 1:
            diff = ndimage.median_filter(diff, size=self.median_size)

        # 3) white-tophat 提取小亮点
        selem = disk(self.tophat_radius)
        topht = white_tophat(diff, footprint=selem)

        # 4) 阈值分割
        if self.use_otsu:
            thr = threshold_otsu(topht)
        else:
            thr = np.percentile(topht, self.thr_percentile)
        print(f"[DEBUG] tophat thr={thr:.1f}, tophat max={topht.max():.1f}")
        mask = topht >= thr

        # 5) 形态学开闭，去掉毛刺
        mask = ndimage.binary_opening(mask, structure=np.ones((3,3)))
        mask = ndimage.binary_closing(mask, structure=np.ones((3,3)))

        # 6) 找 mask 非零像素坐标
        coords = np.column_stack(np.nonzero(mask))
        boxes = []

        # 7) 如果 coords 里有点，则做层次聚类
        if coords.shape[0] > 0:
            try:
                labels = fclusterdata(coords,
                                      t=self.cluster_dist,
                                      criterion='distance',
                                      metric='euclidean')
                # 8) 对每个簇求 bbox 并做面积／宽高／高度过滤
                for lab in np.unique(labels):
                    pts = coords[labels == lab]
                    ys, xs = pts[:,0], pts[:,1]
                    y0, y1 = ys.min(), ys.max()+1
                    x0, x1 = xs.min(), xs.max()+1
                    w, h = x1-x0, y1-y0
                    area = w*h
                    if not (self.min_area <= area <= self.max_area):
                        continue
                    if w > self.max_width or h > self.max_height:
                        continue
                    if y0 < self.min_y:
                        continue
                    boxes.append((x0, x1, y0, y1))
            except ValueError:
                # empty distance matrix 或其他异常，跳过本次聚类
                boxes = []

        # 9) 如果聚类后还不到 topk，就退到峰值检测 + 同样聚类去重
        if len(boxes) < self.topk:
            peaks = peak_local_max(topht,
                                   num_peaks=self.topk * 2,
                                   footprint=np.ones((3,3)))
            # 再次 guard：没有检测到任何峰值就直接返回当前 boxes（可能为空）
            if peaks.size > 0:
                try:
                    labels2 = fclusterdata(peaks,
                                           t=self.cluster_dist,
                                           criterion='distance',
                                           metric='euclidean')
                    cand = []
                    for lab in np.unique(labels2):
                        pts = peaks[labels2 == lab]
                        intens = topht[pts[:,0], pts[:,1]]
                        idx = np.argmax(intens)
                        r, c = pts[idx]
                        half = 3
                        x0 = max(c-half, 0)
                        x1 = min(c+half, diff.shape[1])
                        y0 = max(r-half, 0)
                        y1 = min(r+half, diff.shape[0])
                        cand.append((x0, x1, y0, y1))
                    # 按框心点强度排序并取前 topk
                    cand = sorted(cand,
                                  key=lambda b: - topht[(b[2]+b[3])//2,
                                                        (b[0]+b[1])//2])
                    boxes = cand[:self.topk]
                except ValueError:
                    # 聚类出错就直接跳过
                    pass

        return boxes
