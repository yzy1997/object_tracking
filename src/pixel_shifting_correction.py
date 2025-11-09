# src/pixel_shifting_correction.py
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import mplcursors

class RadarImageProcessor:
    """
    负责将原始雷达 txt 转成校正后的二维矩阵图像。
    支持：
      - 分辨率与 S 型扫描像素校正（原有逻辑）
      - 可选：对“增量点”做邻域一致性去噪（通过传入 denoiser）
    """
    def __init__(self, resolution=(132, 132), shift_pixel=4, denoiser=None):
        self.resolution = resolution         # (W, H)
        self.shift_pixel = shift_pixel
        self.denoiser = denoiser             # 可选：RadarDenoiser 实例（带 filter_points）
        # 注意：内部矩阵用 (H, W) 存储
        self.radar_image = np.zeros((resolution[1], resolution[0]), dtype=np.float32)

    @staticmethod
    def parse_distance(byte4, byte5):
        # 原始两字节合成距离（根据你的数据协议）
        return (byte4 << 8) | byte5

    def read_radar_data(self, filepath):
        """
        读取单个 txt，按坐标校正后返回稀疏点列表：
          [(y, x, distance), ...]
        若传入了 denoiser，则对这些点做“邻域一致性去噪”后再返回。
        """
        data = []
        W, H = self.resolution[0], self.resolution[1]

        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                try:
                    raw_y = int(parts[1], 16)
                    raw_x = int(parts[2], 16)
                    byte4 = int(parts[3], 16)
                    byte5 = int(parts[4], 16)
                    distance = self.parse_distance(byte4, byte5)

                    # --- 坐标校正核心 ---
                    # 1) matrix->笛卡尔：垂直翻转
                    adjusted_y = H - 1 - raw_y
                    # 2) S 型扫描：偶数行水平翻转并做像素修正
                    if adjusted_y % 2 == 0:
                        adjusted_x = W - 1 - raw_x + self.shift_pixel
                    else:
                        adjusted_x = W - 1 - raw_x - self.shift_pixel

                    if 0 <= adjusted_x < W and 0 <= adjusted_y < H:
                        data.append((adjusted_y, adjusted_x, float(distance)))

                except ValueError:
                    continue

        # --- 可选：对稀疏增量点做邻域一致性去噪 ---
        if self.denoiser is not None and len(data) > 0:
            data = self.denoiser.filter_points(data, (H, W))

        return data

    def update_image(self, data):
        """
        将稀疏点写入内部矩阵（注意：此处为“覆盖/叠加更新”，不做清零）
        """
        for y, x, distance in data:
            self.radar_image[int(y), int(x)] = float(distance)

    def process_files(self, full_scan_file, update_files=None):
        """
        处理全景 + 若干更新帧：
          - 先用全景初始化
          - 再叠加若干增量（每个都可经过去噪）
        """
        # 清零再重建
        self.radar_image.fill(0.0)

        # 全景背景
        full_data = self.read_radar_data(full_scan_file)
        self.update_image(full_data)

        # 增量更新
        if update_files:
            for file in update_files:
                update_data = self.read_radar_data(file)
                self.update_image(update_data)

    def show_interactive_image(self, cmap='hot', vmin=None, vmax=None):
        fig, ax = plt.subplots(figsize=(10, 10))

        # 自动确定颜色范围（排除 0 值）
        non_zero = self.radar_image[self.radar_image > 0]
        if non_zero.size > 0:
            data_min = float(np.min(non_zero))
            data_max = float(np.max(non_zero))
            vmin = data_min if vmin is None else vmin
            vmax = (data_max * 0.8) if vmax is None else vmax

        img = ax.imshow(self.radar_image,
                        cmap=cmap,
                        origin='lower',
                        vmin=vmin,
                        vmax=vmax)

        plt.colorbar(img, label='Distance')
        ax.set_title('Corrected Radar Image (S-pattern Fixed)')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')

        # 交互光标
        cursor = mplcursors.cursor(img, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            # 注意：imshow 的 target 是 (y, x)
            y = int(sel.target[0])
            x = int(sel.target[1])
            distance = float(self.radar_image[y, x])
            sel.annotation.set(
                text=f"X: {x}\nY: {y}\nDistance: {distance:.1f}",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.9)
            )

        plt.show()

    def save_image(self, output_path=None, filename="corrected_radar.png",
                   cmap='hot', vmin=None, vmax=None, dpi=150):
        if output_path is None:
            output_path = os.getcwd()

        plt.figure(figsize=(10, 10))
        non_zero = self.radar_image[self.radar_image > 0]
        if non_zero.size > 0:
            data_min = float(np.min(non_zero))
            data_max = float(np.max(non_zero))
            vmin = data_min if vmin is None else vmin
            vmax = (data_max * 0.8) if vmax is None else vmax

        img = plt.imshow(self.radar_image,
                         cmap=cmap,
                         origin='lower',
                         vmin=vmin,
                         vmax=vmax)
        plt.colorbar(img, label='Distance')
        plt.title('Corrected Radar Image')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')

        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, filename)
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        return output_file


def natural_sort_key(s):
    import re
    parts = re.split(r'(\d+)', s)
    return [int(p) if p.isdigit() else p for p in parts]


# 使用示例（可选）
if __name__ == "__main__":
    import glob

    # 1) 指定数据目录
    data_dir = r"D:\codes\object_tracking\dataout_six_sparse_trails_separated"
    pattern = os.path.join(data_dir, "valid_framedata_*.txt")

    all_files = sorted(glob.glob(pattern), key=natural_sort_key)
    if len(all_files) < 2:
        print(f"❌ 在 {data_dir} 下未找到足够的 txt 文件，至少要两个：{pattern}")
        exit(1)

    full_scan_file = all_files[0]
    update_files   = all_files[1:]

    print("[INFO] 全景背景文件：", os.path.basename(full_scan_file))
    print(f"[INFO] 共找到 {len(update_files)} 帧更新文件：")
    for fp in update_files:
        print("       ", os.path.basename(fp))

    # 2) 如需启用去噪，解除下面注释引入 denoiser
    # from src.denoiser import RadarDenoiser
    # denoiser = RadarDenoiser(kernel_size=3, min_neighbors=2)

    processor = RadarImageProcessor(shift_pixel=4)  # 或 RadarImageProcessor(shift_pixel=4, denoiser=denoiser)
    processor.process_files(full_scan_file, update_files)

    processor.show_interactive_image(cmap='hot')
    processor.save_image(output_path=data_dir)
