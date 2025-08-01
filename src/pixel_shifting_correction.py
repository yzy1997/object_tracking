import numpy as np
import matplotlib.pyplot as plt
import os
import mplcursors

class RadarImageProcessor:
    def __init__(self, resolution=(132, 132), shift_pixel=4):
        self.resolution = resolution
        self.shift_pixel = shift_pixel
        # 初始化时创建正确的存储矩阵（注意行列对应关系）
        self.radar_image = np.zeros((resolution[1], resolution[0]), dtype=np.float32)
    
    @staticmethod
    def parse_distance(byte4, byte5):
        return (byte4 << 8) | byte5 
    
    def read_radar_data(self, filepath):
        data = []
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
                    
                    # 坐标修正核心逻辑
                    # 1. 垂直翻转Y轴（矩阵坐标系转笛卡尔坐标系）
                    adjusted_y = self.resolution[1] - 1 - raw_y
                    
                    # 2. S型扫描校正（偶数行需要水平翻转）
                    if adjusted_y % 2 == 0:  # 修改判断条件为偶数行
                        adjusted_x = self.resolution[0] - 1 - raw_x + self.shift_pixel
                    else:
                        adjusted_x = self.resolution[0] - 1 - raw_x - self.shift_pixel
                    
                    if 0 <= adjusted_x < self.resolution[0] and 0 <= adjusted_y < self.resolution[1]:
                        data.append((adjusted_y, adjusted_x, distance))
                        
                except ValueError:
                    continue
        return data
    
    def update_image(self, data):
        # 直接填充到矩阵中
        for y, x, distance in data:
            self.radar_image[y, x] = distance
    
    def process_files(self, full_scan_file, update_files=None):
        # 处理完整扫描文件
        full_data = self.read_radar_data(full_scan_file)
        self.update_image(full_data)
        
        # 处理更新文件
        if update_files:
            for file in update_files:
                update_data = self.read_radar_data(file)
                self.update_image(update_data)
    
    def show_interactive_image(self, cmap='hot', vmin=None, vmax=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 自动确定颜色范围（排除0值）
        non_zero_values = self.radar_image[self.radar_image > 0]
        if len(non_zero_values) > 0:
            data_min = np.min(non_zero_values)
            data_max = np.max(non_zero_values)
            vmin = vmin or data_min
            vmax = vmax or data_max * 0.8
        
        # 注意：不再需要任何额外翻转
        img = ax.imshow(self.radar_image, 
                       cmap=cmap, 
                       origin='lower',  # 保持坐标系原点在左下角
                       vmin=vmin, 
                       vmax=vmax)
        
        plt.colorbar(img, label='Distance')
        ax.set_title('Corrected Radar Image (S-pattern Fixed)')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # 交互式光标调整
        cursor = mplcursors.cursor(img, hover=True)
        @cursor.connect("add")
        def on_add(sel):
            x = int(sel.target[0])
            y = int(sel.target[1])
            distance = self.radar_image[y, x]
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
        # 直接使用校正后的矩阵数据
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


# 使用示例
if __name__ == "__main__":
    # 文件路径配置
    data_dir = r"D:\codes\object_tracking\data\600m"
    full_scan_file = os.path.join(data_dir, "valid_framedata_0113_1.txt")
    update_files = [
        os.path.join(data_dir, "valid_framedata_0113_2.txt"),
        os.path.join(data_dir, "valid_framedata_0113_3.txt"),
        os.path.join(data_dir, "valid_framedata_0113_4.txt"),
        os.path.join(data_dir, "valid_framedata_0113_5.txt"),
        os.path.join(data_dir, "valid_framedata_0113_6.txt"),
    ]
    
    shift_pixel = 4 # 假设每行偏移4个像素
    processor = RadarImageProcessor(resolution=(132, 132))
    processor.process_files(full_scan_file, update_files)
    
    # 显示交互图像
    processor.show_interactive_image(cmap='hot')
    
    # 保存校正后的图像
    processor.save_image(output_path=data_dir)