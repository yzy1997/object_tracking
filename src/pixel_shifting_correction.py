import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

class RadarProcessor:
    def __init__(self, resolution=(132, 132)):
        self.resolution = resolution
        self.speed_model = None
        self.raw_image = np.zeros(resolution[::-1], dtype=np.float32)
        self.corrected_image = np.zeros(resolution[::-1], dtype=np.float32)
    
    @staticmethod
    def parse_distance(byte4, byte5):
        """大端格式解析距离"""
        return (byte4 << 8) | byte5
    
    def load_data(self, filepath):
        """加载单个数据文件"""
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    y = int(parts[1], 16)
                    x = int(parts[2], 16)
                    distance = self.parse_distance(int(parts[3], 16), int(parts[4], 16))
                    if 0 <= x < self.resolution[0] and 0 <= y < self.resolution[1]:
                        data.append((self.resolution[1]-1-y, x, distance))  # Y轴翻转
                except ValueError:
                    continue
        return data
    
    def calibrate(self, calibration_files):
        """自动校准速度-偏移关系"""
        speeds, offsets = [], []
        
        for file in calibration_files:
            data = self.load_data(file)
            if len(data) < 2:
                continue
                
            # 计算扫描速度 (像素/行)
            x_coords = [x for (_,x,_) in data]
            speed = np.mean(np.abs(np.diff(x_coords)))
            
            # 估计偏移量（假设目标应为连续区域）
            unique_y = list(set(y for (y,_,_) in data))
            y_offsets = []
            for current_y in unique_y:
                x_values = [tx for (ty,tx,_) in data if ty == current_y]  # 修正变量名
                if len(x_values) > 1:
                    y_offsets.append(np.max(x_values) - np.min(x_values) - (len(x_values)-1))
            
            if y_offsets:
                speeds.append(speed)
                offsets.append(np.mean(y_offsets))
        
        if not speeds:
            print("警告：校准数据不足，使用默认模型")
            self.speed_model = lambda v: 0.3 * v
            return
        
        # 拟合模型 (二次函数)
        def model(v, a, b):
            return a * v**2 + b * v
        
        try:
            popt, _ = curve_fit(model, speeds, offsets, maxfev=5000)
            self.speed_model = lambda v: model(v, *popt)
        except:
            print("曲线拟合失败，使用线性模型")
            self.speed_model = lambda v: 0.3 * v
        
        # 显示校准结果
        plt.figure()
        v_range = np.linspace(0, max(speeds)*1.1, 100)
        plt.plot(speeds, offsets, 'ro', label='实测数据')
        plt.plot(v_range, self.speed_model(v_range), 'b-', label='拟合曲线')
        plt.xlabel('扫描速度 (像素/行)')
        plt.ylabel('像元走动量 (像素)')
        plt.title('自动校准曲线')
        plt.legend()
        plt.grid()
        plt.show()
    
    def build_image(self, data):
        """从数据点构建图像"""
        if not data:
            return np.zeros(self.resolution[::-1])
            
        points = np.array([(x,y) for (y,x,_) in data])
        values = np.array([d for (_,_,d) in data])
        grid_x, grid_y = np.mgrid[0:self.resolution[0], 0:self.resolution[1]]
        return griddata(points, values, (grid_x, grid_y), method='linear', fill_value=0)
    
    def correct_pixel_walk(self, data):
        """应用像元走动校正"""
        if not self.speed_model:
            raise ValueError("请先执行校准")
            
        if not data:
            return []
            
        # 检测扫描方向
        x_coords = [x for (_,x,_) in data]
        direction = 'left_right' if np.mean(np.diff(x_coords)) > 0 else 'right_left'
        
        # 计算扫描速度
        speed = np.mean(np.abs(np.diff(x_coords)))
        offset = max(0, self.speed_model(speed))  # 确保偏移量为正
        
        # 应用校正
        corrected = []
        for y, x, distance in data:
            if direction == 'left_right':
                new_x = int(round(x - offset))
            else:
                new_x = int(round(x + offset))
            new_x = np.clip(new_x, 0, self.resolution[0]-1)
            corrected.append((y, new_x, distance))
            
        return corrected
    
    def process_files(self, files):
        """处理所有文件"""
        all_data = []
        for file in files:
            if not os.path.exists(file):
                print(f"文件不存在: {file}")
                continue
            all_data.extend(self.load_data(file))
        
        if not all_data:
            print("错误：没有加载到有效数据")
            return
        
        # 原始图像
        self.raw_image = self.build_image(all_data)
        
        # 校正后图像
        corrected_data = self.correct_pixel_walk(all_data)
        self.corrected_image = self.build_image(corrected_data)
        
        # 显示结果
        self.show_results()
    
    def show_results(self):
        """显示矫正效果对比"""
        plt.figure(figsize=(14, 6))
        
        plt.subplot(121)
        plt.imshow(np.fliplr(self.raw_image.T), cmap='hot', origin='lower')
        plt.colorbar(label='距离值')
        plt.title('原始雷达图像')
        plt.xlabel('X坐标')
        plt.ylabel('Y坐标')
        
        plt.subplot(122)
        plt.imshow(np.fliplr(self.corrected_image.T), cmap='hot', origin='lower')
        plt.colorbar(label='距离值')
        plt.title('像元走动矫正后')
        plt.xlabel('X坐标')
        
        plt.tight_layout()
        plt.savefig('radar_correction_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

# 使用真实数据文件
if __name__ == "__main__":
    # 文件路径配置
    data_dir = r"D:\codes\object_tracking\data\600m"
    files = [os.path.join(data_dir, f"valid_framedata_0113_{i}.txt") for i in range(1, 7)]
    
    # 检查文件是否存在
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print("错误：未找到任何数据文件")
        exit()
    
    processor = RadarProcessor()
    
    # 使用前2个文件自动校准（至少需要2个文件）
    if len(files) >= 2:
        processor.calibrate(files[:2])
    else:
        print("警告：文件不足，使用默认校正模型")
        processor.speed_model = lambda v: 0.3 * v
    
    # 处理所有文件并显示结果
    processor.process_files(files)