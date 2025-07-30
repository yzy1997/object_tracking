import numpy as np
from scipy import ndimage

class RadarDenoiser:
    def __init__(self, filter_size=3, distance_threshold=100):
        self.filter_size = filter_size
        self.distance_threshold = distance_threshold

    def denoise(self, radar_image):
        """应用去噪处理"""
        # 中值滤波消除孤立噪声点
        denoised = ndimage.median_filter(radar_image, size=self.filter_size)
        
        # 距离阈值过滤
        denoised[denoised < self.distance_threshold] = 0
        
        # 形态学闭运算填充小孔
        structure = np.ones((3,3), dtype=np.int32)
        denoised = ndimage.grey_closing(denoised, structure=structure)
        
        return denoised

def test_denoise():
    import matplotlib.pyplot as plt
    from pixel_shifting_correction import RadarImageProcessor
    
    processor = RadarImageProcessor()
    # ...文件路径需要根据实际情况配置...
    
    denoiser = RadarDenoiser()
    denoised_image = denoiser.denoise(processor.radar_image)
    
    plt.imshow(denoised_image, cmap='hot', origin='lower')
    plt.colorbar()
    plt.show()
