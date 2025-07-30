# main.py
import os
from src.pixel_shifting_correction import RadarImageProcessor
from src.denoise import RadarDenoiser
from src.object_detection import EnhancedDroneDetector

data_dir = r"D:\codes\object_tracking\data\600m"

full_scan_file = os.path.join(data_dir, "valid_framedata_0113_1.txt")
update_files = [
    os.path.join(data_dir, f"valid_framedata_0113_{i}.txt") 
    for i in range(2,7)  # 2-6号文件
]

def main():
    # 初始化处理器
    processor = RadarImageProcessor(resolution=(132, 132), shift_pixel=4)
    processor.process_files(full_scan_file, update_files)

    # 去噪处理
    denoiser = RadarDenoiser(filter_size=3, distance_threshold=100)
    processor.radar_image = denoiser.denoise(processor.radar_image)

    # 配置改进后的检测参数
    detector = EnhancedDroneDetector(
        processor=processor,
        update_files=update_files,
        y_threshold=40,
        eps=8,
        min_samples=1,
        temporal_window=5,
        speed_threshold=0.8,
        forbidden_zones=[(60, 80, 40, 60)],  # 调整后的禁止区域
        signal_adaptation=0.2
    )

    # 执行检测并保存结果
    boxes = detector.detect()
    detector.plot_result(boxes, "enhanced_detection_v2.png")

    # 更新验证用例
    test_cases = [
        {'file': full_scan_file, 'expected': 0},
        {'file': update_files[0], 'expected': 4},  # 预期检测4个目标
        {'file': update_files[1], 'expected': 4}
    ]
    
    validation_results = EnhancedDroneDetector.validate_detection(
        test_cases=test_cases,
        processor_class=RadarImageProcessor,
        y_threshold=40,
        eps=8,
        min_samples=1,
        speed_threshold=0.8,
        forbidden_zones=[(60, 80, 40, 60)]
    )
    
    print("\n验证结果:")
    for res in validation_results:
        if res['status'] == 'ERROR':
            print(f"{res['file']}: 错误 - {res['error']}")
        else:
            print(f"{res['file']}: 预期{res['expected']} 检出{res['detected']} [{res['status']}]")

if __name__ == "__main__":
    main()
