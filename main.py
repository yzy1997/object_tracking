# main.py

import os
from src.pixel_shifting_correction import RadarImageProcessor
from src.object_detection import SpatialDroneDetector

def main():
    # 1. 配置数据路径
    data_dir = r"D:\codes\object_tracking\data\600m"
    full_scan_file = os.path.join(data_dir, "valid_framedata_0113_1.txt")
    # 假设一共 6 帧增量数据，编号 2~6
    update_files = [
        os.path.join(data_dir, f"valid_framedata_0113_{i}.txt")
        for i in range(2, 7)
    ]

    # 2. 初始化雷达图像处理器
    #    resolution=(132,132) 和 shift_pixel=4 只是示例，请根据自己的数据调整
    proc = RadarImageProcessor(resolution=(132, 132), shift_pixel=4)
    # 先让 processor 读一次全扫描，以便初始化内部的 radar_image 大小
    proc.process_files(full_scan_file, [])

    # 3. 初始化检测器
    detector = SpatialDroneDetector(
        processor=proc,
        full_scan_file=full_scan_file,
        update_files=update_files,

        # 中值滤波核大小
        median_size=3,
        # 是否使用 Otsu 自动阈值，若为 False 则使用 thr_ratio
        use_otsu=True,
        thr_ratio=0.5,        # 手动阈值比例（0~1），仅当 use_otsu=False 时生效
        thr_percentile=85,    # 百分位阈值，亦可根据需要调整

        # 连通域过滤参数
        min_area=2,
        max_area=50,
        max_width=10,
        max_height=10,
        # min_y 用于过滤地面大楼噪声（行坐标小于该值的连通域忽略）
        min_y=70,

        # 目标数（无人机数量）
        n_targets=4,
        # 峰值补齐参数
        peak_min_distance=3,
        peak_pad=1
    )

    # 4. 构造累积图 & 执行检测
    accumulated = detector.build_accumulated_image()
    boxes = detector.detect()

    # 5. 输出结果
    print("累积图像形状：", accumulated.shape)
    print("检测到的框 (xmin, xmax, ymin, ymax)：", boxes)

    # 6. （可选）可视化结果
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(accumulated, cmap='inferno', origin='lower')
        for xmin, xmax, ymin, ymax in boxes:
            rect = plt.Rectangle(
                (xmin, ymin),
                xmax - xmin + 1,
                ymax - ymin + 1,
                edgecolor='yellow',
                facecolor='none',
                linewidth=2
            )
            ax.add_patch(rect)
        ax.set_title("Detected Drones")
        plt.savefig('detected_drones.png')
    except ImportError:
        # 没安装 matplotlib 时跳过可视化
        pass

if __name__ == "__main__":
    main()
