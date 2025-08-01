# main.py
import os
import glob
import matplotlib.pyplot as plt

from src.pixel_shifting_correction import RadarImageProcessor
from src.object_detection      import SpatialDroneDetector

def main():
    # —— 修改成你自己的数据文件夹 —— #
    data_folder = "data/600m"
    # 输出图片目录
    out_folder = "pics"
    os.makedirs(out_folder, exist_ok=True)

    # 1) 动态读取所有 txt 文件并排序
    all_files = sorted(glob.glob(os.path.join(data_folder, "*.txt")))
    if len(all_files) < 2:
        raise RuntimeError(f"在 {data_folder} 下至少需要两个 txt 文件 (一个全景背景 + 至少一帧)。")

    # 第一个当背景，其它当作每一帧
    full_scan   = all_files[0]
    frame_files = all_files[1:]

    print(f"检测到背景文件: {full_scan}")
    print(f"检测到 {len(frame_files)} 帧数据文件，准备逐帧生成图片并保存到 {out_folder}")

    # 2) 创建 RadarImageProcessor（只需传 shift_pixel）
    proc = RadarImageProcessor(shift_pixel=4)

    # 3) 先创建一次检测器，内部会一次性将全景背景读入
    detector = SpatialDroneDetector(
        processor      = proc,
        full_scan_file = full_scan,
        update_files   = [],        # 先留空，后面循环里再按帧更新
        use_otsu       = False,     # 关闭 Otsu，改用百分位阈值
        thr_percentile = 15,        # 相当于你原来想要的 thr_ratio=0.15
        median_size    = 3,
        tophat_radius  = 5,
        min_area       = 3,
        max_area       = 50
    )

    # 4) 逐帧处理
    for idx, frame_path in enumerate(frame_files, start=1):
        print(f"[Frame {idx:03d}] 处理文件 {frame_path} ...")
        # 更新 detector 使用的这一帧
        detector.update_files = [frame_path]

        # 检测无人机，返回若干 (x0,x1,y0,y1)
        boxes = detector.detect()
        print(f"  检测到 {len(boxes)} 个候选框: {boxes}")

        # 重新读一遍当前累积图像（背景 + 这一帧）
        img = detector.build_accumulated_image()

        # 可视化并保存
        plt.figure(figsize=(6,6))
        plt.imshow(img, cmap="jet", origin="lower")
        ax = plt.gca()
        for x0, x1, y0, y1 in boxes:
            w, h = x1 - x0, y1 - y0
            ax.add_patch(plt.Rectangle(
                (x0, y0), w, h,
                edgecolor="yellow", facecolor="none", lw=1.5
            ))
        plt.title(f"Frame {idx:03d}")
        plt.axis("off")

        out_png = os.path.join(out_folder, f"frame_{idx:03d}.png")
        plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"  已保存: {out_png}")

    print("全部帧处理完毕.")

if __name__ == "__main__":
    main()
