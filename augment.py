import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def simple_flip_augmentation(input_dir, output_dir):
    """
    仅进行水平和垂直翻转的数据增强
    :param input_dir: 输入目录路径
    :param output_dir: 输出目录路径
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 支持的图像格式
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']

    # 获取所有图像文件
    img_files = [f for f in Path(input_dir).glob('*') if f.suffix.lower() in valid_exts]

    # 定义翻转变换
    flip_transforms = [
        ('flip_h', lambda img: cv2.flip(img, 1)),  # 水平翻转
        ('flip_v', lambda img: cv2.flip(img, 0)),  # 垂直翻转
    ]

    for img_path in tqdm(img_files, desc="Processing flips"):
        try:
            # 读取图像（保留Alpha通道）
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"警告：无法读取图像 {img_path}")
                continue

            # 应用并保存翻转结果
            for suffix, transform in flip_transforms:
                flipped_img = transform(img)

                # 生成新文件名
                new_name = f"{img_path.stem}_{suffix}{img_path.suffix}"
                output_path = Path(output_dir) / new_name

                # 保存图像（保留原始色彩空间）
                cv2.imwrite(str(output_path), flipped_img)

        except Exception as e:
            print(f"处理 {img_path} 时发生错误: {str(e)}")


if __name__ == "__main__":
    input_dir = "/home/data2/haungqiang/code/defect/data/CFD/train/1"
    output_dir = "/home/data2/haungqiang/code/defect/data/CFD/train/2"
    simple_flip_augmentation(input_dir, output_dir)