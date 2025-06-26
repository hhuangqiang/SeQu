import os
import cv2
import numpy as np
from tqdm import tqdm
from stride_augmentation import *
from checkboard_augmentation import *

def process_dataset(input_dir, output_dir, mask_type='checkerboard'):#checkerboard
    """
    生成与原始文件同名的破坏图像和掩码
    :param input_dir: 输入目录路径（如 /A_Noise）
    :param output_dir: 输出根目录（如 /B_Clean）
    """
    # 创建输出目录结构
    corrupted_dir = os.path.join(output_dir, 'B_Clean1')
    #masks_dir = os.path.join(output_dir, 'B_Clean/masks')
    os.makedirs(corrupted_dir, exist_ok=True)
    #os.makedirs(masks_dir, exist_ok=True)

    # 获取所有PNG文件
    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]

    for filename in tqdm(img_files):
        # 读取原始图像
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)

        # 生成掩码（假设mask函数返回0-1矩阵）
        if mask_type == 'stride':
            mask = random_stride_mask(img, None)
        else:
            mask = random_checkboard_mask_new(img, None)

        # 生成破坏图像
        corrupted_img = img * mask

        # 转换掩码为uint8格式（0-255）
        mask = (mask * 255).astype(np.uint8)

        # 保持原始文件名直接保存
        cv2.imwrite(os.path.join(corrupted_dir, filename), corrupted_img)  # 破坏图像
        #cv2.imwrite(os.path.join(masks_dir, filename), mask)  # 掩码文件


if __name__ == "__main__":
    # 示例路径
    input_folder = "/home/data2/haungqiang/code/defect/data/concrete3k/test/A_Noise/"
    output_folder = "/home/data2/haungqiang/code/defect/data/concrete3k/test/"

    # 运行处理
    process_dataset(input_folder, output_folder)