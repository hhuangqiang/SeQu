import time
import os
import cv2
from options.test_options import TestOptions
# from data.data_loader import CreateDataLoader
from data import create_dataset
from models import create_model
import util.util as util
# from util.visualizer3d import Visualizer
from util.visualizer3d import save_images
import ntpath
from pdb import set_trace as st
from util import html
import VQGAN
import AE
from PIL import Image
import torch
from collections import OrderedDict
import numpy as np
import math
from skimage.metrics import structural_similarity as sk_cpt_ssim
from torchvision.utils import make_grid, save_image
from PIL import Image
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.stats import entropy


def norm_(proj_map):
    _min = proj_map.min()
    _max = proj_map.max()
    if _min == _max:
        print("********")
        return 0
    return (proj_map - _min) / (_max - _min)


def MAE(fake, real):
    # mae = 0.0
    # assert len(fake) == len(real)
    x, y = np.where(real != 0)  # Exclude background
    mae = (np.abs(fake[x, y] - real[x, y])).mean()
    # mae = np.abs(fake-real).mean()
    return mae
    # return mae/(2.0 * len(fake))     #from (-1,1) normaliz  to (0,1)


def MSE(fake, real):
    # mae = 0.0
    # assert len(fake) == len(real)
    x, y = np.where(real != 0)  # Exclude background
    mse = np.square(fake[x, y] * 255 - real[x, y] * 255).mean()
    # mae = np.abs(fake-real).mean()
    return mse
    # return mae/(2.0 * len(fake))     #from (-1,1) normaliz  to (0,1)


def PSNR(fake, real):
    x, y = np.where(real != 0)  # Exclude background
    # mse = np.mean(((fake[x][y]+1)/2. - (real[x][y]+1)/2.) ** 2 )
    # mse = np.mean(((fake+1)/2. - (real+1)/2.) ** 2 )
    mse = np.mean(((fake[x][y]) - (real[x][y])) ** 2)
    if mse < 1.0e-10:
        return 100
    else:
        PIXEL_MAX = 1
        return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def visualize_pixel_diff(img1, img2, title1="Image 1", title2="Image 2", save_path=None):
    """
    可视化两幅图像及其像素级差异热力图（支持RGB）
    Args:
        img1: 图像1数组 [H, W, C]，类型为 numpy.ndarray，取值范围 [0,255]
        img2: 图像2数组 [H, W, C]，类型为 numpy.ndarray，取值范围 [0,255]
        title1: 图像1标题（默认 "Image 1"）
        title2: 图像2标题（默认 "Image 2"）
        save_path: 保存路径（若为None则直接显示）
    """
    # 转换为浮点数并归一化到 [0,1]
    img1_norm = img1.astype(np.float32) / 255.0
    img2_norm = img2.astype(np.float32) / 255.0

    # 计算逐像素 L1 差异（各通道平均）
    pixel_diff = np.mean(np.abs(img1_norm - img2_norm), axis=2)  # [H, W]

    # 绘制图像
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 图像1
    axes[0].imshow(img1_norm)
    axes[0].set_title(title1)
    axes[0].axis('off')

    # 图像2
    axes[1].imshow(img2_norm)
    axes[1].set_title(title2)
    axes[1].axis('off')

    # 差异热力图（jet颜色映射）
    im = axes[2].imshow(pixel_diff, cmap='jet', vmin=0, vmax=0.5)  # 调整vmax为数据范围
    axes[2].set_title("Pixel-wise L1 Difference")
    axes[2].axis('off')

    # 添加颜色条
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


opt = TestOptions().parse()
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batch_size = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

dataset = create_dataset(opt, phase='test')

device = "cuda"
# model_NOise2Clean = VQGAN.VQGAN().to(device) #VQGAN.VQGAN().to(device)
# model = VQGAN.VQGAN().to(device) #VQGAN.VQGAN().to(device)

model_type = "VQGAN"  # VAE VQGAN

if model_type == "AE":
    model = AE.AE().to(device)
elif model_type == "VQGAN":
    model = VQGAN.VQGAN().to(device)

# model = VQGAN.VQGAN().to(device) #VQGAN.VQGAN().to(device)
In_Type = 'A'  # 'A' 'B'
Out_Type = 'B'  # 'A' 'B'
OCT_Type = In_Type + Out_Type
Thesold = 250

Lora_Flag = False

if Lora_Flag == True:
    mode_save_path = "/home/data2/haungqiang/code/defect/checkpoints/transpro_A/best_net_BA.pth"
    mode_save_lora_path = "/home/eezzchen/TransPro/checkpoints/transpro_OCT2OCTA_LoRa_PreB/best_LoRa_B.pth"
    # model.load_state_dict(torch.load(mode_save_path))
    # Load the pretrained checkpoint first
    model.load_state_dict(torch.load(mode_save_path), strict=False)
    # Then load the LoRA checkpoint
    model.load_state_dict(torch.load(mode_save_lora_path), strict=False)
else:
    # mode_save_path = "/mnt/gdut/sam_zgm/C_zgm/DenoisePro/checkpoints/Noise2CleanAligment_Encoder/best_net_AB.pth"
    # model.load_state_dict(torch.load(mode_save_path))

    mode_save_path = "/home/data2/haungqiang/code/defect/checkpoints/concrete_check/latest_net_BA.pth"
    model.load_state_dict(torch.load(mode_save_path))

    # CUDA_VISIBLE_DEVICES=1 python testVQGAN_OCT2OCTA.py --dataroot  /mnt/gdut/sam_zgm/C_zgm/Dataset/OCTA_Denoise --name transpro_3M_Hualiang --test_name N2C_Ali_10  --model TransPro --netG unet_256 --direction AtoB --lambda_A 10 --lambda_C 5 --dataset_mode alignedoct2octa3d --norm batch --input_nc 1 --output_nc 1 --gpu_ids 0 --num_test 15200 --which_epoch 194 --load_iter 194

web_dir = os.path.join(opt.results_dir, opt.test_name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
# codebook_losses = [[] for _ in range(4)]  # 4个Codebook
list_MAE_ = []
list_MSE_ = []
list_PSNR_ = []
list_SSIM_ = []
list_precision = []
list_recall = []
list_accuracy = []
list_f1 = []
list_iou = []

for i, data in enumerate(dataset):

    if i >= opt.num_test:
        break

    input_data_A = data[In_Type].float()  # [0]
    input_data_A = input_data_A.to(device)

    input_data_B = data[Out_Type].float()  # [0]
    input_data_B = input_data_B.to(device)

    clean_data = data['B'].float().to(device)
    label_tensor = data['label'].float().to(device)
    label_np = util.tensor2im(label_tensor.data)  # 转换为numpy格式 [H,W,3]
    # 确保标签为单通道二值图像
    if label_np.ndim == 3:
        # 处理三通道RGB或冗余的单通道维度（如H,W,1）
        if label_np.shape[2] == 3:
            # 三通道转单通道灰度
            label_gray = cv2.cvtColor(label_np, cv2.COLOR_RGB2GRAY)
        else:
            # 移除单通道的冗余维度（例如[H,W,1 -> H,W]）
            label_gray = label_np.squeeze(axis=2)
    else:
        label_gray = label_np.astype(np.uint8)
    label_mask = (label_gray > 128).astype(np.uint8) * 255

    with torch.no_grad():
        # fake_= test_single_case(model,input_data_model) #model(input_data)
        # fake_,_, latent_loss = model(input_data_A)
        fake_, _, latent_loss, _, _, quant_error_maps = model(input_data_A)  # input_data_A input_data_B
    # 将每个层的误差图映射到原始图像尺寸
    original_height, original_width = input_data_A.shape[2], input_data_A.shape[3]
    layer_names = ["Layer1", "Layer2", "Layer3", "Layer4"]
    image_dir = webpage.get_image_dir()
    image_path = data['B_paths']
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    error_maps = []
    # 为每个测试样本生成误差图
    for layer_idx in range(4):
        # 获取当前层的误差图 [H, W]
        error_map = quant_error_maps[layer_idx][0].numpy()  # 假设 batch_size=1

        # 上采样到原始图像尺寸
        error_map_resized = Image.fromarray(error_map).resize(
            (original_width, original_height),
            resample=Image.BILINEAR  # 使用双线性插值
        )
        error_map_resized = np.array(error_map_resized)

        # 归一化到 [0, 1]
        error_map_normalized = (error_map_resized - error_map_resized.min()) / (
                error_map_resized.max() - error_map_resized.min() + 1e-8)
        error_maps.append(error_map_normalized)
        # 生成热力图
        plt.figure(figsize=(10, 8))
        plt.imshow(error_map_normalized, cmap='jet', alpha=0.7)  # jet 颜色映射
        plt.colorbar(label='Quantization Error (L2 Distance)')
        plt.title(f'Pixel-Level Quantization Error - {layer_names[layer_idx]}')
        plt.axis('off')

        # 保存图像
        save_path = os.path.join(web_dir, f'sample_{name}_layer_{layer_idx}_error.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

        real_A = input_data_A.to(device)
        real_B = input_data_B.to(device)

        # fake_proj = torch.mean(fake_,3)
        # real_proj = torch.mean(real_,3)

        fake_ = util.tensor2im(fake_)

        # x,y,z = np.where( fake_ > Thesold )
        # fake_[x,y,z] = 0

        real_A = util.tensor2im(real_A.data)
        real_B = util.tensor2im(real_B.data)
        clean = util.tensor2im(clean_data.data)

    visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_), ('real_B', real_B)])
    image_path = data['B_paths']

    print('process image... %s' % image_path)

    norm_fake_0, norm_real_0 = norm_(fake_[:, :, 0]), norm_(clean[:, :, 0])
    norm_fake_1, norm_real_1 = norm_(fake_[:, :, 1]), norm_(clean[:, :, 1])
    norm_fake_2, norm_real_2 = norm_(fake_[:, :, 2]), norm_(clean[:, :, 2])

    mae = MAE(norm_fake_0, norm_real_0) + MAE(norm_fake_1, norm_real_1) + MAE(norm_fake_2, norm_real_2)
    mse = MSE(norm_fake_0, norm_real_0) + MSE(norm_fake_1, norm_real_1) + MSE(norm_fake_2, norm_real_2)
    psnr = PSNR(norm_fake_0, norm_real_0) + PSNR(norm_fake_1, norm_real_1) + PSNR(norm_fake_2, norm_real_2)

    print("PSNR:", PSNR(norm_fake_0, norm_real_0), PSNR(norm_fake_1, norm_real_1), PSNR(norm_fake_2, norm_real_2))
    ssim_1, diff_1 = sk_cpt_ssim(norm_fake_0, norm_real_0, data_range=1.0, full=True)
    ssim_2, diff_2 = sk_cpt_ssim(norm_fake_1, norm_real_1, data_range=1.0, full=True)
    ssim_3, diff_3 = sk_cpt_ssim(norm_fake_2, norm_real_2, data_range=1.0, full=True)

    print("SSIM:", ssim_1, ssim_2, ssim_3)
    list_MAE_.append(mae / 3.0)
    list_MSE_.append(mse / 3.0)
    list_PSNR_.append(psnr / 3.0)
    list_SSIM_.append((ssim_1 + ssim_2 + ssim_3) / 3.0)

    # save_images(webpage, visuals, img_path)
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    for label, image_numpy in visuals.items():
        image_name = '%s_%s.jpg' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        print("save_path:", save_path)
        # print("img numpy size", image_numpy.shape) #256x256x256
        util.save_image(image_numpy, save_path)
    real_A_np = np.array(util.tensor2im(input_data_A.data))  # 测试输入real_A
    fake_B_np = np.array(util.tensor2im(fake_.data))  # 重构图像（fake_B）
    real_B_np = np.array(util.tensor2im(input_data_B.data))
    recon_error = np.mean(np.abs(real_B_np - fake_B_np), axis=2)
    # 归一化到 [0,1]
    recon_error_norm = (recon_error - recon_error.min()) / (recon_error.max() - recon_error.min() + 1e-8)
    error_maps.append(recon_error_norm)
    # 生成差异图并保存
    diff_save_path = os.path.join(image_dir, f'sample_{name}_realA_fakeB_pixel_diff.png')
    visualize_pixel_diff(fake_B_np, real_B_np,
                         title1="Reconstructed Image (fake_B)",
                         title2="Ground Truth (real_B)",
                         save_path=diff_save_path)


    def gmm_defect_detection(error_maps, label_mask, image_dir, name):
        """
        使用GMM进行多特征缺陷检测
        Args:
            error_maps: 包含5个误差图的列表 (layer1-4 + recon_error)
            label_mask: 真实标签二值掩模 (H,W)
            image_dir: 结果保存路径
            name: 样本名称
        Returns:
            pred_mask: 预测的二值掩模 (H,W) 0/255
        """
        # 1. 特征工程
        # 堆叠多尺度误差图 [H, W, 5]
        feature_matrix = np.stack(error_maps, axis=2)
        h, w, c = feature_matrix.shape

        # 重塑为二维特征矩阵 [N_pixels, 5_features]
        X = feature_matrix.reshape(-1, c)

        # 2. 数据预处理
        # 标准化每个特征到[0,1]
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(
            n_components=2,  # 缺陷/正常两类
            covariance_type='diag',  # 对角协方差加速计算
            init_params='kmeans',  # 改进的初始化
            max_iter=200,
            n_init=3,
            random_state=42,
            reg_covar=1e-6
        )
        gmm.fit(X_scaled)

        # 4. 缺陷概率计算
        # 获取每个像素属于各分量的概率 [N, 2]
        probs = gmm.predict_proba(X_scaled)
        print("probs shape:", probs.shape)  # 应为 (N_samples, 2)
        # 确定缺陷类别（假设均值较大的分量对应缺陷）
        if gmm.means_[:, 0].mean() > gmm.means_[:, 1].mean():
            defect_cls = 0
        else:
            defect_cls = 1

        # 生成缺陷概率图 [H, W]
        defect_prob = probs[:, defect_cls].reshape(h, w)
        defect_prob = np.clip(defect_prob, 1e-10, 1.0)
        # 5. 自适应阈值分割
        from skimage.filters import threshold_otsu
        from skimage.filters import gaussian
        smooth_prob = gaussian(defect_prob, sigma=1)
        thresh = threshold_otsu(smooth_prob)
        pred_mask = (smooth_prob >= thresh).astype(np.uint8) * 255

        # 6. 后处理
        # 形态学闭运算填充小孔
        d = 15  # 邻域直径
        sigmaColor = 300  # 颜色空间标准差
        sigmaSpace = 25  # 空间域标准差
        # 应用滤波
        bilateral = cv2.bilateralFilter(pred_mask, d, sigmaColor, sigmaSpace)
        # Otsu二值化
        _, otsu_mask = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 7. 保存中间结果
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(defect_prob, cmap='jet')
        plt.title('Defect Probability')
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.imshow(bilateral, cmap='gray')
        plt.title('Initial Prediction')

        # 叠加显示
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(real_B_np, cv2.COLOR_RGB2GRAY), cmap='gray')
        plt.imshow(defect_prob, cmap='jet', alpha=0.5)
        plt.title('Overlay')

        plt.savefig(os.path.join(image_dir, f'sample_{name}_gmm_results.png'),
                    bbox_inches='tight', dpi=300)
        plt.close()

        return otsu_mask


    # 执行GMM缺陷检测
    pred_mask = gmm_defect_detection(error_maps, label_mask, image_dir, name)
    print(pred_mask.min(), pred_mask.max())  # 查看预测 mask 值范围
    print(label_mask.min(), label_mask.max())  # 查看标签 mask 值范围

    # 确保预测和标签尺寸一致
    assert pred_mask.shape == label_mask.shape[:2], "预测与标签尺寸不匹配"

    # 展平为1D数组（0或1）
    pred_flat = (pred_mask.flatten() > 128).astype(np.uint8)
    label_flat = (label_mask.flatten() > 128).astype(np.uint8)

    # 计算TP, FP, TN, FN
    TP = np.sum((pred_flat == 1) & (label_flat == 1))
    FP = np.sum((pred_flat == 1) & (label_flat == 0))
    TN = np.sum((pred_flat == 0) & (label_flat == 0))
    FN = np.sum((pred_flat == 0) & (label_flat == 1))

    # 计算指标（避免除以0）
    precision = TP / (TP + FP + 1e-10) * 100
    recall = TP / (TP + FN + 1e-10) * 100
    accuracy = (TP + TN) / (TP + FP + TN + FN + 1e-10) * 100
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    iou = TP / (TP + FP + FN + 1e-10) * 100

    # 保存当前样本的指标
    list_precision.append(precision)
    list_recall.append(recall)
    list_accuracy.append(accuracy)
    list_f1.append(f1)
    list_iou.append(iou)

mean_MAE = sum(list_MAE_) / len(list_MAE_)
mean_MSE = sum(list_MSE_) / len(list_MSE_)
mean_PSNR = sum(list_PSNR_) / len(list_PSNR_)
mean_SSIM = sum(list_SSIM_) / len(list_SSIM_)
# mean_FSIM = FSIM_/num
mean_precision = np.mean(list_precision)
mean_recall = np.mean(list_recall)
mean_accuracy = np.mean(list_accuracy)
mean_f1 = np.mean(list_f1)
mean_iou = np.mean(list_iou)

print("Mean Precision (%):", mean_precision)
print("Mean Recall (%):", mean_recall)
print("Mean Accuracy (%):", mean_accuracy)
print("Mean F1-Score (%):", mean_f1)
print("Mean IoU (%):", mean_iou)

print("Mean MAE:", mean_MAE)
print("Mean MSE:", mean_MSE)
print("Mean PSNR:", mean_PSNR)
print("Mean SSIM:", mean_SSIM)

print("Finish!!!!")
webpage.save()