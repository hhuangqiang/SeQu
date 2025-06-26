import time
import os
from options.test_options import TestOptions
#from data.data_loader import CreateDataLoader
from data import create_dataset
from models import create_model
import util.util3d as util
#from util.visualizer3d import Visualizer
from util.visualizer3d import save_images

from pdb import set_trace as st
from util import html
import VQGAN
import torch
from collections import OrderedDict
import numpy as np
import math

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip


def test_single_case(model, image, stride_xy=64, stride_z=24, patch_size=(128, 128, 48)):

    itr = 0
    _, __, ww, hh, dd = image.size() 

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((ww,hh,dd))
    cnt = np.zeros((ww,hh,dd))

    for x in range(sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(sy):
            ys = min(stride_xy*y, hh-patch_size[1])
            for z in range(sz):
                itr += 1
                zs = min(stride_z*z, dd-patch_size[2])
                test_patch = image[:, :, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]

            #print(test_patch.size(), (xs,xs+patch_size[0], ys,ys+patch_size[1], zs,zs+patch_size[2]))
                out, _, __ = model(test_patch)
                #out = test_patch
                out = out[0, 0, ...].detach().cpu().numpy()
                score_map[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += out
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1

    score_map = score_map / cnt # [Z, Y, X]
    score_map = np.transpose(score_map, (2,0,1))
    score_map = torch.from_numpy(score_map).unsqueeze(0).unsqueeze(0)
    score_map = score_map.to(device)
    print(itr)
    #print(score_map.shape)
    #while(1):True 
    return score_map



dataset = create_dataset(opt, phase='val')


device = "cuda"
model = VQGAN.VQGAN().to(device)

OCT_Type = 'B' # 'A' 'B' 
mode_save_path = "/home/eezzchen/TransPro/checkpoints/transpro_B_TestHualiangNoInfo/best_net_B.pth"
model.load_state_dict(torch.load(mode_save_path)) 
#model = create_model(opt)
#model.setup(opt) 

#visualizer = Visualizer(opt)
# create website

web_dir = os.path.join(opt.results_dir, opt.test_name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test

for i, data in enumerate(dataset):

    if i >= opt.num_test:
        break  
    input_data = data[OCT_Type].float()

    input_data_A = data['A'].float()

    input_data_model = np.transpose(input_data[0][0], (1, 2, 0))
    input_data_model = input_data_model.unsqueeze(0).unsqueeze(0)
    input_data_model = input_data_model.to(device)


    with torch.no_grad():
        fake_= test_single_case(model,input_data_model) #model(input_data) 
        
        real_ = input_data.to(device)
        real_A = input_data_A.to(device)

        #fake_proj = torch.mean(fake_,3)
        #real_proj = torch.mean(real_,3)

        fake_= util.tensor2im3d(fake_.data)
        real_ = util.tensor2im3d(real_.data)
        real_A = util.tensor2im3d(real_A.data)

        #fake_proj = util.tensor2im(fake_proj.data)
        #real_proj = util.tensor2im(real__proj.data)

    #print("66666")
    if OCT_Type == 'A':
        visuals = OrderedDict([('fake_A', fake_), ('real_A', real_)])
        img_path = data['A_paths']
    else:
        visuals = OrderedDict([('real_A', real_A),('fake_B', fake_), ('real_B', real_)])
        img_path = data['B_paths']
    #img_path = model.get_image_paths()
    #print("AAAA:",data['A_paths'])
    #print("BBBB:",data['B_paths'])
    print('process image... %s' % img_path)
    save_images(webpage, visuals, img_path)

print("Finish!!!!")
webpage.save()

# def get_current_visuals(self):
#     real_A = util.tensor2im3d(self.real_A.data)
#     fake_B = util.tensor2im3d(self.fake_B.data)
#     real_B = util.tensor2im3d(self.real_B.data)
#     if self.isTrain:
#         fake_B_proj = util.tensor2im(self.fake_B_proj_s.data)
#         real_B_proj = util.tensor2im(self.real_B_proj.data)
#         fake_B_seg = util.mask2im(self.fake_B_seg)
#         real_B_seg = util.mask2im(self.real_B_seg)
#         return OrderedDict([('fake_B', fake_B), ('real_B', real_B), ('fake_B_proj', fake_B_proj), ('real_B_proj', real_B_proj), ('fake_B_seg', fake_B_seg), ('real_B_seg', real_B_seg)])
#     return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

#python testVQGAN.py --dataroot  /home/eezzchen/OCT2OCTA3M_3D  --name transpro_3M_Hualiang --test_name transpro_3M_Hualiang --model TransPro --netG unet_256 --direction AtoB --lambda_A 10 --lambda_C 5 --dataset_mode alignedoct2octa3d --norm batch --input_nc 1 --output_nc 1 --gpu_ids 7 --num_test 15200 --which_epoch 194 --load_iter 194
#CUDA_VISIBLE_DEVICES=7 python testVQGAN.py --dataroot  /home/eezzchen/OCT2OCTA3M_3D  --name transpro_3M_Hualiang --test_name transpro_3M_Hualiang --model TransPro --netG unet_256 --direction AtoB --lambda_A 10 --lambda_C 5 --dataset_mode alignedoct2octa3d --norm batch --input_nc 1 --output_nc 1 --gpu_ids 7 --num_test 15200 --which_epoch 194 --load_iter 194