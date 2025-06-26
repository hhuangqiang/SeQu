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
import VQVAE
import torch
from collections import OrderedDict

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

#data_loader = CreateDataLoader(opt)
#dataset = data_loader.load_data()

dataset = create_dataset(opt, phase='test')

device = "cuda"
model = VQVAE.VQVAE().to(device)

OCT_Type = 'B' # 'A' 'B' 
mode_save_path = "/home/eezzchen/TransPro/checkpoints/transpro_B_Test1/best_net_B.pth"
model.load_state_dict(torch.load(mode_save_path)) 
#model = create_model(opt)
#model.setup(opt) 

#visualizer = Visualizer(opt)
# create website

web_dir = os.path.join(opt.results_dir, opt.test_name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test

for i, data in enumerate(dataset):

    if i >= opt.num_test:
        break  
    input_data = data[OCT_Type].float().to(device)

    with torch.no_grad():

        fake_A, _ = model(input_data)
        real_A = input_data

        fake_A_proj = torch.mean(fake_A,3)
        real_A_proj = torch.mean(real_A,3)

        fake_A = util.tensor2im3d(fake_A.data)
        real_A = util.tensor2im3d(real_A.data)

        fake_A_proj = util.tensor2im(fake_A_proj.data)
        real_A_proj = util.tensor2im(real_A_proj.data)

    #print("66666")
    if OCT_Type == 'A':
        visuals = OrderedDict([('fake_A', fake_A), ('real_A', real_A)])
        img_path = data['A_paths']
    else:
        visuals = OrderedDict([('fake_B', fake_A), ('real_B', real_A)])
        img_path = data['B_paths']
    #img_path = model.get_image_paths()

    print('process image... %s' % img_path)
    save_images(webpage, visuals, img_path)

webpage.save()

#python testVQVAE.py --dataroot  /home/eezzchen/OCT2OCTA3M_3D  --name transpro_3M_B --test_name transpro_3M_B --model TransPro --netG unet_256 --direction AtoB --lambda_A 10 --lambda_C 5 --dataset_mode alignedoct2octa3d --norm batch --input_nc 1 --output_nc 1 --gpu_ids 7 --num_test 15200 --which_epoch 194 --load_iter 194

