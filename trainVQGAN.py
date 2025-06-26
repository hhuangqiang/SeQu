import time
from options.train_options import TrainOptions
from util.visualizer3d import Visualizer
from data import create_dataset
from models import create_model
import torch
import numpy as np
import VQGAN
from argparse import ArgumentParser, Namespace
from scheduler import CycleScheduler,LRFinder
from torch import nn, optim
import util.util3d as util
from util.image_pool import ImagePool
from collections import OrderedDict
import random
import torch.nn.functional as F
import math 

def MAE_(fake,real):
    mae = 0.0
    mae = np.mean(np.abs(fake-real))
    return mae

def Norm(a):
    max_ = torch.max(a)
    min_ = torch.min(a)
    a_0_1 = (a-min_)/(max_-min_)
    return (a_0_1-0.5)*2



opt = TrainOptions().parse()
dataset = create_dataset(opt, phase="train")  # create a dataset given opt.dataset_mode and other options
dataset_size = len(dataset)
print('#training images = %d' % dataset_size)


val_dataset = create_dataset(opt, phase="val") 
val_dataset_size = len(val_dataset)
print('#validation images = %d' % val_dataset_size)

#define VQ-VAE
#model = create_model(opt)
#model.setup(opt)       
device = "cuda"
model = VQGAN.VQGAN().to(device)
lr = 3e-4 #3e-4
In_Type = 'A' # 'A' 'B' 
label_Type = 'B'
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler_type = "cycle"

print("In_Type:",In_Type)
#scheduler_type = "LRFinder"
if scheduler_type == "cycle":
    scheduler = CycleScheduler(
        optimizer,
        lr,
        n_iter=len(dataset) * (opt.n_epochs + opt.n_epochs_decay + 1),
        momentum=None,
        warmup_proportion=0.05,
    )
elif scheduler_type == "LRFinder":
    scheduler = LRFinder(
        optimizer, 
        lr_min = lr*0.001, 
        lr_max = lr, 
        step_size =50, 
        linear=True
    )

#visualizer = Visualizer(opt)
total_steps = 0
val_total_iters = 0 

global_mae = 100000000000000
criterion = nn.MSELoss()
latent_loss_weight = 0.25


for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
    epoch_start_time = time.time()
    iter_start_time = time.time()
    epoch_iter = 0
    #if "adap" in opt.name:
    #    model.update_weight_alpha()
    Train_MAE = 0
    Train_num = 0

    # A noise B clean

    for i, data in enumerate(dataset):
        train_data = data[In_Type].float()  
        label_data = data[label_Type].float().to(device)

        #input_data = np.transpose(input_data[0][0], (0, 2, 1))
        #train_data, info = crop(input_data, h=128, l=256, train=True) 

        #print(train_data.shape)
        #train_data = train_data.unsqueeze(0)
        train_data =train_data.to(device)
        # info = torch.Tensor([info])
        # info  = info.to(device)

        total_steps += opt.batch_size
        epoch_iter += opt.batch_size
        model.zero_grad()
        out,_, latent_loss,_,_ = model(train_data) 

        # recon_loss = criterion(out, train_data)
        recon_loss = criterion(out, label_data)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        cur_mae = MAE_(out.detach().cpu().numpy(),label_data.cpu().numpy())
        Train_MAE += cur_mae
        Train_num += 1

        if scheduler is not None:
            scheduler.step()
        optimizer.step()
        #print(loss)
    
    print("Train MAE:",Train_MAE/Train_num)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        #model.save('latest')
        model.save_network(model, In_Type, 'latest',opt,device) 
    

    if epoch % opt.val_epoch_freq == 0: 
        device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        with torch.no_grad():
            MAE = 0 
            num = 0 
            for i, data in enumerate(val_dataset):
                gt_data = data[In_Type].float()
                gt_data = gt_data.to(device)
                fake_,_, latent_loss,_,_ = model(gt_data)
                #input_data = np.transpose(gt_data[0][0], (0, 2, 1))
                #input_data = input_data.unsqueeze(0).unsqueeze(0)
                #input_data = input_data.to(device)

                #fake_= test_single_case(model,input_data) #model(input_data) 

                # train_data,info = crop(input_data, h=128, l=48, train=True)
                # #print("333")
                # #print(train_data.shape)
                # train_data = train_data.unsqueeze(0)
                # train_data =train_data.to(device)
                # info = torch.Tensor([info])
                # info  = info.to(device)

                # real_ =train_data
                # fake_,_,_ = model(real_.float().to(device),info)
                mae = MAE_(fake_.detach().cpu().numpy(),gt_data.cpu().numpy())
                MAE += mae 
                num += 1 

            print ('Val MAE:',MAE/num)
            if MAE/num < global_mae:
                global_mae = MAE/num
                # Save best models checkpoints
                print('saving the current best model at the end of epoch %d, iters %d' % (epoch, total_steps))
                #model.save('best')
                #model.save(epoch)
                #model.save_network(model, OCT_Type, 'best',opt,device,Lora = )
                model.save_network(model, In_Type, 'best',opt,device)  
                print("saving best...")

            #visualizer.print_current_metrics(epoch, MAE/num) 
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))  

    #if epoch > opt.n_epochs:
    #    model.update_learning_rate()


#CUDA_VISIBLE_DEVICES=1  python trainVQGAN_OCT2OCTA_Ali.py --dataroot /mnt/gdut/sam_zgm/C_zgm/Dataset/OCTA_Denoise --name Clean2NoiseAE --model TransPro --netG unet_256 --direction AtoB --lambda_A 10 --lambda_C 5 --dataset_mode alignedoct2octa3d --norm batch --pool_size 0 --load_size 304 --input_nc 1 --output_nc 1 --display_port 6031 --gpu_ids 0 --no_flip >Clean2NoiseAE.txt#

#CUDA_VISIBLE_DEVICES=0 nohup python trainVQGAN.py --dataroot /mnt/gdut/sam_zgm/C_zgm/Dataset/OCTA_Denoise  --name transpro_Noise_Recon --model TransPro --netG unet_256 --direction AtoB --lambda_A 10 --lambda_C 5 --dataset_mode alignedoct2octa3d --norm batch --pool_size 0 --load_size 304 --input_nc 1 --output_nc 1 --display_port 6031 --gpu_ids 0 --no_flip >Noise_Recon.txt

#CUDA_VISIBLE_DEVICES=1 nohup python trainVQGAN.py --dataroot /mnt/gdut/sam_zgm/C_zgm/Dataset/OCTA_Denoise  --name transpro_Clean_Recon --model TransPro --netG unet_256 --direction AtoB --lambda_A 10 --lambda_C 5 --dataset_mode alignedoct2octa3d --norm batch --pool_size 0 --load_size 304 --input_nc 1 --output_nc 1 --display_port 6031 --gpu_ids 0 --no_flip >Clean_Recon.txt  