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
import loralib as lora
import os
import torchvision.utils as tvu
from PIL import Image
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
In_Type = 'B' # 'A' 'B' 
Out_Type = 'A' # 'A' 'B' 
OCT_Type = In_Type+Out_Type

pre_train = False   
Lora_Flag = False   

if pre_train ==False:
    optimizer = optim.Adam(model.parameters(), lr=lr) 
else:
    # Encoder_path =  '/home/eezzchen/TransPro/checkpoints/transpro_A/best_encoder_A.pth'
    # model.encoder.load_state_dict(torch.load(Encoder_path)) 

    Codebooks_path = '/mnt/gdut/sam_zgm/C_zgm/DenoisePro/checkpoints/Clean2Noise/best_codebook_BA.pth'  
    model.codebooks.load_state_dict(torch.load(Codebooks_path)) 
    print("Load codebooks success!!!!")

    Decoder_path = '/mnt/gdut/sam_zgm/C_zgm/DenoisePro/checkpoints/Clean2Noise/best_decoder_BA.pth'  
    model.decoder.load_state_dict(torch.load(Decoder_path)) 
    print("Load decoder success!!!!")  

    #lora.mark_only_lora_as_trainable(model)
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    if Lora_Flag == False: 
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # for param in model.encoder.parameters():
        #     param.requires_grad = False  
        for param in model.codebooks.parameters():
            param.requires_grad = False

        for param in model.decoder.parameters(): 
            param.requires_grad = False  
        
    else:
        #lora.mark_only_lora_as_trainable(model)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        lora.mark_only_lora_as_trainable(model)

        for param in model.Adapter.parameters():
            param.requires_grad = True

        # for param in model.encoder.parameters():
        #     param.requires_grad = False
        
        # for param in model.codebooks.parameters():
        #     param.requires_grad = False 

scheduler_type = "cycle"
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
min_batch_size = 4

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

    for i, data in enumerate(dataset):
        #if i ==0: continue      
        train_data_A = data[In_Type].float() #[0]
        train_data_B = data[Out_Type].float() #[0]

        train_data_A = train_data_A.to(device)

        train_data_B = train_data_B.to(device)

        # info = torch.Tensor([info])
        # info  = info.to(device)
        #print(train_data_A.shape)
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size
        model.zero_grad()
        #print(train_data_A.shape)  
        out,_,latent_loss,_,_ = model(train_data_A)
        
        # print(out.shape)
        # while(1):True
        
        recon_loss = criterion(out, train_data_B)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()
        
        cur_mae = MAE_(out.detach().cpu().numpy(),train_data_B.cpu().numpy())
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
        model.save_network(model,OCT_Type, 'latest',opt,device)
    

    if epoch % opt.val_epoch_freq == 0: 
        device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        with torch.no_grad():
            MAE = 0 
            num = 0 
            for i, data in enumerate(val_dataset):
                train_data_A = data[In_Type].float() #[0]
                train_data_B = data[Out_Type].float() #[0] 

                train_data_A = train_data_A.to(device) 
                train_data_B = train_data_B.to(device)  

                fake_,_,latent_loss,_,_ = model(train_data_A)

                mae = MAE_(fake_.detach().cpu().numpy(),train_data_B.cpu().numpy())
                MAE += mae
                num += 1 

            print ('Val MAE:',MAE/num)
            if MAE/num < global_mae:
                global_mae = MAE/num
                # Save best models checkpoints
                print('saving the current best model at the end of epoch %d, iters %d' % (epoch, total_steps))
                #model.save('best')
                #model.save(epoch)
                model.save_network(model, OCT_Type, 'best',opt,device)
                print("saving best...")

                if Lora_Flag == True:
                    print("Save LoRa params") 
                    save_dir = os.path.join(opt.checkpoints_dir, opt.name) 
                    save_filenameLoRa = '%s_LoRa_%s.pth' % ('best', OCT_Type)
                    save_filenameLoRa = os.path.join(save_dir, save_filenameLoRa)
                    torch.save(lora.lora_state_dict(model), save_filenameLoRa)
                    #save_path_encoder  = os.path.join(save_dir, save_filename_encoder)

            #visualizer.print_current_metrics(epoch, MAE/num)
          
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))  



# CUDA_VISIBLE_DEVICES=2 nohup python trainVQGAN_OCT2OCTA.py --dataroot /mnt/gdut/sam_zgm/C_zgm/OCT2OCTA3M_3D --name transpro_B_Test2 --model TransPro --netG unet_256 --direction AtoB --lambda_A 10 --lambda_C 5 --dataset_mode alignedoct2octa3d --norm batch --pool_size 0 --load_size 304 --input_nc 1 --output_nc 1 --display_port 6031 --gpu_ids 0 --no_flip >Test.txt
# CUDA_VISIBLE_DEVICES=2  python trainVQGAN_OCT2OCTA.py --dataroot /mnt/gdut/sam_zgm/C_zgm/Dataset/OCTA_Denoise --name NoiseInputFinetune --model TransPro --netG unet_256 --direction AtoB --lambda_A 10 --lambda_C 5 --dataset_mode alignedoct2octa3d --norm batch --pool_size 0 --load_size 304 --input_nc 1 --output_nc 1 --display_port 6031 --gpu_ids 0 --no_flip > NoiseInputFinetune.txt  


# CUDA_VISIBLE_DEVICES=3  python trainVQGAN_OCT2OCTA.py --dataroot /mnt/gdut/sam_zgm/C_zgm/Dataset/OCTA_Denoise --name Clean2Noise_1 --model TransPro --netG unet_256 --direction AtoB --lambda_A 10 --lambda_C 5 --dataset_mode alignedoct2octa3d --norm batch --pool_size 0 --load_size 304 --input_nc 1 --output_nc 1 --display_port 6031 --gpu_ids 0 --no_flip > Clean2Noise_1.txt


