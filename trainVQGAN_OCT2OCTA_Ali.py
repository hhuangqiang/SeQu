import time
from options.train_options import TrainOptions
from util.visualizer3d import Visualizer
from data import create_dataset
from models import create_model
import torch
import numpy as np
import VQGAN
import AE
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
import itertools
import torchvision.utils as tvu
from PIL import Image
import math
from einops import rearrange, reduce, repeat

def MAE_(fake,real):
    mae = 0.0
    mae = np.mean(np.abs(fake-real))
    return mae

def Norm(a):
    max_ = torch.max(a)
    min_ = torch.min(a)
    a_0_1 = (a-min_)/(max_-min_)
    return (a_0_1-0.5)*2

class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.proj_head = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, dim_out)
        )
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.proj_head(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class MultiProjector(nn.Module):
    def __init__(self,):
        super().__init__()
        print("Multi Projector")
        self.enc_block_0 = Embed(dim_in=32, dim_out=1024)
        self.enc_block_1 = Embed(dim_in=64, dim_out=1024)
        self.enc_block_2 = Embed(dim_in=128, dim_out=1024)
        self.enc_block_3 = Embed(dim_in=256, dim_out=1024) 
        

    def forward(self, x_lis):
        x_0, x_1, x_2, x_3 = x_lis
        
        x_0_denoise = self.enc_block_0(x_0) #+ x_0
        
        x_1_denoise = self.enc_block_1(x_1) #+ x_1
        
        x_2_denoise = self.enc_block_2(x_2) #+ x_2
        
        x_3_denoise = self.enc_block_3(x_3) #+ x_3
        
        return [x_0_denoise, x_1_denoise, x_2_denoise, x_3_denoise]

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

model_type = "VQGAN" # VAE VQGAN

if model_type == "AE":
    model = AE.AE().to(device) 
elif model_type == "VQGAN":
    model = VQGAN.VQGAN().to(device)  

lr = 3e-4 #3e-4
In_Type = 'A' # 'A' 'B' 
Out_Type = 'B' # 'A' 'B' 
OCT_Type = In_Type + Out_Type


model_Clean2Noise = VQGAN.VQGAN().to(device)  
Clean2Noise_path = '/mnt/gdut/sam_zgm/C_zgm/DenoisePro/checkpoints/Clean2Noise/best_net_BA.pth'
model_Clean2Noise.load_state_dict(torch.load(Clean2Noise_path)) 
print("Load Clean2Noise success!!!!")  


Unquant_proj_T = MultiProjector().to(device)
quant_proj_T = MultiProjector().to(device)
Unquant_proj_S = MultiProjector().to(device)
quant_proj_S = MultiProjector().to(device)


params = itertools.chain(model.parameters(), Unquant_proj_T.parameters(), Unquant_proj_S.parameters(), quant_proj_T.parameters(), quant_proj_S.parameters())  
optimizer = optim.Adam(params, lr=lr) 


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
Patch_list= [32,16,8,4]   
Patch_Num = 64 # 128/32 * 128/32 
Multal_weight = 0.04
tau = 0.1 
Exp = 2

print("Patch_Num: ", Patch_Num)
print("Patch_list: ", Patch_list)
print("Multal_weight: ", Multal_weight)  
print("tau: ", tau)
print("Exp: ", Exp)
print("Demision: 256 ")

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
        train_data_A = data[In_Type].float() 
        train_data_B = data[Out_Type].float() 

        train_data_A = train_data_A.to(device)

        #train_data_B = train_data_B.unsqueeze(0)
        train_data_B = train_data_B.to(device)

        # info = torch.Tensor([info])
        # info  = info.to(device)
        #print(train_data_A.shape)
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size

        with torch.no_grad():
            out_Clean2Noise,_, _, ref_unquant_T, ref_quant_T  = model_Clean2Noise(train_data_B)

        model.zero_grad()  
        out,_, latent_loss,ref_unquant_S,ref_quant_S = model(train_data_A)

        Proj_ref_unquant_T = []
        Proj_ref_quant_T = []
        Proj_ref_unquant_S = []
        Proj_ref_quant_S = []

        for ind in range(4): 
            #print(ref_unquant_T[ind].shape)
            #while(1):True 
            Cur_ref_unquant_T = rearrange(ref_unquant_T[ind], 'b c (h p1) (w p2)  -> (b h w) c p1 p2 ', p1= Patch_list[ind] , p2= Patch_list[ind])
            Cur_ref_quant_T = rearrange(ref_quant_T[ind], 'b c (h p1) (w p2)  -> (b h w) c p1 p2 ', p1= Patch_list[ind] , p2= Patch_list[ind])
            Cur_ref_unquant_S = rearrange(ref_unquant_S[ind], 'b c (h p1) (w p2)  -> (b h w) c p1 p2 ', p1= Patch_list[ind] , p2= Patch_list[ind])
            Cur_ref_quant_S = rearrange(ref_quant_S[ind], 'b c (h p1) (w p2)  -> (b h w) c p1 p2 ', p1= Patch_list[ind] , p2= Patch_list[ind])

            # print(Cur_ref_unquant_T.shape) 
            # print(Cur_ref_quant_T.shape) 
            # print(Cur_ref_unquant_S.shape) 
            # print(Cur_ref_quant_S.shape)      

            Cur_ref_unquant_T = F.adaptive_avg_pool2d(Cur_ref_unquant_T,(1, 1)).view(Patch_Num, -1)
            Cur_ref_quant_T = F.adaptive_avg_pool2d(Cur_ref_quant_T,(1, 1)).view(Patch_Num, -1)
            Cur_ref_unquant_S = F.adaptive_avg_pool2d(Cur_ref_unquant_S,(1, 1)).view(Patch_Num, -1)
            Cur_ref_quant_S = F.adaptive_avg_pool2d(Cur_ref_quant_S,(1, 1)).view(Patch_Num, -1) 

            # print(Cur_ref_unquant_T.shape) 
            # print(Cur_ref_quant_T.shape) 
            # print(Cur_ref_unquant_S.shape) 
            # print(Cur_ref_quant_S.shape)  

            Proj_ref_unquant_T.append(Cur_ref_unquant_T)
            Proj_ref_quant_T.append(Cur_ref_quant_T)
            Proj_ref_unquant_S.append(Cur_ref_unquant_S)
            Proj_ref_quant_S.append(Cur_ref_quant_S)


        Proj_ref_unquant_T = Unquant_proj_T(Proj_ref_unquant_T)
        Proj_ref_quant_T = quant_proj_T(Proj_ref_quant_T)
        Proj_ref_unquant_S = Unquant_proj_S(Proj_ref_unquant_S)
        Proj_ref_quant_S = quant_proj_S(Proj_ref_quant_S)    


        mutual_contrastive_loss_unquant = 0.0 
        for ind in range(4):
            ## Exp 1
            if Exp == 1:
                embeddings_a = Proj_ref_unquant_S[ind]
                embeddings_b = Proj_ref_quant_T[ind]
            else:
                embeddings_a = Proj_ref_unquant_S[ind]
                embeddings_b = Proj_ref_unquant_T[ind]


            diag_mask = torch.ones([Patch_Num,Patch_Num]).cuda() #  (1.-torch.eye(4).cuda())
            intra_mask = torch.eye(Patch_Num).cuda() 

            logit = torch.div(torch.mm(embeddings_b.detach().clone(), embeddings_a.T),tau) #cos_simi_ji

            log_prob = logit - torch.log((torch.exp(logit) * diag_mask).sum(1, keepdim=True))
            mean_log_prob_pos = (intra_mask * log_prob).sum(1) / intra_mask.sum(1)
            icl_loss = - mean_log_prob_pos.mean()
            mutual_contrastive_loss_unquant = mutual_contrastive_loss_unquant + icl_loss*0.25

        mutual_contrastive_loss_unquant = mutual_contrastive_loss_unquant * Multal_weight

        mutual_contrastive_loss_quant  = 0.0   

        for ind in range(4):
            if Exp == 1:
                embeddings_a = Proj_ref_quant_S[ind] 
                embeddings_b = Proj_ref_unquant_T[ind] 
            else:
                embeddings_a = Proj_ref_quant_S[ind] 
                embeddings_b = Proj_ref_quant_T[ind]                 

            ## Exp 2


            #mutual_contrastive_loss = mutual_contrastive_loss + Multal_paparms[ind]*criterion(embeddings_a,embeddings_b) 
            diag_mask = torch.ones([Patch_Num,Patch_Num]).cuda() #  (1.-torch.eye(4).cuda())
            intra_mask = torch.eye(Patch_Num).cuda() 

            logit = torch.div(torch.mm(embeddings_b.detach().clone(), embeddings_a.T),tau) #cos_simi_ji

            log_prob = logit - torch.log((torch.exp(logit) * diag_mask).sum(1, keepdim=True))
            mean_log_prob_pos = (intra_mask * log_prob).sum(1) / intra_mask.sum(1)
            icl_loss = - mean_log_prob_pos.mean()
            mutual_contrastive_loss_quant = mutual_contrastive_loss_quant +  icl_loss*0.25

        mutual_contrastive_loss_quant = mutual_contrastive_loss_quant * Multal_weight 


        recon_loss = criterion(out, train_data_B)  

        if model_type == "AE": 
            latent_loss = 0.0
        else:
            latent_loss = latent_loss.mean()

        #print("recon_loss:", recon_loss , " latent_loss: ", latent_loss_weight * latent_loss, " mutual_unquant: ",  mutual_contrastive_loss_unquant , " mutual__quant: ",  mutual_contrastive_loss_quant) 
        loss = recon_loss + latent_loss_weight * latent_loss + mutual_contrastive_loss_unquant +  mutual_contrastive_loss_quant 
        loss.backward()   
        

        cur_mae = MAE_(out.detach().cpu().numpy(),train_data_B.cpu().numpy())
        Train_MAE += cur_mae
        Train_num += 1


        if scheduler is not None:
            scheduler.step()
        optimizer.step()
 
    print("recon_loss:", recon_loss , " latent_loss: ", latent_loss_weight * latent_loss, " mutual_unquant: ",  mutual_contrastive_loss_unquant , " mutual__quant: ",  mutual_contrastive_loss_quant)  
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

                fake_,_, latent_loss,_,_ = model(train_data_A)

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

                # if Lora_Flag == True:
                #     print("Save LoRa params") 
                #     save_dir = os.path.join(opt.checkpoints_dir, opt.name) 
                #     save_filenameLoRa = '%s_LoRa_%s.pth' % ('best', OCT_Type)
                #     save_filenameLoRa = os.path.join(save_dir, save_filenameLoRa)
                #     torch.save(lora.lora_state_dict(model), save_filenameLoRa)
                    #save_path_encoder  = os.path.join(save_dir, save_filename_encoder)

            #visualizer.print_current_metrics(epoch, MAE/num)
          
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))  

# CUDA_VISIBLE_DEVICES=1 nohup python  trainVQGAN_OCT2OCTA_Ali.py --dataroot /mnt/gdut/sam_zgm/C_zgm/Dataset/OCTA_Denoise --name N2C_Ali_1 --model TransPro --netG unet_256 --direction AtoB --lambda_A 10 --lambda_C 5 --dataset_mode alignedoct2octa3d --norm batch --pool_size 0 --load_size 304 --input_nc 1 --output_nc 1 --display_port 6031 --gpu_ids 0 --no_flip >N2C_Ali_1.txt

# CUDA_VISIBLE_DEVICES=7 nohup python  trainVQGAN_OCT2OCTA_Ali.py --dataroot /mnt/gdut/sam_zgm/C_zgm/Dataset/OCTA_Denoise --name N2C_Ali_7 --model TransPro --netG unet_256 --direction AtoB --lambda_A 10 --lambda_C 5 --dataset_mode alignedoct2octa3d --norm batch --pool_size 0 --load_size 304 --input_nc 1 --output_nc 1 --display_port 6031 --gpu_ids 0 --no_flip >N2C_Ali_7.txt





