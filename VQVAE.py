import torch
from torch import nn
from torch.nn import functional as F
import distributed as dist_fn
import os

# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch

class MedNeXtBlock(nn.Module):

    def __init__(self, 
                in_channels:int, 
                out_channels:int, 
                exp_r:int=2, 
                kernel_size:int=7, 
                do_res:int=True,
                norm_type:str = 'group',
                dim = '3d',
                grn = True
                ):

        super().__init__()

        self.do_res = do_res

        assert dim in ['2d', '3d']
        self.dim = dim
        if self.dim == '2d':
            conv = nn.Conv2d
        elif self.dim == '3d':
            conv = nn.Conv3d
            
        # First convolution layer with DepthWise Convolutions
        self.conv1 = conv(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = 1,
            padding = kernel_size//2,
            groups = in_channels,
        )

        # Normalization Layer. GroupNorm is used by default.
        if norm_type=='group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels, 
                num_channels=in_channels
                )
            
        # Second convolution (Expansion) layer with Conv3D 1x1x1
        self.conv2 = conv(
            in_channels = in_channels,
            out_channels = exp_r*in_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )
        
        # GeLU activations
        self.act = nn.GELU()
        
        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv(
            in_channels = exp_r*in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )

        self.grn = grn
        if grn:
            if dim == '3d':
                self.grn_beta = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1,1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1,1), requires_grad=True)
            elif dim == '2d':
                self.grn_beta = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1), requires_grad=True)

 
    def forward(self, x, dummy_tensor=None):

        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        if self.grn:
            # gamma, beta: learnable affine transform parameters
            # X: input of shape (N,C,H,W,D)
            if self.dim == '3d':
                gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
            elif self.dim == '2d':
                gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True)+1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1  
        return x1


class MedNeXtDownBlock(MedNeXtBlock):

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, 
                do_res=True, norm_type = 'group', dim='3d', grn=True):

        super().__init__(in_channels, out_channels, exp_r, kernel_size, 
                        do_res = False, norm_type = norm_type, dim=dim,
                        grn=grn)

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
        self.resample_do_res = do_res
        if do_res:
            self.res_conv = conv(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 2
            )

        self.conv1 = conv(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = 2,
            padding = kernel_size//2,
            groups = in_channels,
        )

    def forward(self, x, dummy_tensor=None):
        
        x1 = super().forward(x)
        
        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res

        return x1
    
class MedNeXtUpBlock(MedNeXtBlock):

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, 
                do_res=True, norm_type = 'group', dim='3d', grn = True):
        super().__init__(in_channels, out_channels, exp_r, kernel_size,
                         do_res=False, norm_type = norm_type, dim=dim,
                         grn=grn)

        self.resample_do_res = do_res
        
        self.dim = dim
        if dim == '2d':
            conv = nn.ConvTranspose2d
        elif dim == '3d':
            conv = nn.ConvTranspose3d
        if do_res:            
            self.res_conv = conv(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 2
                )

        self.conv1 = conv(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = 2,
            padding = kernel_size//2,
            groups = in_channels,
        )


    def forward(self, x, dummy_tensor=None):
        
        x1 = super().forward(x)
        # Asymmetry but necessary to match shape
        
        if self.dim == '2d':
            x1 = torch.nn.functional.pad(x1, (1,0,1,0))
        elif self.dim == '3d':
            x1 = torch.nn.functional.pad(x1, (1,0,1,0,1,0))
        
        if self.resample_do_res:
            res = self.res_conv(x)
            if self.dim == '2d':
                res = torch.nn.functional.pad(res, (1,0,1,0))
            elif self.dim == '3d':
                res = torch.nn.functional.pad(res, (1,0,1,0,1,0))
            x1 = x1 + res

        return x1


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input
        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, kernel_size,type = 0):
        super().__init__()
        print("using MedNext Encoder")
        num_res_blocks = 2
        exp_r = 2
        conv = nn.Conv3d

        if type == 0:
            #self.stem  =  #nn.Conv3d(in_channel, channel, 3, padding=1)
            blocks = [conv(in_channel, channel,  3, padding=1)]
            #blocks.append()

            for i in range(n_res_block):
                blocks.append(MedNeXtBlock(in_channels=channel, out_channels=channel, exp_r=1, kernel_size=kernel_size))

            blocks.append(MedNeXtDownBlock(in_channels=channel, out_channels=channel*2, exp_r=exp_r, kernel_size=kernel_size))

        if type == 1:
            blocks = []
            for i in range(n_res_block):
                blocks.append(MedNeXtBlock(in_channels=channel, out_channels=channel, exp_r=1, kernel_size=kernel_size))

            blocks.append(MedNeXtDownBlock(in_channels=channel, out_channels=channel*2, exp_r=exp_r, kernel_size=kernel_size))

        # if stride == 4:
        #     blocks = [
        #         nn.Conv3d(in_channel, channel // 2, 4, stride=2, padding=1),
        #         nn.ReLU(inplace=True),
        #         nn.Conv3d(channel // 2, channel, 4, stride=2, padding=1),
        #         nn.ReLU(inplace=True),
        #         nn.Conv3d(channel, channel, 3, padding=1),
        #     ]

        # elif stride == 2:
        #     blocks = [
        #         nn.Conv3d(in_channel, channel // 2, 4, stride=2, padding=1),
        #         nn.ReLU(inplace=True),
        #         nn.Conv3d(channel // 2, channel, 3, padding=1),
        #     ]

        # for i in range(n_res_block):
        #     blocks.append(ResBlock(channel, n_res_channel))

        #blocks.append(nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        #print("aaaa")
        #print(self.stem(input).shape)
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, kernel_size,type =0 ):
        super().__init__()
        print("using MedNext Decoder")
        num_res_blocks = 1
        exp_r = 2

        if type ==0:
            blocks = []
            for i in range(n_res_block):
                blocks.append(MedNeXtBlock(in_channels=in_channel, out_channels=out_channel, exp_r=exp_r, kernel_size=kernel_size))

            blocks.append(MedNeXtUpBlock(in_channels=out_channel, out_channels=int(out_channel/2.0), exp_r=exp_r, kernel_size=kernel_size))

        if type ==1:
            blocks = []
            for i in range(n_res_block):
                blocks.append(MedNeXtBlock(in_channels=in_channel, out_channels=channel, exp_r=exp_r, kernel_size=kernel_size))

            blocks.append(MedNeXtUpBlock(in_channels=channel, out_channels=int(channel/2.0), exp_r=exp_r, kernel_size=kernel_size))
            blocks.append(nn.Conv3d(int(channel/2.0), 1, 3, 1, 1))

        self.blocks = nn.Sequential(*blocks)

        # blocks = [nn.Conv3d(in_channel, channel, 3, padding=1)]

        # for i in range(n_res_block):
        #     blocks.append(ResBlock(channel, n_res_channel))

        # blocks.append(nn.ReLU(inplace=True))

        # if stride == 4:
        #     blocks.extend(
        #         [
        #             nn.ConvTranspose3d(channel, channel // 2, 4, stride=2, padding=1),
        #             nn.ReLU(inplace=True),
        #             nn.ConvTranspose3d(
        #                 channel // 2, out_channel, 4, stride=2, padding=1
        #             ),
        #         ]
        #     )

        # elif stride == 2:
        #     blocks.append(
        #         nn.ConvTranspose3d(channel, out_channel, 4, stride=2, padding=1)
        #     )

    def forward(self, input):
        return self.blocks(input)

class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=1,
        channel=32,
        n_res_block=2,
        kernel_size=5,
        embed_dim=128,
        n_embed=512,
        decay=0.99,
    ):  
        super().__init__()
        
        self.enc_b = Encoder(in_channel, channel, n_res_block, kernel_size, type=0)
        self.enc_t = Encoder(in_channel, channel*2, n_res_block, kernel_size, type=1)

        self.dec_t = Decoder(embed_dim, embed_dim, channel, n_res_block, kernel_size, type=0)
        self.dec = Decoder(embed_dim + embed_dim,in_channel,embed_dim+embed_dim,n_res_block,kernel_size,type=1)

        self.quantize_conv_t = nn.Conv3d(channel*4, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)

        self.quantize_conv_b = nn.Conv3d(channel*2  + int(embed_dim/2.0) , embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)

        self.upsample_t = nn.ConvTranspose3d(embed_dim, embed_dim, 4, stride=2, padding=1)


    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        #(enc_b.shape)
        enc_t = self.enc_t(enc_b)


        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 4, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 4, 1, 2, 3)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 4, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)


        quant_b = quant_b.permute(0, 4, 1, 2, 3)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)
        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 4, 1, 2, 3)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 4, 1, 2, 3)
        dec = self.decode(quant_t, quant_b)
        return dec

    def save_network(self, network, network_label, epoch_label,opt,device):
        save_dir = os.path.join(opt.checkpoints_dir, opt.name) 
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        network.to(device)  


#CUDA_VISIBLE_DEVICES=0 nohup python trainVQVAE.py --dataroot /home/eezzchen/OCT2OCTA3M_3D --name transpro_B_Test2 --model TransPro --netG unet_256 --direction AtoB --lambda_A 10 --lambda_C 5 --dataset_mode alignedoct2octa3d --norm batch --pool_size 0 --load_size 304 --input_nc 1 --output_nc 1 --display_port 6031 --gpu_ids 0 --no_flip >Test2.txt
#
# For tet
if 0:
    model = VQVAE().cuda()
    inpput = torch.randn((1,1,8,256,256)).cuda()
    print(model(inpput)[0].shape)