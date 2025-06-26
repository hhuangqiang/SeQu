import os
from data.base_dataset3d import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchio as tio
import torch


class AlignedOCT2OCTA3DDataset(BaseDataset):
    """A dataset class for paired image dataset.

    OCT to OCTA 3D

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    

    def __init__(self, opt, phase):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, phase, 'A_Noise')  # get the image directory
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # get image paths

        self.dir_B = os.path.join(opt.dataroot, phase, 'B_Clean')  # get the image directory
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # get image paths

        self.dir_label = os.path.join(opt.dataroot, phase, 'label')
        self.label_paths = sorted(make_dataset(self.dir_label, opt.max_dataset_size))
        print("AB paths", self.dir_A, self.dir_B)
        print("AB paths length", len(self.A_paths), len(self.B_paths))
        assert(len(self.A_paths)==len(self.B_paths)== len(self.label_paths))
        #assert (len(self.A_paths) == len(self.B_paths))
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        label_path = self.label_paths[index]

        A = Image.open(A_path)
        A = A.convert('RGB')

        A = np.array(A)
        
        #print(A.shape)  # 查看数组的维度

        
        B = Image.open(B_path)
        B = B.convert('RGB')
        B = np.array(B)
        if label_path.endswith('.txt'):
        #    # 文本标签（假设存储为整数）
            with open(label_path, 'r') as f:
                  label = int(f.read().strip())
        else:
        #     # 图像标签（如二值掩码）
             label = Image.open(label_path).convert('L')
             label = np.array(label)
        
        # A = np.load(A_path)
        # B = np.load(B_path)
        #A = np.expand_dims(A, axis=0)
        #B = np.expand_dims(B, axis=0)

        # # apply the same transform to both A and B
        # transform_params = get_params(self.opt, A.size)
        # A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        # B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))  
        
        #transform_list = []
        transform_list = [transforms.ToTensor()]
        A_transform = transforms.Compose(transform_list)
        B_transform = transforms.Compose(transform_list)
        A = A_transform(A)
        B = B_transform(B) 

        A = (A-A.min())/(A.max()-A.min())
        B = (B-B.min())/(B.max()-B.min())
        A = 2*A - 1
        B = 2*B - 1

        #标签处理（若为图像）
        if isinstance(label, np.ndarray):
              label = transforms.ToTensor()(label).float()

        

        # print("1111 ",A.shape)
        # print("2222 ",B.shape)
        return {'A': A, 'B': B, 'label': label, 'A_paths': A_path, 'B_paths': B_path}
        #return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        assert (len(self.A_paths)==len(self.B_paths))  
        return len(self.A_paths)
