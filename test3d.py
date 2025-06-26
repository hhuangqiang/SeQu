import time
import os
from options.test_options import TestOptions
#from data.data_loader import CreateDataLoader
from data import create_dataset
from models import create_model
#from util.visualizer3d import Visualizer
from util.visualizer3d import save_images
from pdb import set_trace as st
from util import html

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

#data_loader = CreateDataLoader(opt)
#dataset = data_loader.load_data()

dataset = create_dataset(opt, phase='test')
model = create_model(opt)
model.setup(opt) 

#visualizer = Visualizer(opt)
# create website

web_dir = os.path.join(opt.results_dir, opt.test_name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
for i, data in enumerate(dataset):
    if i >= opt.num_test:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    save_images(webpage, visuals, img_path)

webpage.save()

# python test3d.py --dataroot /home/slidm/OCTA/octa-500/OCT2OCTA3M_3D --name transpro_3M_C5 --test_name transpro_3M_C5 --model TransPro --netG unet_256 --direction AtoB --lambda_A 10 --lambda_C 5 --dataset_mode alignedoct2octa3d --norm batch --input_nc 1 --output_nc 1 --gpu_ids 7 --num_test 15200 --which_epoch 194 --load_iter 194

