
# CUDA_VISIBLE_DEVICES=0 nohup  python trainVQGAN_OCT2OCTA_KD.py --dataroot /home/eezzchen/OCT2OCTA3M_3D --name transpro_OCT2OCTA_KD_Baseline  -
# \ -model TransPro --netG unet_256 --direction AtoB --lambda_A 10 --lambda_C 5 --dataset_mode alignedoct2octa3d --norm batch --pool_size 0 
# \ --load_size 304 --input_nc 1 --output_nc 1 --display_port 6031 --gpu_ids 0 --no_flip >OCT2OCTA_KD_Baseline.txt

CUDA_VISIBLE_DEVICES=2 python testVQGAN_OCT2OCTA.py --dataroot  /home/eezzchen/OCT2OCTA3M_3D  --name transpro_OCT2OCTA_KD_Tau2 --test_name transpro_OCT2OCTA_KD_Sim_Mul_2 
\ --model TransPro --netG unet_256 --direction AtoB --lambda_A 10 --lambda_C 5 --dataset_mode alignedoct2octa3d --norm batch 
\ --input_nc 1 --output_nc 1 --gpu_ids 0 --num_test 15200 --which_epoch 194 --load_iter 194