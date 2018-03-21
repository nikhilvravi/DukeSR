#!/usr/bin/env bash
# python main.py --model EDSR --scale 4 --save EDSR_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train ../experiment/EDSR_x2/model/model_lastest.pt

python lrfinder.py --model EDSR --scale 4 --n_GPU 2 --save EDSRNOISE_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --epochs 1 --batch_size 32 --pre_train ~/SuperResolution/EDSR-PyTorch/experiment/EDSR_x4/model/model_lastest.pt

#python cos_sgdr.py --model EDSR --scale 2 --save EDSR_x2 --n_resblocks 32 --n_feats 256 --res_scale 0.1  --reset --epochs 1 --print_every 5 --test_every 50
