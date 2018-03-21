#!/usr/bin/env bash

#python lrfinder.py --model EDSR --scale 4 --n_GPU 2 --save EDSRNOISE_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --epochs 1 --batch_size 64 --pre_train ~/SuperResolution/EDSR-PyTorch/experiment/EDSR_x4/model/model_lastest.pt
#python cos_sgdr.py --ext bin --model EDSR --test_every 500 --scale 4 --n_GPU 2 --save EDSRNOISE3 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --batch_size 64 --pre_train ../experiment/EDSRNOISE_x4/model/model_best.pt


python cos_sgdr.py --ext bin  --optimizer SGD --model EDSR --test_every 1000 --scale 4 --n_GPU 2 --save EDSRFINALSGD --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/EDSRFINAL/model/model_lastest.pt

#python lrfinder.py --ext bin --model DNEDSR --test_every 250 --scale 4 --n_GPU 2 --save DNEDSR3 --epochs 1 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --batch_size 32 --pre_train ../experiment/DNEDSR/model/model_lastest.pt
#python cos_sgdr.py --ext bin --model EDSR --test_every 500 --scale 8 --n_GPU 2 --save EDSRx8 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --batch_size 64 --pre_train ~/SuperResolution/MODELS/X4.pt
