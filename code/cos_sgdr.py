import torch

import utils
from option import args
from data import data
from trainer import Trainer
from trainer import CosineAnneal

torch.manual_seed(args.seed)
checkpoint = utils.checkpoint(args)

if checkpoint.ok:
    my_loader = data(args).get_loader()
    t = Trainer(my_loader, checkpoint, args)
    t.callbacks = [CosineAnneal(t.optimizer,lr_min=1e-7, lr_max=1e-5, cycle_len=8, cycle_mult=2)]
    t.fit()

    checkpoint.done()

