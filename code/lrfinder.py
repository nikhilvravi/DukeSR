import torch

import utils
from option import args
from data import data
from trainer import Trainer
from trainer import LRFinder

torch.manual_seed(args.seed)
checkpoint = utils.checkpoint(args)

if checkpoint.ok:
    my_loader = data(args).get_loader()
    t = Trainer(my_loader, checkpoint, args)
    t.callbacks = [LRFinder(t.optimizer, 1e-9,10,1000)]
    t.fit()

    t.callbacks[0].plot()

    checkpoint.done()

