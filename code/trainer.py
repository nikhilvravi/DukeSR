import math
import random
from decimal import Decimal
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import bcolz

import utils

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import torchvision.utils as tu

class Trainer():
    def __init__(self, loader, ckp, args):
        self.args = args
        self.scale = args.scale

        self.loader_train, self.loader_test = loader
        self.model, self.loss, self.optimizer, self.scheduler = ckp.load()
        self.ckp = ckp

        self.log_training = 0
        self.log_test = 0
        
        if(self.loader_train):
            self.hyper = {}
            self.callbacks = []
            self.steps_per_epoch = len(self.loader_train)
            self.callback_dict = {}
            self.iteration = 0
            self.should_stop = False

    def _scale_change(self, idx_scale, testset=None):
        if len(self.scale) > 1:
            if self.args.n_GPUs == 1:
                self.model.set_scale(idx_scale)
            else:
                self.model.module.set_scale(idx_scale)

            if testset is not None:
                testset.dataset.set_scale(idx_scale)

    def fit(self):
        self.callback_dict = {
            "trainer": self,
            "model": self.model,
            "hyper": self.hyper,
            "steps_per_epoch": self.steps_per_epoch,
        }

        utils.apply_on_all(self.callbacks, "on_train_begin", self.callback_dict)

        have_header = False
        column_names = []
        hyper_keys = list(self.hyper.keys())

        while not self.terminate():
            self.train()
            self.test()
            if self.should_stop:
                break

    def train(self):
        self.scheduler.step()
        epoch = self.scheduler.last_epoch + 1
        self.callback_dict["epoch"] = epoch
        utils.apply_on_all(self.callbacks, "on_epoch_begin", self.callback_dict)

        total_steps = 0
        total_examples = 0
        results = {}
        #running_metrics = defaultdict(float)

        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.ckp.add_log(torch.zeros(1, len(self.loss)))
        self.model.train()

        timer_data, timer_model = utils.timer(), utils.timer()
        for batch, (input, target, idx_scale) in enumerate(self.loader_train):
            self.callback_dict["batch"] = batch
            self.callback_dict["iteration"] = self.iteration
            utils.apply_on_all(self.callbacks, "on_batch_begin", self.callback_dict)
            self.iteration += 1
            total_steps += 1

            input, target = self._prepare(input, target)
            chunks_input = input.chunk(self.args.superfetch, dim=0)
            chunks_target = target.chunk(self.args.superfetch, dim=0)
            self._scale_change(idx_scale)

            timer_data.hold()
            timer_model.tic()
            for ci, ct in zip(chunks_input, chunks_target): 
                self.optimizer.zero_grad()
                output = self.model(ci)
                loss = self._calc_loss(output, ct)
                temp_loss = loss
                loss.backward()
                self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self._display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            #self.callback_dict["batch_loss"] = loss.data[0]
            self.callback_dict["batch_loss"] = temp_loss.data[0]
            utils.apply_on_all(self.callbacks, "on_batch_end", self.callback_dict)


            timer_data.tic()
            if self.should_stop:
                break

        utils.apply_on_all(self.callbacks, "on_epoch_end", self.callback_dict)
        self.ckp.log_training[-1, :] /= len(self.loader_train)


    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)), False)
        self.model.eval()

        # We can use custom forward function 
        def _test_forward(x, scale):
            if self.args.self_ensemble:
                return utils.x8_forward(x, self.model, self.args.precision)
            elif self.args.chop_forward:
                return utils.chop_forward(x, self.model, scale)
            else:
                return self.model(x)

        timer_test = utils.timer()
        set_name = type(self.loader_test.dataset).__name__
        for idx_scale, scale in enumerate(self.scale):
            eval_acc = 0
            self._scale_change(idx_scale, self.loader_test)
            for idx_img, (input, target, _) in enumerate(self.loader_test):
                input, target = self._prepare(input, target, volatile=True)
                output = _test_forward(input, scale)
                eval_acc += utils.calc_PSNR(
                    output, target, set_name, self.args.rgb_range, scale)
                self.ckp.save_results(idx_img, input, output, target, scale)

            self.ckp.log_test[-1, idx_scale] = eval_acc / len(self.loader_test)
            best = self.ckp.log_test.max(0)
            performance = 'PSNR: {:.3f}'.format(
                self.ckp.log_test[-1, idx_scale])
            self.ckp.write_log(
                '[{} x{}]\t{} (Best: {:.3f} from epoch {})'.format(
                    set_name,
                    scale,
                    performance,
                    best[0][idx_scale],
                    best[1][idx_scale] + 1))

        if best[1][0] + 1 == epoch:
            is_best = True
        else:
            is_best = False

        self.ckp.write_log(
            'Time: {:.2f}s\n'.format(timer_test.toc()), refresh=True)
        self.ckp.save(self, epoch, is_best=is_best)

    def _prepare(self, input, target, volatile=False):
        if not self.args.no_cuda:
            input = input.cuda()
            target = target.cuda()

        input = Variable(input, volatile=volatile)
        target = Variable(target)
           
        return input, target

    def _calc_loss(self, output, target):
        loss_list = [] 
        
        for i, l in enumerate(self.loss):
            if isinstance(output, list):
                if isinstance(target, list):
                    loss = l['function'](output[i], target[i])
                else:
                    loss = l['function'](output[i], target)
            else:
                loss = l['function'](output, target)

            loss_list.append(l['weight'] * loss)
            self.ckp.log_training[-1, i] += loss.data[0]

        loss_total = reduce((lambda x, y: x + y), loss_list)
        if len(self.loss) > 1:
            self.ckp.log_training[-1, -1] += loss_total.data[0]

        return loss_total

    def _display_loss(self, batch):
        n_samples = self.args.superfetch * (batch + 1)
        log = [
            '[{}: {:.4f}] '.format(t['type'], l / n_samples) \
            for l, t in zip(self.ckp.log_training[-1], self.loss)]

        return ''.join(log)

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.args.epochs

class Callback():
    """Training callbacks must extend this abstract base class."""

    def __init__(self): pass

    def on_train_begin(self, info_dict): pass

    def on_epoch_begin(self, info_dict): pass

    def on_batch_begin(self, info_dict): pass

    def on_batch_end(self, info_dict): pass

    def on_eval_begin(self, info_dict): pass

    def on_eval_end(self, info_dict): pass

    def on_epoch_end(self, info_dict): pass

    def on_train_end(self, info_dict): pass


class LRFinder(Callback):
    """Increments the learning rate on every batch until the loss starts
    increasing again. Use this to determine a good learning rate to start
    training the model with.

    Based on code and ideas from https://github.com/fastai/fastai

    Parameters
    ----------
    optimizer: torch.optim object
    start_lr: float
        The learning rate to start with (should be quite small).
    end_lr: float
        The maximum learning rate to try (should be large-ish).
    steps: int
        How many batches to evaluate. One epoch is usually enough.
    """

    def __init__(self, optimizer, start_lr=1e-5, end_lr=10, steps=100):
        self.optimizer = optimizer
        self.steps = steps
        self.values = np.logspace(np.log10(start_lr), np.log10(end_lr), steps)
        self.epoch = 1

    def on_train_begin(self, info_dict):
        self.best_loss = 1e9
        self.loss_history = []
        self.lr_history = []

    def on_batch_begin(self, info_dict):
        lr = self.values[info_dict["iteration"]]
        set_lr(self.optimizer, lr)
        self.lr_history.append(lr)

    def on_batch_end(self, info_dict):
        loss = info_dict["batch_loss"]
        iteration = info_dict["iteration"]

        # Note: in the last couple of batches the loss may explode,
        # which is why we don't plot those.
        self.loss_history.append(loss)

        if math.isnan(loss) or loss > self.best_loss * 4 or iteration >= self.steps - 1:
            info_dict["trainer"].should_stop = True
            return

        if loss < self.best_loss and iteration > 10:
            self.best_loss = loss

    def plot(self, figsize=(12, 8)):
        fig = plt.figure(figsize=figsize)
        plt.ylabel("loss", fontsize=16)
        plt.xlabel("learning rate (log scale)", fontsize=16)
        plt.xscale("log")
        plt.tick_params(axis='x', which='minor')
        plt.plot(self.lr_history[10:-5], self.loss_history[10:-5])
        utils.save_array(f'../experiment/lr_find_edsr2'
                         f'/lr_history_steps_{self.steps}_epoch{self.epoch}.bc',self.lr_history[10:-5])
        utils.save_array(f'../experiment/lr_find_edsr2'
                         f'/loss_history_steps_{self.steps}_epoch{self.epoch}.bc',self.loss_history[10:-5])
        plt.savefig(f'../experiment/lr_find_edsr2/lr_find_steps_'
                    f'{self.steps}_epoch{self.epoch}.png',bbox_inches='tight')
        plt.show()
        #print('It worked!')

    def on_epoch_end(self, info_dict):
        self.plot()
        self.epoch += 1


class CosineAnneal(Callback):
    """Cosine annealing for the learning rate, with restarts.

    The learning rate is varied between lr_max and lr_min over cycle_len
    epochs.

    Note: The validation score may temporarily be worse in the first part
    of the cycle (where the learning rate is high). This is why you should
    always train for a round number of cycles. For example, if cycle_len=1
    and cycle_mult=2 then train for 1, 3, 7, 15, 31 etc epochs.

    It's allowed to change the cycle_len and cycle_mult parameters before
    the next training run, but make sure you do this after the last cycle
    has completely finished (or else there will be abrupt changes in the LR).

    Based on the paper 'SGDR: Stochastic Gradient Descent with Warm Restarts',
    arXiv:1608.03983 and code from https://github.com/fastai/fastai

    Parameters
    ----------
    optimizer: torch.optim object
    lr_min: float
        The lowest learning rate.
    lr_max: float
        The highest learning rate.
    cycle_len: int
        How many epochs there are in one cycle.
    cycle_mult: int
        After each complete cycle, the cycle_len is multiplied by this number.
        This makes the learning rate anneal at a slower pace over time.
    """

    def __init__(self, optimizer, lr_min, lr_max, cycle_len, cycle_mult=1):
        self.optimizer = optimizer
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.cycle_len = cycle_len
        self.cycle_mult = cycle_mult
        self.cycles_completed = 0
        self.cycle_iter = 0
        self.cycle_width = 1
        self.lr_history = []
        self.epoch = 1

    def on_batch_begin(self, info_dict):
        steps_per_epoch = info_dict["steps_per_epoch"]
        steps = steps_per_epoch * self.cycle_len * self.cycle_width

        # Use a low learning rate for the very first batches.
        if info_dict["iteration"] < steps_per_epoch / 20:
            lr = self.lr_max / 100.
        else:
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * \
                 (1. + np.cos(self.cycle_iter * np.pi / steps))

        set_lr(self.optimizer, lr)

        if(self.cycle_iter % 10 == 0): print(lr)	

        self.cycle_iter += 1
        if self.cycle_iter >= steps:
            self.cycle_iter = 0
            self.cycle_width *= self.cycle_mult
            self.cycles_completed += 1
        self.lr_history.append(lr)

    def on_epoch_end(self, info_dict):
        self.epoch += 1
        self.lr_max = self.lr_max*0.95
        self.lr_min = self.lr_min*0.95

def set_lr(optimizer, lr):
    """Use this to manually change the learning rate."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
