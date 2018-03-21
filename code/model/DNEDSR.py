from model import common
import pdb

import torch.nn as nn
from torch import cat

def make_model(args, parent=False):
    return DNEDSR(args)

class DNEDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DNEDSR, self).__init__()

        n_resblock = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, -1)
        
        # define head module
        modules_head1 = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body1 = [
            common.ResBlock(
                conv, n_feats, kernel_size,bn=True, act=act, res_scale=args.res_scale) \
            for _ in range(n_resblock)]
        modules_body1.append(conv(n_feats, n_feats, kernel_size))
        
        modules_tail1 = [conv(n_feats, args.n_colors, kernel_size)]
        
        modules_head2 = [conv(2*args.n_colors, n_feats, kernel_size)]
        
        modules_body2 = [
            common.ResBlock(
                conv, n_feats, kernel_size,bn=False, act=act, res_scale=args.res_scale) \
            for _ in range(n_resblock)]
        modules_body2.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail2 = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, 1)

        self.head1 = nn.Sequential(*modules_head1)
        self.body1 = nn.Sequential(*modules_body1)
        self.tail1 = nn.Sequential(*modules_tail1)
        self.head2 = nn.Sequential(*modules_head2)
        self.body2 = nn.Sequential(*modules_body2)
        self.tail2 = nn.Sequential(*modules_tail2)

    def forward(self, x):
        sm = self.sub_mean(x)
        x = self.head1(sm)
        res = self.body1(x)
        res += x
        x = self.tail1(res)
        x = cat([x,sm],1)
        x = self.head2(x)
        res = self.body2(x)
        res += x
        x = self.tail2(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

