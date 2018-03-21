from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
import numpy as np

class loss:
    def __init__(self, args):
        self.args = args

    def get_loss(self):
        print('Preparing loss function...')

        my_loss = []
        losslist = self.args.loss.split('+')
        for loss in losslist:
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'vgg':
                loss_function = vggloss
            elif loss_type == 'edge':
                loss_function = edgeloss
           
            my_loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function})

        if len(losslist) > 1:
            my_loss.append({
                'type': 'Total',
                'weight': 0,
                'function': None})

        print(my_loss)

        return my_loss

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out
    
def edgeloss(inp,tar):
    inp2 = np.hypot(ndimage.sobel(inp, 0),ndimage.sobel(inp, 1))
    tar2 = np.hypot(ndimage.sobel(tar, 0),ndimage.sobel(tar, 1))
    return nn.L1(inp2,tar2)

def vggloss(inp,tar):
    vgg = Vgg16().type(dtype)
    a = vgg(inp)[1]
    b = vgg(tar)[1]
    return nn.MSELoss(a,b)
