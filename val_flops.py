import torch
from torchvision import models
from thop import profile
from network.models import *


device          = 'cuda:2'

if __name__ == '__main__':
    #net = Model_5(num_class=7,device=device)
    net = models.resnet18().to(device)
    inputs = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(net, inputs=(inputs, ))
    print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
    print("params=", str(params/1e6)+'{}'.format("M"))
