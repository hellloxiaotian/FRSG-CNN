import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class STN_Net(nn.Module):
    def __init__(self,in_c=512):
        super(STN_Net, self).__init__()

        self.in_c=in_c
        #localisation net
        #从输入图像中提取特征
        #输入图片的shape为(-1,512,7,7)
        self.localization = nn.Sequential(
            #卷积输出shape为(-1,8,7,7)
            nn.Conv2d(in_channels=self.in_c,out_channels=8,kernel_size=5,padding=2),
            #最大池化输出shape为(-1,8,3,3)
            nn.MaxPool2d(kernel_size= 2,stride=2),
            nn.ReLU(True),
        )
        #利用全连接层回归theta参数
        self.fc_loc = nn.Sequential(
            nn.Linear(8*3*3,32),
            nn.ReLU(True),
            nn.Linear(32,2*3)
        )

    def forward(self,x):
        
        #提取输入图像中的特征
        xs = self.localization(x)
        xs = xs.view(-1,8*3*3)
        #回归theta参数
        theta = self.fc_loc(xs)
        theta = theta.view(-1,2,3)

        #利用theta参数计算变换后图片的位置
        grid = F.affine_grid(theta,x.size(),align_corners=True)
        #根据输入图片计算变换后图片位置填充的像素值
        x = F.grid_sample(x,grid,align_corners=True)
        return x

