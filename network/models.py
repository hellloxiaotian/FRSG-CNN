from turtle import forward
from cv2 import inpaint
from sqlalchemy import false
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import torch.nn.init as init

import torchvision.models as models
from torchvision.utils import save_image
from torchvision.transforms.functional import hflip


#resnet18
class Model_0(nn.Module):      
    def __init__(self,num_class=7,device='cpu'):
        super(Model_0,self).__init__()
        
        self.resnet=models.resnet18()#定义resnet18模型
        checkpoint=torch.load('models/resnet18_msceleb.pth',map_location=device)#加载指定路径的模型
        self.resnet.load_state_dict(checkpoint['state_dict'],strict=True)#加载模型参数
        
    def forward(self,x):

        x1=self.resnet(x)
        
        return x1,x1
        
#resnet18+五官拼接融合
class Model_1(nn.Module):      
    def __init__(self,num_class=7,device='cpu'):
        super(Model_1,self).__init__()
        
        self.resnet=models.resnet18()#定义resnet18模型
        self.resnet1=models.resnet18()#定义resnet18模型
        self.resnet2=models.resnet18()#定义resnet18模型
        self.resnet3=models.resnet18()#定义resnet18模型
        self.resnet4=models.resnet18()#定义resnet18模型
        self.resnet5=models.resnet18()#定义resnet18模型
        checkpoint=torch.load('models/resnet18_msceleb.pth',map_location=device)#加载指定路径的模型
        self.resnet.load_state_dict(checkpoint['state_dict'],strict=True)#加载模型参数

        self.features1=nn.Sequential(*list(self.resnet.children())[:-3])#定义特征提取器1
        self.features2=nn.Sequential(*list(self.resnet1.children())[:-3])#定义特征提取器2
        self.features3=nn.Sequential(*list(self.resnet2.children())[:-3])#定义特征提取器3
        self.features4=nn.Sequential(*list(self.resnet3.children())[:-3])#定义特征提取器4
        self.features5=nn.Sequential(*list(self.resnet4.children())[:-3])#定义特征提取器5
        self.features6=nn.Sequential(*list(self.resnet5.children())[:-3])#定义特征提取器6
        self.features7=nn.Sequential(*list(self.resnet.children())[-3:-2])#定义特征提取器7
        

        # 定义五官权重参数作为可学习参数
        self.w1 = nn.Parameter(torch.tensor(0.5))
        self.h2 = nn.Parameter(torch.tensor(0.35))
        self.h3 = nn.Parameter(torch.tensor(0.65))

        self.fc=nn.Linear(512,num_class)#定义全连接层
        
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))#定义一个全局平均池化层
        
        self.relu=nn.ReLU()#定义ReLU激活函数
        
    def forward(self,x):
        w=x.size(3)#获取张量的宽度
        h=x.size(2)#获取张量的高度
        w_2=int(w * self.w1)#获取图片中间线
        h_2=int(h * self.h2)#获取图片眼部水平线
        h_3=int(h * self.h3)#获取图片嘴鼻分界线

        x2=x[:,:,0:h_2,0:w_2]#截取左眼图像
        x3=x[:,:,0:h_2,w_2:w]#截取右眼图像
        x4=x[:,:,h_2:h_3,0:w_2]#截取左脸图像
        x5=x[:,:,h_2:h_3,w_2:w]#截取右脸图像
        x6=x[:,:,h_3:h,:]#截取嘴部图像
        
        x1=self.features1(x) # 全局特征        
        x2=self.features2(x2) #左眼特征
        x3=self.features3(x3) #右眼特征
        x4=self.features4(x4) #右脸特征
        x5=self.features5(x5) #右脸特征
        x6=self.features6(x6) #嘴部特征
        
        x7=torch.cat([x2,x3],dim=3) #拼接左右眼部特征
        x8=torch.cat([x4,x5],dim=3) #拼接左右脸部特征
        x9=torch.cat([x7,x8,x6],dim=2) #拼接眼部+脸部+嘴部特征
        x9=F.interpolate(x9, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=True)
        
        x1=x1+x9
        x1=self.features7(x1)#特征提取
        x1=self.avgpool(x1)#平均池化
        x1=x1.view(x.size(0),-1)#降维(b,c*h*w)
        x1=self.fc(x1)#全连接
        
        return x1,x1
        
#resnet18+五官拼接融合+对称共用特征提取器
class Model_2(nn.Module):      
    def __init__(self,num_class=7,device='cpu'):
        super(Model_2,self).__init__()
        
        self.resnet=models.resnet18()#定义resnet18模型
        self.resnet1=models.resnet18()#定义resnet18模型
        self.resnet2=models.resnet18()#定义resnet18模型
        self.resnet3=models.resnet18()#定义resnet18模型
        checkpoint=torch.load('models/resnet18_msceleb.pth',map_location=device)#加载指定路径的模型
        self.resnet.load_state_dict(checkpoint['state_dict'],strict=True)#加载模型参数

        self.features1=nn.Sequential(*list(self.resnet.children())[:-3])#定义特征提取器1
        self.features2=nn.Sequential(*list(self.resnet1.children())[:-3])#定义特征提取器2
        self.features3=nn.Sequential(*list(self.resnet2.children())[:-3])#定义特征提取器3
        self.features4=nn.Sequential(*list(self.resnet3.children())[:-3])#定义特征提取器4
        self.features5=nn.Sequential(*list(self.resnet.children())[-3:-2])#定义特征提取器7
        

        # 定义五官权重参数作为可学习参数
        self.w1 = nn.Parameter(torch.tensor(0.5))
        self.h2 = nn.Parameter(torch.tensor(0.35))
        self.h3 = nn.Parameter(torch.tensor(0.65))

        self.fc=nn.Linear(512,num_class)#定义全连接层
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))#定义一个全局平均池化层        
        self.relu=nn.ReLU()#定义ReLU激活函数
        
    def forward(self,x):
        w=x.size(3)#获取张量的宽度
        h=x.size(2)#获取张量的高度
        w_2=int(w * self.w1)#获取图片中间线
        h_2=int(h * self.h2)#获取图片眼部水平线
        h_3=int(h * self.h3)#获取图片嘴鼻分界线

        x2=x[:,:,0:h_2,0:w_2]#截取左眼图像
        x3=x[:,:,0:h_2,w_2:w]#截取右眼图像
        x4=x[:,:,h_2:h_3,0:w_2]#截取左脸图像
        x5=x[:,:,h_2:h_3,w_2:w]#截取右脸图像
        x6=x[:,:,h_3:h,:]#截取嘴部图像
        
        x1=self.features1(x) # 全局特征        
        x2=self.features2(x2) #左眼特征
        x3=self.features2(x3) #右眼特征
        x4=self.features3(x4) #右脸特征
        x5=self.features3(x5) #右脸特征
        x6=self.features4(x6) #嘴部特征
        
        x7=torch.cat([x2,x3],dim=3) #拼接左右眼部特征
        x8=torch.cat([x4,x5],dim=3) #拼接左右脸部特征
        x9=torch.cat([x7,x8,x6],dim=2) #拼接眼部+脸部+嘴部特征
        x9=F.interpolate(x9, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=True)
        
        x1=x1+x9
        
        x1=self.features5(x1)#特征提取
        x1=self.avgpool(x1)#平均池化
        x1=x1.view(x.size(0),-1)#降维(b,c*h*w)
        x1=self.fc(x1)#全连接
        
        return x1
        
#resnet18+五官拼接融合+图像配准
class Model_3(nn.Module):      
    def __init__(self,num_class=7,device='cpu'):
        super(Model_3,self).__init__()
        
        self.resnet=models.resnet18()#定义resnet18模型
        self.resnet1=models.resnet18()#定义resnet18模型
        self.resnet2=models.resnet18()#定义resnet18模型
        self.resnet3=models.resnet18()#定义resnet18模型
        self.resnet4=models.resnet18()#定义resnet18模型
        self.resnet5=models.resnet18()#定义resnet18模型
        self.resnet6=models.resnet18()#定义resnet18模型
        
        checkpoint=torch.load('models/resnet18_msceleb.pth',map_location=device)#加载指定路径的模型
        
        self.resnet.load_state_dict(checkpoint['state_dict'],strict=True)#加载模型参数
        self.resnet1.load_state_dict(checkpoint['state_dict'],strict=True)#加载模型参数
        self.resnet2.load_state_dict(checkpoint['state_dict'],strict=True)#加载模型参数
        self.resnet3.load_state_dict(checkpoint['state_dict'],strict=True)#加载模型参数
        self.resnet4.load_state_dict(checkpoint['state_dict'],strict=True)#加载模型参数
        self.resnet5.load_state_dict(checkpoint['state_dict'],strict=True)#加载模型参数
        self.resnet6.load_state_dict(checkpoint['state_dict'],strict=True)#加载模型参数

        self.features1=nn.Sequential(*list(self.resnet.children())[:-3])#定义特征提取器1--全局
        self.features2=nn.Sequential(*list(self.resnet1.children())[:-3])#定义特征提取器2--左眼
        self.features3=nn.Sequential(*list(self.resnet2.children())[:-3])#定义特征提取器3--右眼
        self.features4=nn.Sequential(*list(self.resnet3.children())[:-3])#定义特征提取器4--左脸
        #self.features5=nn.Sequential(*list(self.resnet4.children())[:-3])#定义特征提取器5--鼻子
        self.features6=nn.Sequential(*list(self.resnet5.children())[:-3])#定义特征提取器6--右脸
        self.features7=nn.Sequential(*list(self.resnet6.children())[:-3])#定义特征提取器6--嘴部
        self.features8=nn.Sequential(*list(self.resnet.children())[-3:-2])#定义特征提取器7
        

        # 定义五参考线作为可学习参数
        self.w1 = nn.Parameter(torch.tensor(0.5))
        self.w2 = nn.Parameter(torch.tensor(0.75))

        self.h1 = nn.Parameter(torch.tensor(0.5))
        self.h2 = nn.Parameter(torch.tensor(0.65))
        
        #回归层
        self.fc_loc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 3 * 2)#仿射变换
        )

        #定义全连接层
        self.fc=nn.Linear(512,num_class)
        
        #定义一个全局平均池化层
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        
        #定义ReLU激活函数
        self.relu=nn.ReLU()
        
    def forward(self,x):
    
        w=x.size(3)#获取张量的宽度
        h=x.size(2)#获取张量的高度
        w_1=int(w * self.w1)#获取图片脸鼻分界线（左）
        w_2=int(w * self.w2)#获取图片脸鼻分界线（右）
        h_1=int(h * self.h1)#获取图片眼鼻分界线
        h_2=int(h * self.h2)#获取图片嘴鼻分界线

        x2=x[:,:,0:h_1,0:w_1]#截取左眼图像
        x3=x[:,:,0:h_1,w_1:w]#截取右眼图像
        x4=x[:,:,h_1:h_2,0:w_1]#截取左脸图像
        #x5=x[:,:,h_1:h_2,w_1:w_2]#截取鼻子图像
        x6=x[:,:,h_1:h_2,w_1:w]#截取右脸图像
        x7=x[:,:,h_2:h,:]#截取嘴部图像
        
        x1=self.features1(x) # 全局特征        
        x2=self.features2(x2) #左眼特征
        x3=self.features3(x3) #右眼特征
        x4=self.features4(x4) #右脸特征
        #x5=self.features5(x5) #鼻子特征
        x6=self.features6(x6) #右脸特征
        x7=self.features7(x7) #嘴部特征
        
        x8=torch.cat([x2,x3],dim=3) #拼接左右眼部特征
        x8=F.interpolate(x8, size=(x8.size(2), x1.size(3)), mode='bilinear', align_corners=True)
        x9=torch.cat([x4,x6],dim=3) #拼接左右脸部特征
        x9=F.interpolate(x9, size=(x9.size(2), x1.size(3)), mode='bilinear', align_corners=True)
        x10=torch.cat([x8,x9,x7],dim=2) #拼接眼部+脸部+嘴部特征
        x10=F.interpolate(x10, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=True)
        
        xs=self.features8(x1)#特征提取
        #回归层
        xs = xs.view(x1.size(0), -1)#降维(b,c*h*w)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        #利用theta参数计算变换后图片的位置
        grid = F.affine_grid(theta, x1.size(), align_corners=True)
        #根据输入图片计算变换后图片位置填充的像素值
        x11 = F.grid_sample(x10, grid, align_corners=True)
        
        x1=x1+x11
        
        x1=self.features8(x1)#特征提取
        x1=self.avgpool(x1)#平均池化
        x1=x1.view(x.size(0),-1)#降维(b,c*h*w)
        x1=self.fc(x1)#全连接
        
        heads = x1
        
        return x1,heads
    
        
#resnet18+stn
class Model_4(nn.Module):      
    def __init__(self,num_class=7,device='cpu'):
        super(Model_4,self).__init__()
        
        self.resnet=models.resnet18()#定义resnet18模型
        
        checkpoint=torch.load('models/resnet18_msceleb.pth',map_location=device)#加载指定路径的模型
        
        self.resnet.load_state_dict(checkpoint['state_dict'],strict=True)#加载模型参数

        self.features1=nn.Sequential(*list(self.resnet.children())[:-3])#定义特征提取器1--全局
        self.features2=nn.Sequential(*list(self.resnet.children())[-3:-2])#定义特征提取器2

        # 定义五参考线作为可学习参数
        self.w1 = nn.Parameter(torch.tensor(0.25))
        self.w2 = nn.Parameter(torch.tensor(0.75))

        self.h1 = nn.Parameter(torch.tensor(0.5))
        self.h2 = nn.Parameter(torch.tensor(0.65))
        
        #回归层
        self.fc_loc = nn.Sequential(
            nn.Linear(256 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 3 * 2)#仿射变换
        )

        #定义全连接层
        self.fc=nn.Linear(512,num_class)
        
        #定义一个全局平均池化层
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        
        #定义ReLU激活函数
        self.relu=nn.ReLU()
        
    def forward(self,x):
        
        x1=self.features1(x) # 全局特征     
        
        #定位
        xs = self.features2(x1)#特征提取
        #回归
        xs = x1.view(x1.size(0), -1)#降维(b,c*h*w)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        #利用theta参数计算变换后图片的位置
        grid = F.affine_grid(theta, x1.size(), align_corners=True)
        #根据输入图片计算变换后图片位置填充的像素值
        x1 = F.grid_sample(x1, grid, align_corners=True)
        
        x1=self.features2(x1)#特征提取
        x1=self.avgpool(x1)#平均池化
        x1=x1.view(x1.size(0),-1)#降维(b,c*h*w)
        x1=self.fc(x1)#全连接
        
        return x1
        
#resnet18+五官拼接融合+图像配准
class Model_5(nn.Module):      
    def __init__(self,num_class=7,device='cpu'):
        super(Model_5,self).__init__()
        
        self.resnet=models.resnet18()#定义resnet18模型
        self.resnet1=models.resnet18()#定义resnet18模型
        self.resnet2=models.resnet18()#定义resnet18模型
        self.resnet3=models.resnet18()#定义resnet18模型
        self.resnet4=models.resnet18()#定义resnet18模型
        self.resnet5=models.resnet18()#定义resnet18模型
        self.resnet6=models.resnet18()#定义resnet18模型
        
        checkpoint=torch.load('models/resnet18_msceleb.pth',map_location=device)#加载指定路径的模型
        
        self.resnet.load_state_dict(checkpoint['state_dict'],strict=True)#加载模型参数
        self.resnet1.load_state_dict(checkpoint['state_dict'],strict=True)#加载模型参数
        self.resnet2.load_state_dict(checkpoint['state_dict'],strict=True)#加载模型参数
        self.resnet3.load_state_dict(checkpoint['state_dict'],strict=True)#加载模型参数
        self.resnet4.load_state_dict(checkpoint['state_dict'],strict=True)#加载模型参数
        self.resnet5.load_state_dict(checkpoint['state_dict'],strict=True)#加载模型参数
        self.resnet6.load_state_dict(checkpoint['state_dict'],strict=True)#加载模型参数

        self.features1=nn.Sequential(*list(self.resnet.children())[:-3])#定义特征提取器1--全局
        self.features2=nn.Sequential(*list(self.resnet1.children())[:-3])#定义特征提取器2--左眼
        self.features3=nn.Sequential(*list(self.resnet2.children())[:-3])#定义特征提取器3--右眼
        self.features4=nn.Sequential(*list(self.resnet3.children())[:-3])#定义特征提取器4--左脸
        #self.features5=nn.Sequential(*list(self.resnet4.children())[:-3])#定义特征提取器5--鼻子
        self.features6=nn.Sequential(*list(self.resnet5.children())[:-3])#定义特征提取器6--右脸
        self.features7=nn.Sequential(*list(self.resnet6.children())[:-3])#定义特征提取器6--嘴部
        self.features8=nn.Sequential(*list(self.resnet.children())[-3:-2])#定义特征提取器7
        

        # 定义五参考线作为可学习参数
        self.w1 = nn.Parameter(torch.tensor(0.5))
        self.w2 = nn.Parameter(torch.tensor(0.75))

        self.h1 = nn.Parameter(torch.tensor(0.5))
        self.h2 = nn.Parameter(torch.tensor(0.65))
        
        #回归层
        self.fc_loc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 3 * 2)#仿射变换
        )

        #定义全连接层
        self.fc=nn.Linear(512,num_class)
        self.fc2=nn.Linear(256,num_class)
        self.fc3=nn.Linear(256,num_class)
        self.fc4=nn.Linear(256,num_class)
        self.fc6=nn.Linear(256,num_class)
        self.fc7=nn.Linear(256,num_class)
        
        #定义一个全局平均池化层
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        
        #定义ReLU激活函数
        self.relu=nn.ReLU()
        
    def forward(self,x):
    
        w=x.size(3)#获取张量的宽度
        h=x.size(2)#获取张量的高度
        w_1=int(w * self.w1)#获取图片脸鼻分界线（左）
        w_2=int(w * self.w2)#获取图片脸鼻分界线（右）
        h_1=int(h * self.h1)#获取图片眼鼻分界线
        h_2=int(h * self.h2)#获取图片嘴鼻分界线

        x2=x[:,:,0:h_1,0:w_1]#截取左眼图像
        x3=x[:,:,0:h_1,w_1:w]#截取右眼图像
        x4=x[:,:,h_1:h_2,0:w_1]#截取左脸图像
        #x5=x[:,:,h_1:h_2,w_1:w_2]#截取鼻子图像
        x6=x[:,:,h_1:h_2,w_1:w]#截取右脸图像
        x7=x[:,:,h_2:h,:]#截取嘴部图像
        
        x1=self.features1(x) # 全局特征        
        x2=self.features2(x2) #左眼特征
        x3=self.features3(x3) #右眼特征
        x4=self.features4(x4) #右脸特征
        #x5=self.features5(x5) #鼻子特征
        x6=self.features6(x6) #右脸特征
        x7=self.features7(x7) #嘴部特征
        
        x8=torch.cat([x2,x3],dim=3) #拼接左右眼部特征
        x8=F.interpolate(x8, size=(x8.size(2), x1.size(3)), mode='bilinear', align_corners=True)
        x9=torch.cat([x4,x6],dim=3) #拼接左右脸部特征
        x9=F.interpolate(x9, size=(x9.size(2), x1.size(3)), mode='bilinear', align_corners=True)
        x10=torch.cat([x8,x9,x7],dim=2) #拼接眼部+脸部+嘴部特征
        x10=F.interpolate(x10, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=True)
        


        x2=self.avgpool(x2)#平均池化
        x2=x2.view(x2.size(0),-1)
        x2=self.fc2(x2)
        
        x3=self.avgpool(x3)#平均池化
        x3=x3.view(x.size(0),-1)
        x3=self.fc3(x3)
        
        x4=self.avgpool(x4)#平均池化
        x4=x4.view(x.size(0),-1)
        x4=self.fc4(x4)
        
        x6=self.avgpool(x6)#平均池化
        x6=x6.view(x.size(0),-1)
        x6=self.fc6(x6)
        
        x7=self.avgpool(x7)#平均池化
        x7=x7.view(x.size(0),-1)
        x7=self.fc7(x7)
        
        heads = x2 + x3 + x4 + x6 + x7
        
        xs=self.features8(x1)#特征提取
        #回归层
        xs = xs.view(x1.size(0), -1)#降维(b,c*h*w)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        #利用theta参数计算变换后图片的位置
        grid = F.affine_grid(theta, x1.size(), align_corners=True)
        #根据输入图片计算变换后图片位置填充的像素值
        x11 = F.grid_sample(x10, grid, align_corners=True)
        
        # toPIL = transforms.ToPILImage()
        # local_fm = toPIL((x11.squeeze()[:3,:,:] + 1) * 127.5)
        # global_fm = toPIL((x1.squeeze()[:3,:,:] + 1) * 127.5)
        # local_fm.save("local.jpg")
        # global_fm.save("global.jpg")
        save_image(x1, "feature")
        print("x1:{}, x11:{}".format(x1.size(), x11.size()))

        x1=x1+x11
        
        x1=self.features8(x1)#特征提取
        x1=self.avgpool(x1)#平均池化
        x1=x1.view(x.size(0),-1)#降维(b,c*h*w)
        x1=self.fc(x1)#全连接
        
        return x1,heads
        
#resnet18 + 五官拼接融合 + 五官STN
class Model_6(nn.Module):      
    def __init__(self,num_class=7,device='cpu'):
        super(Model_6, self).__init__()
        
        self.resnet = models.resnet18()                                                # 定义resnet18模型
        checkpoint = torch.load('models/resnet18_msceleb.pth',map_location = device)   # 加载指定路径的模型
        self.resnet.load_state_dict(checkpoint['state_dict'],strict = True)            # 加载模型参数

        self.features0 = nn.Sequential(*list(self.resnet.children())[:-3])    # 定义特征提取器0--全局
        self.features1 = nn.Sequential(*list(self.resnet.children())[:-3])    # 定义特征提取器1--眼部
        self.features2 = nn.Sequential(*list(self.resnet.children())[:-3])    # 定义特征提取器2--脸部
        self.features3 = nn.Sequential(*list(self.resnet.children())[:-3])    # 定义特征提取器3--嘴部
        
        self.features4 = nn.Sequential(*list(self.resnet.children())[-3:-2])  # 定义特征提取器4

        # 定义五参考线作为可学习参数
        self.h1 = nn.Parameter(torch.tensor(0.4))
        self.h2 = nn.Parameter(torch.tensor(0.65))
        
        # 回归层
        self.fc_loc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 3 * 2)  # 仿射变换
        )

        # 定义全连接层
        self.fc = nn.Linear(512, num_class)
        
        # 定义一个全局平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        # 定义ReLU激活函数
        self.relu = nn.ReLU()
        
    def forward(self,x):
    
        h = x.size(2)            # 获取张量的高度
        h_1 = int(h * self.h1)   # 获取图片眼鼻分界线
        h_2 = int(h * self.h2)   # 获取图片嘴鼻分界线
        
        # 截取图像
        x1 = x[:, :, 0 : h_1, :]    # 截取眼部图像
        x2 = x[:, :, h_1 : h_2, :]  # 截取脸部图像
        x3 = x[:, :, h_2 : h, :]    # 截取嘴部图像
        
        # 特征提取
        x0 = self.features0(x)  # 全局特征        
        x1 = self.features1(x1) # 眼部特征
        x2 = self.features2(x2) # 脸部特征
        x3 = self.features3(x3) # 嘴部特征
        
        # 特征拼接
        x4 = torch.cat([x1, x2, x3], dim = 2)           # 拼接眼部+脸部+嘴部特征      
        x4 = F.interpolate(x4, size = (x0.size(2), x0.size(3)), mode='bilinear', align_corners=True)  
        
        xs = self.features4(x4)         # 特征提取
        xs = xs.view(x4.size(0), -1)    # 降维(b,c*h*w)
        # 回归层
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # 利用theta参数计算变换后图片的位置
        grid = F.affine_grid(theta, x4.size(), align_corners=True)
        # 根据输入图片计算变换后图片位置填充的像素值
        x4 = F.grid_sample(x4, grid, align_corners=True)
        
        x0 = x0 + x4
        
        x0 = self.features4(x0)        #  特征提取
        x0 = self.avgpool(x0)          # 平均池化
        x0 = x0.view(x0.size(0),-1)    # 降维(b,c*h*w)
        x0 = self.fc(x0)               # 全连接
        
        return x0, x4
        
#resnet18 + 五官拼接融合（3） + 全局STN
class Model_7(nn.Module):      
    def __init__(self,num_class=7,device='cpu'):
        super(Model_7, self).__init__()
        
        self.resnet = models.resnet18()                                                # 定义resnet18模型
        checkpoint = torch.load('models/resnet18_msceleb.pth',map_location = device)   # 加载指定路径的模型
        self.resnet.load_state_dict(checkpoint['state_dict'],strict = True)            # 加载模型参数

        self.features0 = nn.Sequential(*list(self.resnet.children())[:-3])    # 定义特征提取器0--全局
        self.features1 = nn.Sequential(*list(self.resnet.children())[:-3])    # 定义特征提取器1--眼部
        self.features2 = nn.Sequential(*list(self.resnet.children())[:-3])    # 定义特征提取器2--脸部
        self.features3 = nn.Sequential(*list(self.resnet.children())[:-3])    # 定义特征提取器3--嘴部
        
        self.features4 = nn.Sequential(*list(self.resnet.children())[-3:-2])  # 定义特征提取器4

        # 定义五参考线作为可学习参数
        self.h1 = nn.Parameter(torch.tensor(0.4))
        self.h2 = nn.Parameter(torch.tensor(0.65))
        
        # 回归层
        self.fc_loc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 3 * 2)  # 仿射变换
        )

        # 定义全连接层
        self.fc = nn.Linear(512, num_class)
        
        # 定义一个全局平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        # 定义ReLU激活函数
        self.relu = nn.ReLU()
        
    def forward(self,x):
    
        h = x.size(2)            # 获取张量的高度
        h_1 = int(h * self.h1)   # 获取图片眼鼻分界线
        h_2 = int(h * self.h2)   # 获取图片嘴鼻分界线
        
        # 截取图像
        x1 = x[:, :, 0 : h_1, :]    # 截取眼部图像
        x2 = x[:, :, h_1 : h_2, :]  # 截取脸部图像
        x3 = x[:, :, h_2 : h, :]    # 截取嘴部图像
        
        # 特征提取
        x0 = self.features0(x)  # 全局特征        
        x1 = self.features1(x1) # 眼部特征
        x2 = self.features2(x2) # 脸部特征
        x3 = self.features3(x3) # 嘴部特征
        
        # 特征拼接
        x4 = torch.cat([x1, x2, x3], dim = 2)           # 拼接眼部+脸部+嘴部特征      
        x4 = F.interpolate(x4, size = (x0.size(2), x0.size(3)), mode='bilinear', align_corners=True)  
        
        xs = self.features4(x0)         # 特征提取
        xs = xs.view(x4.size(0), -1)    # 降维(b,c*h*w)
        
        # 回归层
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # 利用theta参数计算变换后图片的位置
        grid = F.affine_grid(theta, x4.size(), align_corners=True)
        # 根据输入图片计算变换后图片位置填充的像素值
        x4 = F.grid_sample(x0, grid, align_corners=True)
        
        x0 = x0 + x4
        
        x0 = self.features4(x0)        #  特征提取
        x0 = self.avgpool(x0)          # 平均池化
        x0 = x0.view(x0.size(0),-1)    # 降维(b,c*h*w)
        x0 = self.fc(x0)               # 全连接
        
        return x0, x0
        
#resnet18 + 五官拼接融合（3） + 配准（5-0）
class Model_8(nn.Module):      
    def __init__(self,num_class=7,device='cpu'):
        super(Model_8, self).__init__()
        
        self.resnet = models.resnet18()                                                # 定义resnet18模型
        checkpoint = torch.load('models/resnet18_msceleb.pth',map_location = device)   # 加载指定路径的模型
        self.resnet.load_state_dict(checkpoint['state_dict'],strict = True)            # 加载模型参数

        self.features0 = nn.Sequential(*list(self.resnet.children())[:-3])    # 定义特征提取器0--全局
        self.features1 = nn.Sequential(*list(self.resnet.children())[:-3])    # 定义特征提取器1--眼部
        self.features2 = nn.Sequential(*list(self.resnet.children())[:-3])    # 定义特征提取器2--脸部
        self.features3 = nn.Sequential(*list(self.resnet.children())[:-3])    # 定义特征提取器3--嘴部
        
        self.features4 = nn.Sequential(*list(self.resnet.children())[-3:-2])  # 定义特征提取器4

        # 定义五参考线作为可学习参数
        self.h1 = nn.Parameter(torch.tensor(0.4))
        self.h2 = nn.Parameter(torch.tensor(0.65))
        
        # 回归层
        self.fc_loc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 3 * 2)  # 仿射变换
        )

        # 定义全连接层
        self.fc = nn.Linear(512, num_class)
        
        # 定义一个全局平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        # 定义ReLU激活函数
        self.relu = nn.ReLU()
        
    def forward(self,x):
    
        h = x.size(2)            # 获取张量的高度
        h_1 = int(h * self.h1)   # 获取图片眼鼻分界线
        h_2 = int(h * self.h2)   # 获取图片嘴鼻分界线
        
        # 截取图像
        x1 = x[:, :, 0 : h_1, :]    # 截取眼部图像
        x2 = x[:, :, h_1 : h_2, :]  # 截取脸部图像
        x3 = x[:, :, h_2 : h, :]    # 截取嘴部图像
        
        # 特征提取
        x0 = self.features0(x)  # 全局特征        
        x1 = self.features1(x1) # 眼部特征
        x2 = self.features2(x2) # 脸部特征
        x3 = self.features3(x3) # 嘴部特征
        
        # 特征拼接
        x4 = torch.cat([x1, x2, x3], dim = 2)           # 拼接眼部+脸部+嘴部特征      
        x5 = F.interpolate(x4, size = (x0.size(2), x0.size(3)), mode='bilinear', align_corners=True)  
        
        xs = self.features4(x0)         # 特征提取
        xs = xs.view(x4.size(0), -1)    # 降维(b,c*h*w)
        
        # 回归层
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # 利用theta参数计算变换后图片的位置
        grid = F.affine_grid(theta, x0.size(), align_corners=True)
        # 根据输入图片计算变换后图片位置填充的像素值
        x5 = F.grid_sample(x5, grid, align_corners=True)
        
        x0 = x0 + x5
        
        x0 = self.features4(x0)        #  特征提取
        x0 = self.avgpool(x0)          # 平均池化
        x0 = x0.view(x0.size(0),-1)    # 降维(b,c*h*w)
        x0 = self.fc(x0)               # 全连接
        
        return x0, x0
        
#resnet18 + 五官拼接融合（3） + 配准（0-5）
class Model_9(nn.Module):      
    def __init__(self,num_class=7,device='cpu'):
        super(Model_9, self).__init__()
        
        self.resnet = models.resnet18()                                                # 定义resnet18模型
        checkpoint = torch.load('models/resnet18_msceleb.pth',map_location = device)   # 加载指定路径的模型
        self.resnet.load_state_dict(checkpoint['state_dict'],strict = True)            # 加载模型参数

        self.features0 = nn.Sequential(*list(self.resnet.children())[:-3])    # 定义特征提取器0--全局
        self.features1 = nn.Sequential(*list(self.resnet.children())[:-3])    # 定义特征提取器1--眼部
        self.features2 = nn.Sequential(*list(self.resnet.children())[:-3])    # 定义特征提取器2--脸部
        self.features3 = nn.Sequential(*list(self.resnet.children())[:-3])    # 定义特征提取器3--嘴部
        
        self.features4 = nn.Sequential(*list(self.resnet.children())[-3:-2])  # 定义特征提取器4

        # 定义五参考线作为可学习参数
        self.h1 = nn.Parameter(torch.tensor(0.4))
        self.h2 = nn.Parameter(torch.tensor(0.65))
        
        # 回归层
        self.fc_loc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 3 * 2)  # 仿射变换
        )

        # 定义全连接层
        self.fc = nn.Linear(512, num_class)
        
        # 定义一个全局平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        # 定义ReLU激活函数
        self.relu = nn.ReLU()
        
    def forward(self,x):
    
        h = x.size(2)            # 获取张量的高度
        h_1 = int(h * self.h1)   # 获取图片眼鼻分界线
        h_2 = int(h * self.h2)   # 获取图片嘴鼻分界线
        
        # 截取图像
        x1 = x[:, :, 0 : h_1, :]    # 截取眼部图像
        x2 = x[:, :, h_1 : h_2, :]  # 截取脸部图像
        x3 = x[:, :, h_2 : h, :]    # 截取嘴部图像
        
        # 特征提取
        x0 = self.features0(x)  # 全局特征        
        x1 = self.features1(x1) # 眼部特征
        x2 = self.features2(x2) # 脸部特征
        x3 = self.features3(x3) # 嘴部特征
        
        # 特征拼接
        x4 = torch.cat([x1, x2, x3], dim = 2)           # 拼接眼部+脸部+嘴部特征      
        x5 = F.interpolate(x4, size = (x0.size(2), x0.size(3)), mode='bilinear', align_corners=True)  
        
        xs = self.features4(x5)         # 特征提取
        xs = xs.view(x4.size(0), -1)    # 降维(b,c*h*w)
        
        # 回归层
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # 利用theta参数计算变换后图片的位置
        grid = F.affine_grid(theta, x0.size(), align_corners=True)
        # 根据输入图片计算变换后图片位置填充的像素值
        x5 = F.grid_sample(x0, grid, align_corners=True)
        
        x0 = x0 + x5
        
        x0 = self.features4(x0)        #  特征提取
        x0 = self.avgpool(x0)          # 平均池化
        x0 = x0.view(x0.size(0),-1)    # 降维(b,c*h*w)
        x0 = self.fc(x0)               # 全连接
        
        return x0, x0

#resnet18 + 五官拼接融合（3)
class Model_10(nn.Module):      
    def __init__(self,num_class=7,device='cpu'):
        super(Model_10, self).__init__()
        
        self.resnet = models.resnet18()                                                # 定义resnet18模型
        checkpoint = torch.load('models/resnet18_msceleb.pth',map_location = device)   # 加载指定路径的模型
        self.resnet.load_state_dict(checkpoint['state_dict'],strict = True)            # 加载模型参数

        self.features0 = nn.Sequential(*list(self.resnet.children())[:-3])    # 定义特征提取器0--全局
        self.features1 = nn.Sequential(*list(self.resnet.children())[:-3])    # 定义特征提取器1--眼部
        self.features2 = nn.Sequential(*list(self.resnet.children())[:-3])    # 定义特征提取器2--脸部
        self.features3 = nn.Sequential(*list(self.resnet.children())[:-3])    # 定义特征提取器3--嘴部
        
        self.features4 = nn.Sequential(*list(self.resnet.children())[-3:-2])  # 定义特征提取器4

        # 定义五参考线作为可学习参数
        self.h1 = nn.Parameter(torch.tensor(0.4))
        self.h2 = nn.Parameter(torch.tensor(0.65))
        
        # 回归层
        self.fc_loc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 3 * 2)  # 仿射变换
        )

        # 定义全连接层
        self.fc = nn.Linear(512, num_class)
        
        # 定义一个全局平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        # 定义ReLU激活函数
        self.relu = nn.ReLU()
        
    def forward(self,x):
    
        h = x.size(2)            # 获取张量的高度
        h_1 = int(h * self.h1)   # 获取图片眼鼻分界线
        h_2 = int(h * self.h2)   # 获取图片嘴鼻分界线
        
        # 截取图像
        x1 = x[:, :, 0 : h_1, :]    # 截取眼部图像
        x2 = x[:, :, h_1 : h_2, :]  # 截取脸部图像
        x3 = x[:, :, h_2 : h, :]    # 截取嘴部图像
        
        # 特征提取
        x0 = self.features0(x)  # 全局特征        
        x1 = self.features1(x1) # 眼部特征
        x2 = self.features2(x2) # 脸部特征
        x3 = self.features3(x3) # 嘴部特征
        
        # 特征拼接
        x4 = torch.cat([x1, x2, x3], dim = 2)           # 拼接眼部+脸部+嘴部特征      
        x5 = F.interpolate(x4, size = (x0.size(2), x0.size(3)), mode='bilinear', align_corners=True)  
        
        x0 = x0 + x5
        
        x0 = self.features4(x0)        #  特征提取
        x0 = self.avgpool(x0)          # 平均池化
        x0 = x0.view(x0.size(0),-1)    # 降维(b,c*h*w)
        x0 = self.fc(x0)               # 全连接
        
        return x0, x0
        
#resnet18 + 五官拼接融合（3) + 配准（5-0）
class Model_11(nn.Module):      
    def __init__(self,num_class=7,device='cpu'):
        super(Model_11, self).__init__()
        
        self.resnet = models.resnet18()                                                # 定义resnet18模型
        checkpoint = torch.load('models/resnet18_msceleb.pth',map_location = device)   # 加载指定路径的模型
        self.resnet.load_state_dict(checkpoint['state_dict'],strict = True)            # 加载模型参数

        self.features0 = nn.Sequential(*list(self.resnet.children())[:-3])    # 定义特征提取器0--全局
        self.features1 = nn.Sequential(*list(self.resnet.children())[:-3])    # 定义特征提取器1--眼部
        self.features2 = nn.Sequential(*list(self.resnet.children())[:-3])    # 定义特征提取器2--脸部
        self.features3 = nn.Sequential(*list(self.resnet.children())[:-3])    # 定义特征提取器3--嘴部
        
        self.features4 = nn.Sequential(*list(self.resnet.children())[-3:-2])  # 定义特征提取器4

        # 定义五参考线作为可学习参数
        self.h1 = nn.Parameter(torch.tensor(0.4))
        self.h2 = nn.Parameter(torch.tensor(0.65))
        
        # 回归层
        self.fc_loc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 3 * 2)  # 仿射变换
        )

        # 定义全连接层
        self.fc = nn.Linear(512, num_class)
        self.fc2 = nn.Linear(512, num_class)
        
        # 定义一个全局平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        # 定义ReLU激活函数
        self.relu = nn.ReLU()
        
    def forward(self,x):
    
        h = x.size(2)            # 获取张量的高度
        h_1 = int(h * self.h1)   # 获取图片眼鼻分界线
        h_2 = int(h * self.h2)   # 获取图片嘴鼻分界线
        
        # 截取图像
        x1 = x[:, :, 0 : h_1, :]    # 截取眼部图像
        x2 = x[:, :, h_1 : h_2, :]  # 截取脸部图像
        x3 = x[:, :, h_2 : h, :]    # 截取嘴部图像
        
        # 特征提取
        x0 = self.features0(x)  # 全局特征        
        x1 = self.features1(x1) # 眼部特征
        x2 = self.features2(x2) # 脸部特征
        x3 = self.features3(x3) # 嘴部特征
        
        # 特征拼接
        x4 = torch.cat([x1, x2, x3], dim = 2)           # 拼接眼部+脸部+嘴部特征      
        x5 = F.interpolate(x4, size = (x0.size(2), x0.size(3)), mode='bilinear', align_corners=True) 
        
        xs = self.features4(x0)         # 特征提取
        xs = xs.view(x4.size(0), -1)    # 降维(b,c*h*w)
        
        # 回归层
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # 利用theta参数计算变换后图片的位置
        grid = F.affine_grid(theta, x0.size(), align_corners=True)
        # 根据输入图片计算变换后图片位置填充的像素值
        x5 = F.grid_sample(x5, grid, align_corners=True)
        
        x0 = x0 + x5
        
        x5 = self.features4(x5)        #  特征提取
        x5 = self.avgpool(x5)          # 平均池化
        x5 = x5.view(x5.size(0),-1)    # 降维(b,c*h*w)
        x5 = self.fc2(x5)               # 全连接
        
        x0 = self.features4(x0)        #  特征提取
        x0 = self.avgpool(x0)          # 平均池化
        x0 = x0.view(x0.size(0),-1)    # 降维(b,c*h*w)
        x0 = self.fc(x0)               # 全连接
        
        return x0, x5

#resnet18 + 五官拼接融合（6) + 配准（5-0）+ 混合损失
class Model_12(nn.Module):      
    def __init__(self,num_class=7,device='cpu'):
        super(Model_12, self).__init__()
        
        self.resnet0 = models.resnet18()
        self.resnet1 = models.resnet18()   
        self.resnet2 = models.resnet18()   
        self.resnet3 = models.resnet18()   
        self.resnet4 = models.resnet18()   
        self.resnet5 = models.resnet18()   
        self.resnet6 = models.resnet18()                                                # 定义resnet18模型
        checkpoint = torch.load('models/resnet18_msceleb.pth',map_location = device)    # 加载指定路径的模型
        self.resnet0.load_state_dict(checkpoint['state_dict'],strict = True)            # 加载模型参数
        self.resnet1.load_state_dict(checkpoint['state_dict'],strict = True)            # 加载模型参数
        self.resnet2.load_state_dict(checkpoint['state_dict'],strict = True)            # 加载模型参数
        self.resnet3.load_state_dict(checkpoint['state_dict'],strict = True)            # 加载模型参数
        self.resnet4.load_state_dict(checkpoint['state_dict'],strict = True)            # 加载模型参数
        self.resnet5.load_state_dict(checkpoint['state_dict'],strict = True)            # 加载模型参数
        self.resnet6.load_state_dict(checkpoint['state_dict'],strict = True)            # 加载模型参数

        self.features0 = nn.Sequential(*list(self.resnet0.children())[:-3])    # 定义特征提取器0--全局
        self.features1 = nn.Sequential(*list(self.resnet1.children())[:-3])    # 定义特征提取器1--眼部
        self.features2 = nn.Sequential(*list(self.resnet2.children())[:-3])    # 定义特征提取器1--眼部
        self.features3 = nn.Sequential(*list(self.resnet3.children())[:-3])    # 定义特征提取器2--脸部
        self.features4 = nn.Sequential(*list(self.resnet4.children())[:-3])    # 定义特征提取器2--脸部
        self.features5 = nn.Sequential(*list(self.resnet5.children())[:-3])    # 定义特征提取器3--嘴部
        self.features6 = nn.Sequential(*list(self.resnet6.children())[:-3])    # 定义特征提取器3--嘴部
        
        self.features00 = nn.Sequential(*list(self.resnet0.children())[-3:-2])  # 定义特征提取器00
        self.features11 = nn.Sequential(*list(self.resnet1.children())[-3:-2])  # 定义特征提取器00
        self.features22 = nn.Sequential(*list(self.resnet2.children())[-3:-2])  # 定义特征提取器00
        self.features33 = nn.Sequential(*list(self.resnet3.children())[-3:-2])  # 定义特征提取器00
        self.features44 = nn.Sequential(*list(self.resnet4.children())[-3:-2])  # 定义特征提取器00
        self.features55 = nn.Sequential(*list(self.resnet5.children())[-3:-2])  # 定义特征提取器00
        self.features66 = nn.Sequential(*list(self.resnet6.children())[-3:-2])  # 定义特征提取器00
        
        # 回归层
        self.fc_loc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 3 * 2)  # 仿射变换
        )

        # 定义全连接层
        self.fc = nn.Linear(512, num_class)
        self.fc1 = nn.Linear(512, num_class)
        self.fc2 = nn.Linear(512, num_class)
        self.fc3 = nn.Linear(512, num_class)
        self.fc4 = nn.Linear(512, num_class)
        self.fc5 = nn.Linear(512, num_class)
        self.fc6 = nn.Linear(512, num_class)
        
        # 定义一个全局平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        # 定义五参考线作为可学习参数
        self.w1 = nn.Parameter(torch.tensor(0.5))
        self.w2 = nn.Parameter(torch.tensor(0.75))

        self.h1 = nn.Parameter(torch.tensor(0.4))
        self.h2 = nn.Parameter(torch.tensor(0.65))
        
        self.wa = nn.Parameter(torch.tensor(12.0))
        self.wb = nn.Parameter(torch.tensor(8.0))
        
        self.f1 = nn.Parameter(torch.tensor(1.0))
        self.f2 = nn.Parameter(torch.tensor(1.0))
        self.f3 = nn.Parameter(torch.tensor(1.0))
        self.f4 = nn.Parameter(torch.tensor(1.0))
        self.f5 = nn.Parameter(torch.tensor(1.0))
        self.f6 = nn.Parameter(torch.tensor(1.0))
        
    def forward(self,x):
    
        h = x.size(2)            # 获取张量的高度
        W = x.size(3)
        W_1 = int(W * self.w1)
        h_1 = int(h * self.h1)    # 获取图片眼鼻分界线
        h_2 = int(h * self.h2)    # 获取图片嘴鼻分界线
        
        # 截取图像
        x1 = x[:, :, 0 : h_1, 0 : W_1]    # 截取眼部图像
        x2 = x[:, :, 0 : h_1, W_1 : W]    # 截取眼部图像
        x3 = x[:, :, h_1 : h_2, 0 : W_1]  # 截取脸部图像
        x4 = x[:, :, h_1 : h_2, W_1 : W]  # 截取脸部图像
        x5 = x[:, :, h_2 : h, 0 : W_1]    # 截取嘴部图像
        x6 = x[:, :, h_2 : h, W_1 : W]    # 截取嘴部图像
        
        # 特征提取
        x0 = self.features0(x)  # 全局特征        
        x1 = self.features1(x1) # 眼部特征
        x2 = self.features2(x2) # 眼部特征
        x3 = self.features3(x3) # 脸部特征
        x4 = self.features4(x4) # 脸部特征
        x5 = self.features5(x5) # 嘴部特征
        x6 = self.features6(x6) # 嘴部特征
        
        # 特征拼接
        x12 = torch.cat([x1, x2], dim = 3)
        x12 = F.interpolate(x12, size = (x12.size(2), x0.size(3)), mode='bilinear', align_corners=True) 
        x34 = torch.cat([x3, x4], dim = 3)
        x34 = F.interpolate(x34, size = (x34.size(2), x0.size(3)), mode='bilinear', align_corners=True) 
        x56 = torch.cat([x5, x6], dim = 3)
        x56 = F.interpolate(x56, size = (x56.size(2), x0.size(3)), mode='bilinear', align_corners=True) 
        x16 = torch.cat([x12, x34, x56], dim = 2)           # 拼接眼部+脸部+嘴部特征      
        x16 = F.interpolate(x16, size = (x0.size(2), x0.size(3)), mode='bilinear', align_corners=True) 
        
        xs = self.features00(x0)         # 特征提取
        xs = xs.view(xs.size(0), -1)    # 降维(b,c*h*w)
        
        # 回归层
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # 利用theta参数计算变换后图片的位置
        grid = F.affine_grid(theta, x0.size(), align_corners=True)
        # 根据输入图片计算变换后图片位置填充的像素值
        x16 = F.grid_sample(x16, grid, align_corners=True)
        
        x0 = x0 + x16
           
        x0 = self.features00(x0)
        x0 = self.avgpool(x0)          # 平均池化
        x0 = x0.view(x0.size(0),-1)    # 降维(b,c*h*w)
        x0 = self.fc(x0)               # 全连接
        
        x1 = self.features11(x1)
        x1 = self.avgpool(x1)           # 平均池化
        x1 = x1.view(x1.size(0),-1)     # 降维(b,c*h*w)
        x1 = self.fc1(x1)               # 全连接
        
        x2 = self.features22(x2)
        x2 = self.avgpool(x2)           # 平均池化
        x2 = x2.view(x2.size(0),-1)     # 降维(b,c*h*w)
        x2 = self.fc2(x2)               # 全连接
        
        x3 = self.features33(x3)
        x3 = self.avgpool(x3)           # 平均池化
        x3 = x3.view(x3.size(0),-1)     # 降维(b,c*h*w)
        x3 = self.fc3(x3)               # 全连接
        
        x4 = self.features44(x4)
        x4 = self.avgpool(x4)           # 平均池化
        x4 = x4.view(x4.size(0),-1)     # 降维(b,c*h*w)
        x4 = self.fc4(x4)               # 全连接
        
        x5 = self.features55(x5)
        x5 = self.avgpool(x5)           # 平均池化
        x5 = x5.view(x5.size(0),-1)     # 降维(b,c*h*w)
        x5 = self.fc5(x5)               # 全连接
        
        x6 = self.features66(x6)
        x6 = self.avgpool(x6)           # 平均池化
        x6 = x6.view(x6.size(0),-1)     # 降维(b,c*h*w)
        x6 = self.fc6(x6)               # 全连接
        
        xf = x1 * self.f1 + x2 * self.f2 + x3 * self.f3 + x4 * self.f4 + x5 * self.f5 + x6 * self.f6
        
        return x0, xf
        
#resnet18+五官拼接融合+图像配准
class Model_13(nn.Module):      
    def __init__(self,num_class=7,device='cpu'):
        super(Model_13,self).__init__()
        
        self.resnet=models.resnet18()#定义resnet18模型
        
        checkpoint=torch.load('models/resnet18_msceleb.pth',map_location=device)#加载指定路径的模型
        
        self.resnet.load_state_dict(checkpoint['state_dict'],strict=True)#加载模型参数

        self.features1=nn.Sequential(*list(self.resnet.children())[:-3])#定义特征提取器1--全局
        self.features2=nn.Sequential(*list(self.resnet.children())[:-3])#定义特征提取器2--左眼
        self.features3=nn.Sequential(*list(self.resnet.children())[:-3])#定义特征提取器3--右眼
        self.features4=nn.Sequential(*list(self.resnet.children())[:-3])#定义特征提取器4--左脸
        #self.features5=nn.Sequential(*list(self.resnet4.children())[:-3])#定义特征提取器5--鼻子
        self.features6=nn.Sequential(*list(self.resnet.children())[:-3])#定义特征提取器6--右脸
        self.features7=nn.Sequential(*list(self.resnet.children())[:-3])#定义特征提取器6--嘴部
        self.features8=nn.Sequential(*list(self.resnet.children())[-3:-2])#定义特征提取器8
        

        # 定义五参考线作为可学习参数
        self.w1 = nn.Parameter(torch.tensor(0.5))
        self.w2 = nn.Parameter(torch.tensor(0.75))

        self.h1 = nn.Parameter(torch.tensor(0.5))
        self.h2 = nn.Parameter(torch.tensor(0.65))
        
        #回归层
        self.fc_loc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 3 * 2)#仿射变换
        )

        #定义全连接层
        self.fc=nn.Linear(512,num_class)
        self.fc2=nn.Linear(256,num_class)
        
        #定义一个全局平均池化层
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        
        #定义ReLU激活函数
        self.relu=nn.ReLU()
        
    def forward(self,x):
    
        w=x.size(3)#获取张量的宽度
        h=x.size(2)#获取张量的高度
        w_1=int(w * self.w1)#获取图片脸鼻分界线（左）
        w_2=int(w * self.w2)#获取图片脸鼻分界线（右）
        h_1=int(h * self.h1)#获取图片眼鼻分界线
        h_2=int(h * self.h2)#获取图片嘴鼻分界线

        x2=x[:,:,0:h_1,0:w_1]#截取左眼图像
        x3=x[:,:,0:h_1,w_1:w]#截取右眼图像
        x4=x[:,:,h_1:h_2,0:w_1]#截取左脸图像
        #x5=x[:,:,h_1:h_2,w_1:w_2]#截取鼻子图像
        x6=x[:,:,h_1:h_2,w_1:w]#截取右脸图像
        x7=x[:,:,h_2:h,:]#截取嘴部图像
        
        x1=self.features1(x) # 全局特征        
        x2=self.features2(x2) #左眼特征
        x3=self.features3(x3) #右眼特征
        x4=self.features4(x4) #右脸特征
        #x5=self.features5(x5) #鼻子特征
        x6=self.features6(x6) #右脸特征
        x7=self.features7(x7) #嘴部特征
        
        x8=torch.cat([x2,x3],dim=3) #拼接左右眼部特征
        x8=F.interpolate(x8, size=(x8.size(2), x1.size(3)), mode='bilinear', align_corners=True)
        x9=torch.cat([x4,x6],dim=3) #拼接左右脸部特征
        x9=F.interpolate(x9, size=(x9.size(2), x1.size(3)), mode='bilinear', align_corners=True)
        x10=torch.cat([x8,x9,x7],dim=2) #拼接眼部+脸部+嘴部特征
        x10=F.interpolate(x10, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=True)
        
        xs=self.features8(x10)#特征提取
        #回归层
        xs = xs.view(x1.size(0), -1)#降维(b,c*h*w)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        #利用theta参数计算变换后图片的位置
        grid = F.affine_grid(theta, x1.size(), align_corners=True)
        #根据输入图片计算变换后图片位置填充的像素值
        x11 = F.grid_sample(x1, grid, align_corners=True)
        
        x1=x1+x11
        
        x1=self.features8(x1)#特征提取
        x1=self.avgpool(x1)#平均池化
        x1=x1.view(x.size(0),-1)#降维(b,c*h*w)
        x1=self.fc(x1)#全连接
        
        
        x10=self.avgpool(x11)#平均池化
        x10=x10.view(x.size(0),-1)
        x10=self.fc2(x10)
        
        return x1,x10

#resnet50
class ResNet_50(nn.Module):      
    def __init__(self,num_class=7,device='cpu'):
        super(ResNet_50, self).__init__()
        
        self.resnet = models.resnet50()                                                # 定义resnet18模型
        checkpoint = torch.load('models/resnet50_pretrained.pth',map_location = device)   # 加载指定路径的模型
        self.resnet.load_state_dict(checkpoint)            # 加载模型参数

        # 获取 ResNet-50 最后一层的输入特征维度
        self.num_features = self.resnet.fc.in_features  
        # 修改最后一层，输出维度为类别数
        self.resnet.fc = torch.nn.Linear(self.num_features, num_class)
        
    def forward(self,x):
        
        x0 = self.resnet(x)
        
        return x0
        