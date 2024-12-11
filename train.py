import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import datetime

from network.models import *  # 导入自定义的模型
from tqdm import tqdm
import matplotlib.pyplot as plt

# 用于服务器生成图形
plt.switch_backend('agg')

# 记录时间
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")

# 设置使用的设备
device = 'cuda:3'

# 定义数据集名称和路径
dataset_name = 'rafdb'
data_path = './dataset/' + dataset_name
model_name = dataset_name + '_'
checkpoint_path = './checkpoints/' + model_name + time_str + '.pth'
best_checkpoint_path = './checkpoints/' + model_name + time_str + '_best.pth'
txt_name = './logs/' + model_name + time_str + '.txt'
curve_name = './logs/' + model_name + time_str + '.png'

# 设置训练超参数
net                  = 13
#12 8 89.21
alpha                = 12
beta                 = 8
eval                 = False
lr                   = 0.01 
momentum             = 0.9
weight_decay         = 1e-4
epochs               = 100
ls                   = 15
batch_size           = 256
workers              = 8
print_freq           = 100
pretrained           = False

# 定义训练集和验证集路径
traindir = os.path.join(data_path, 'train')
valdir = os.path.join(data_path, 'test')

def main():
    best_acc = 0
    start_epoch = 0
     
    network_name = 'Model_' + str(net)
    print('Training time: ' + now.strftime("%m-%d %H:%M"))
    print('device:    ' + device)
    print('dataset:    ' + dataset_name)
    print('network:    ' + network_name)
    print('alpha:  ' + str(alpha) + '   beta:  ' + str(beta))

    # 初始化模型
    model = Model_13(num_class=7, device=device)
    model = model.to(device)


    # 定义损失函数、优化器和学习率调度器
    criterion_cls = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=ls, gamma=0.1)
    recorder = RecorderMeter( epochs)
    
    # 数据并行加速设置
    cudnn.benchmark = True

    # 加载训练和验证数据集
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(20), transforms.RandomCrop(224, padding=32)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25)),
    ]))

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    with open(txt_name, 'a') as f:
        f.write('model: ' + str(net) + '\n' + 'time: ' + time_str + '\n')

    # 遍历每个epoch，并显示进度条
    for epoch in tqdm(range(start_epoch, epochs)):
        start_time = time.time()  # 记录当前时间，用于计算每个epoch的耗时
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']  # 获取当前的学习率
        tqdm.write('Current learning rate: ' + str(current_learning_rate))  # 在进度条中显示当前学习率
        with open(txt_name, 'a') as f:
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')  # 将当前学习率写入文本文件
    
        # 训练模型，并获取训练集上的准确率和损失
        train_acc, train_los = train(train_loader, model, criterion_cls, optimizer, epoch+1)      
        
        # 在验证集上验证模型，并获取验证集上的准确率和损失
        val_acc, val_los = validate(val_loader, model, criterion_cls)   
        
        scheduler.step()  # 更新学习率
        
        # 更新记录器（recorder）中的数据，并绘制曲线
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)    
        recorder.plot_curve(curve_name)  
    
        is_best = val_acc > best_acc  # 判断当前模型在验证集上的表现是否超过历史最佳
        best_acc = max(best_acc, val_acc)  # 更新历史最佳准确率
    
        tqdm.write('Current best accuracy: ' + str(best_acc.item()))  # 在进度条中显示当前最佳准确率
        with open(txt_name, 'a') as f:
            f.write('********************Current best accuracy: ' + str(best_acc.item()) + '\n')  # 将当前最佳准确率写入文本文件  
    
        # 保存当前模型的checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'recorder': recorder,
        }, is_best)
    
        end_time = time.time()  # 记录当前时间，用于计算每个epoch的耗时
        epoch_time = end_time - start_time  # 计算每个epoch的耗时
        tqdm.write("An Epoch Time: " + str(epoch_time))  # 在进度条中显示当前epoch的耗时
        with open(txt_name, 'a') as f:      
            f.write('An epoch time: ' + str(epoch_time) + '\n')  # 将每个epoch的耗时写入文本文件
    
    print('Training time: ' + now.strftime("%m-%d %H:%M"))
    print('device:    ' + device)
    print('dataset:    ' + dataset_name) 
    print('network:    ' + network_name)
    print('best_checkpoint_path: ' + best_checkpoint_path)
    print('alpha:  ' + str(alpha) + '   beta:  ' + str(beta))


# 训练函数
def train(train_loader, model, criterion_cls, optimizer, epoch):
    losses = AverageMeter('Loss', ':.4f')  # 用于记录平均损失的对象
    top1 = AverageMeter('Accuracy', ':6.3f')  # 用于记录平均准确率的对象
    progress = ProgressMeter(len(train_loader), [losses, top1], prefix="Epoch: [{}]".format(epoch))  # 用于显示训练进度条的对象
    model.train()  # 设置模型为训练模式
    
    for i, (images, targets) in enumerate(train_loader):
        targets = targets.to(device)  # 将目标值移动到指定设备上（如GPU）
        images = images.to(device)  # 将输入图像移动到指定设备上
        
        optimizer.zero_grad()  # 清空梯度
        out,heads = model(images)  # 前向传播，得到模型输出
        loss = criterion_cls(out, targets) * alpha + criterion_cls(heads, targets) * beta # 计算损失
        acc = accuracy(out, targets)  # 计算准确率

        losses.update(loss.item(), images.size(0))  # 更新平均损失       
        top1.update(acc.item(), images.size(0))  # 更新平均准确率      
        
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数
        
        if i % print_freq == 0:
            progress.display(i)  # 每隔一定的步数显示训练进度
     
    return top1.avg, losses.avg  # 返回平均准确率和平均损失


# 验证函数
def validate(val_loader, model, criterion_cls):
    losses = AverageMeter('Loss', ':.4f')  # 用于记录验证过程中的平均损失的对象
    top1 = AverageMeter('Accuracy', ':6.3f')  # 用于记录验证过程中的平均准确率的对象
    progress = ProgressMeter(len(val_loader), [losses, top1], prefix='Test: ')  # 用于显示验证进度条的对象
    
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁止梯度计算
        for i, (images, targets) in enumerate(val_loader):
            targets = targets.to(device)  # 将目标值移动到指定设备上
            images = images.to(device)  # 将输入图像移动到指定设备上
            out,heads = model(images)  # 前向传播，得到模型输出
            loss = criterion_cls(out, targets) * alpha + criterion_cls(heads, targets) * beta    # 计算损失
            acc = accuracy(out, targets)  # 计算准确率
            losses.update(loss.item(), images.size(0))  # 更新平均损失
            top1.update(acc, images.size(0))  # 更新平均准确率

            if i % print_freq == 0:
                progress.display(i)  # 每隔一定的步数显示验证进度
        
        tqdm.write(' **** Accuracy {top1.avg:.3f} *** '.format(top1=top1))  # 打印最终准确率
        with open(txt_name, 'a') as f:
            f.write(' * Accuracy {top1.avg:.3f}'.format(top1=top1) + '\n')  # 将准确率写入文件

    return top1.avg, losses.avg  # 返回平均准确率和平均损失作为验证结果


# 计算准确率
def accuracy(logits, labels):
    acc = (logits.argmax(dim=-1) == labels).float().mean()
    return acc * 100.0

# 保存模型
def save_checkpoint(state, is_best):
    torch.save(state, checkpoint_path)   
    if is_best: 
        shutil.copyfile(checkpoint_path, best_checkpoint_path)

# 计算平均值
class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        """
        初始化函数，用于设置名称和格式

        Args:
        - name: str，平均值记录器的名称
        - fmt: str，格式字符串，默认为浮点数格式
        """
        self.name = name
        self.fmt = fmt
        self.reset()  # 调用reset函数初始化记录器的值

    def reset(self):
        """
        重置记录器的值
        """
        self.val = 0   # 当前值
        self.avg = 0   # 平均值
        self.sum = 0   # 总和
        self.count = 0  # 计数

    def update(self, val, n=1):
        """
        更新记录器的值

        Args:
        - val: float，当前值
        - n: int，更新次数，默认为1
        """
        self.val = val  # 更新当前值
        self.sum += val * n  # 更新总和
        self.count += n  # 更新计数
        self.avg = self.sum / self.count  # 更新平均值

    def __str__(self):
        """
        格式化输出字符串
        """
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'  # 格式化字符串
        return fmtstr.format(**self.__dict__)  # 使用format函数格式化输出字符串


# 显示训练进度
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]       
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        tqdm.write(print_txt)
        with open(txt_name, 'a') as f:    
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

# 记录器
class RecorderMeter(object):
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        # 画出训练和验证的准确率/损失曲线
        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            tqdm.write('Saved figure')
        plt.close(fig)

if __name__ == '__main__':
    main()
