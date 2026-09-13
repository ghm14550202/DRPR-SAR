import torch
from torch import nn
from torch.nn import functional as F
# from SAR_loader.normalize import *
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2470, 0.2435, 0.2616]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_cifar_params(resol):
    mean_list = []
    std_list = []
    for i in range(3):
        mean_list.append(torch.full((resol, resol), CIFAR_MEAN[i], device='cuda'))
        std_list.append(torch.full((resol, resol), CIFAR_STD[i], device='cuda'))
    return torch.unsqueeze(torch.stack(mean_list), 0), torch.unsqueeze(torch.stack(std_list), 0)


def get_SAR_params(resol, dataset):
    # 根据数据集选择不同的均值和标准差
    if dataset == 'mstar':
        mean_values = (0.13620707)
        std_values = (0.11149936)
    elif dataset == 'FUSAR':
        mean_values = (0.10994489)
        std_values = (0.10608266)
    else:
        raise ValueError(f"Dataset {dataset} not recognized. Choose from ['mstar', 'acd', 'opensar']")

        # 创建对应分辨率的均值和标准差张量
    mean = torch.full((1, 1, resol, resol), mean_values, device='cuda')  # 单通道均值
    std = torch.full((1, 1, resol, resol), mean_values, device='cuda')  # 单通道标准差

    # 返回均值和标准差的张量
    return mean, std


def get_imagenet_params(resol):
    mean_list = []
    std_list = []
    for i in range(3):
        mean_list.append(torch.full((resol, resol), IMAGENET_MEAN[i], device='cuda'))
        std_list.append(torch.full((resol, resol), IMAGENET_STD[i], device='cuda'))
    return torch.unsqueeze(torch.stack(mean_list), 0), torch.unsqueeze(torch.stack(std_list), 0)

class SARNORMALIZE(nn.Module):
    def __init__(self, resol, datasets):
        super().__init__()
        self.mean, self.std = get_SAR_params(resol, dataset=datasets)

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to ([0, 1] - IMAGENET_MEAN) / IMAGENET_STD
        '''
        x = x.sub(self.mean)
        x = x.div(self.std)
        return x

class SARINNORMALIZE(nn.Module):
    def __init__(self, resol, datasets):
        super().__init__()
        self.mean, self.std = get_SAR_params(resol, dataset=datasets)

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to ([0, 1] - IMAGENET_MEAN) / IMAGENET_STD
        '''
        x = x.mul(self.std)
        x = x.add(*self.mean)
        return x


class CIFARNORMALIZE(nn.Module):
    def __init__(self, resol):
        super().__init__()
        self.mean, self.std = get_cifar_params(resol)

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to ([0, 1] - IMAGENET_MEAN) / IMAGENET_STD
        '''
        x = x.sub(self.mean)
        x = x.div(self.std)
        return x

class CIFARINNORMALIZE(nn.Module):
    def __init__(self, resol):
        super().__init__()
        self.mean, self.std = get_cifar_params(resol)

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to ([0, 1] - IMAGENET_MEAN) / IMAGENET_STD
        '''
        x = x.mul(self.std)
        x = x.add(*self.mean)
        return x

class IMAGENETNORMALIZE(nn.Module):
    def __init__(self, resol):
        super().__init__()
        self.mean, self.std = get_imagenet_params(resol)

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to ([0, 1] - IMAGENET_MEAN) / IMAGENET_STD
        '''
        x = x.sub(self.mean)
        x = x.div(self.std)
        return x

class IMAGENETINNORMALIZE(nn.Module):
    def __init__(self, resol):
        super().__init__()
        self.mean, self.std = get_imagenet_params(resol)

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to ([0, 1] - IMAGENET_MEAN) / IMAGENET_STD
        '''
        x = x.mul(self.std)
        x = x.add(*self.mean)
        return x
