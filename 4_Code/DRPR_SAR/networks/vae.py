from __future__ import print_function
import abc
import os
import math

from torchvision.models import  vgg13_bn, densenet121,  alexnet, efficientnet_b0, shufflenet_v2_x2_0, mobilenet_v3_large, wide_resnet50_2, VGG13_BN_Weights

import numpy as np
import logging
import torch
import torch.utils.data
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Variable
import pdb
import sys
from .resnet_cbs import resnet50
from .nearest_embed import NearestEmbed
from .preactresnet import get_PreActResNet
from .widerresnet import WideResNet
from .vgg import get_VGG
from .mobilenet import get_Mobilenet
from .densenet import get_Densenet


sys.path.append('.')

sys.path.append('..')
from utils.normalize import *




def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


def densenet(num_class):
    model = densenet121(num_classes=num_class)
    model.features.conv0 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model

# ------------------- 模型定义 -------------------
#以下模型用于classifier_train_sar.py；在disentangle_SAR.py调用；本文件内函数的调用

def wideresnet(num_classes, pretrained_path='./checkpoints/wide_resnet50_2-95faca4d.pth'):
    # 创建模型并修改第一层卷积
    model = wide_resnet50_2(num_classes=num_classes)  # 注意：此处会自动初始化分类头为 num_classes
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    if pretrained_path is not None:
        pretrained_dict = torch.load(pretrained_path)

        # 需要移除的键：输入层和分类头的不匹配参数
        keys_to_remove = [
            'conv1.weight',  # 输入通道不同（原3通道，你改为1通道）
            'fc.weight',  # 分类头输出维度不同（原1000类，你改为num_classes）
            'fc.bias'  # 同上
        ]
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k not in keys_to_remove
        }

        # 加载可匹配的参数（非严格模式）
        model.load_state_dict(pretrained_dict, strict=False)

    return model


def vgg(num_classes=10, pretrained=True):
    # 创建基础模型
    model = vgg13_bn(weights=VGG13_BN_Weights.IMAGENET1K_V1 if pretrained else None)

    # 修改输入通道（3通道 -> 单通道）
    with torch.no_grad():
        # 获取原始第一层参数
        original_conv = model.features[0]
        new_conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        # 智能权重转换（RGB三通道取平均）
        new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        new_conv.bias.data = original_conv.bias.data.clone()

        model.features[0] = new_conv

    # 修改分类头
    model.classifier[6] = nn.Linear(4096, num_classes)

    # 初始化新分类头
    nn.init.normal_(model.classifier[6].weight, 0, 0.01)
    nn.init.constant_(model.classifier[6].bias, 0)

    return model


class AbstractAutoEncoder(nn.Module):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode(self, x):
        return

    @abc.abstractmethod
    def decode(self, z):
        return

    @abc.abstractmethod
    def forward(self, x):
        """model return (reconstructed_x, *)"""
        return

    @abc.abstractmethod
    def sample(self, size):
        """sample new images from model"""
        return

    @abc.abstractmethod
    def loss_function(self, **kwargs):
        """accepts (original images, *) where * is the same as returned from forward()"""
        return

    @abc.abstractmethod
    def latest_losses(self):
        """returns the latest losses in a dictionary. Useful for logging."""
        return


class VQVAE_SAR(nn.Module):
    def __init__(self, d, k=10, num_classes=6, num_channels=1, **kwargs):
        super(VQVAE_SAR, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(inplace=True),
            ResBlock(d, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            nn.BatchNorm2d(d),
        )
        self.decoder = nn.Sequential(
            ResBlock(d, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d, num_channels, kernel_size=4, stride=2, padding=1),
        )
        self.d = d
        self.emb = NearestEmbed(k, d)

        for l in self.modules():
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                l.weight.detach().normal_(0, 0.02)
                torch.fmod(l.weight, 0.04)
                nn.init.constant_(l.bias, 0)

        self.encoder[-1].weight.detach().fill_(1 / 40)

        self.emb.weight.detach().normal_(0, 0.02)
        torch.fmod(self.emb.weight, 0.04)

        self.classifier = resnet50(pretrained=True, num_classes=num_classes)

        self.L_bn = nn.BatchNorm2d(num_channels)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return torch.tanh(self.decoder(x))

    def forward(self, x):

        # z_e = self.encode(x)
        z_e = self.encode(x)

        z_q, _ = self.emb(z_e, weight_sg=True)
        emb, _ = self.emb(z_e.detach())

        l = self.decode(z_q)
        gx = self.L_bn(l)
        out = self.classifier(x-gx)

        return  out, gx, z_e, emb

# ------------------- 模型定义 -------------------
#以下模型用于disentangle_SAR.py

class CVAE_SAR_Res(AbstractAutoEncoder):#vae+resnet50
    def __init__(self, d, z, num_classes=6, num_channels=1, **kwargs):
        super(CVAE_SAR_Res, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )

        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),

            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, 1, kernel_size=4, stride=2, padding=1, bias=False),
        )

        self.xi_bn = nn.BatchNorm2d(1)#3

        self.f = 32 #8#8
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f ** 2, self.z)
        self.fc12 = nn.Linear(d * self.f ** 2, self.z)
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)

        self.classifier = resnet50(pretrained=True, num_classes=num_classes)

        # self.with_classifier = with_classifier
        # if self.with_classifier:
        #     self.classifier = Wide_ResNet(28, 10, 0.3, num_classes=10)

    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(-1, self.d * self.f ** 2)
        return h, self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu  # mu是均值，数方差 logvar，std是标准差

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        _, mu, logvar = self.encode(x)
        hi = self.reparameterize(mu, logvar)  # + noise* torch.randn(mu.size()).cuda()  #潜在变量hi
        hi_projected = self.fc21(hi)
        xi = self.decode(hi_projected)  # 通过解码器恢复图像 xi，即获得重建图像
        xi = self.xi_bn(xi)

        # if self.with_classifier:
        
        # out = self.classifier(x - xi)  # 得到R（x）用于分类任务
        out = self.classifier(xi) # 得到G（x）用于分类任务
        return out, hi, xi, mu, logvar
        # else:
        #     return xi



class CVAE_SAR_Wid(AbstractAutoEncoder):#vae+resnet50
    def __init__(self, d, z, num_classes=6, num_channels=1, **kwargs):
        super(CVAE_SAR_Wid, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )

        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),

            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, 1, kernel_size=4, stride=2, padding=1, bias=False),
        )

        self.xi_bn = nn.BatchNorm2d(1)#3

        self.f = 32 #8
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f ** 2, self.z)
        self.fc12 = nn.Linear(d * self.f ** 2, self.z)
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)

        # self.classifier = resnet50(pretrained=True, num_classes=num_classes)
        self.classifier = wideresnet(num_classes)

        # self.with_classifier = with_classifier
        # if self.with_classifier:
        #     self.classifier = Wide_ResNet(28, 10, 0.3, num_classes=10)

    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(-1, self.d * self.f ** 2)
        return h, self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu  # mu是均值，数方差 logvar，std是标准差

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        _, mu, logvar = self.encode(x)
        hi = self.reparameterize(mu, logvar)  # + noise* torch.randn(mu.size()).cuda()  #潜在变量hi
        hi_projected = self.fc21(hi)
        xi = self.decode(hi_projected)  # 通过解码器恢复图像 xi，即获得重建图像
        xi = self.xi_bn(xi)

        # if self.with_classifier:
        out = self.classifier(x - xi)  # 得到R（x）用于分类任务
        return out, hi, xi, mu, logvar
        # else:
        #     return xi




class CVAE_SAR_PreRes(AbstractAutoEncoder):#vae+resnet50
    def __init__(self, d, z, num_classes=10, num_channels=1, **kwargs):
        super(CVAE_SAR_PreRes, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )

        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),

            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, 1, kernel_size=4, stride=2, padding=1, bias=False),
        )

        self.xi_bn = nn.BatchNorm2d(1)#3

        self.f = 32 #8
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f ** 2, self.z)
        self.fc12 = nn.Linear(d * self.f ** 2, self.z)
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)

        # self.classifier = resnet50(pretrained=True, num_classes=num_classes)
        self.classifier = get_PreActResNet(50,num_classes)

        # self.with_classifier = with_classifier
        # if self.with_classifier:
        #     self.classifier = Wide_ResNet(28, 10, 0.3, num_classes=10)

    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(-1, self.d * self.f ** 2)
        return h, self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu  # mu是均值，数方差 logvar，std是标准差

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        _, mu, logvar = self.encode(x)
        hi = self.reparameterize(mu, logvar)  # + noise* torch.randn(mu.size()).cuda()  #潜在变量hi
        hi_projected = self.fc21(hi)
        xi = self.decode(hi_projected)  # 通过解码器恢复图像 xi，即获得重建图像
        xi = self.xi_bn(xi)

        # if self.with_classifier:
        out = self.classifier(x - xi)  # 得到R（x）用于分类任务
        return out, hi, xi, mu, logvar
        # else:
        #     return xi



class CVAE_SAR_vgg(AbstractAutoEncoder):#vae+vgg
    def __init__(self, d, z, num_classes=10, num_channels=1, **kwargs):
        super(CVAE_SAR_vgg, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )

        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),

            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, 1, kernel_size=4, stride=2, padding=1, bias=False),
        )

        self.xi_bn = nn.BatchNorm2d(1)#3

        self.f = 32 #8
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f ** 2, self.z)
        self.fc12 = nn.Linear(d * self.f ** 2, self.z)
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)

        # self.classifier = resnet50(pretrained=True, num_classes=num_classes)
        self.classifier = get_VGG(13,num_classes)

        # self.with_classifier = with_classifier
        # if self.with_classifier:
        #     self.classifier = Wide_ResNet(28, 10, 0.3, num_classes=10)

    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(-1, self.d * self.f ** 2)
        return h, self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu  # mu是均值，数方差 logvar，std是标准差

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        _, mu, logvar = self.encode(x)
        hi = self.reparameterize(mu, logvar)  # + noise* torch.randn(mu.size()).cuda()  #潜在变量hi
        hi_projected = self.fc21(hi)
        xi = self.decode(hi_projected)  # 通过解码器恢复图像 xi，即获得重建图像
        xi = self.xi_bn(xi)

        # if self.with_classifier:
        # out = self.classifier(x - xi)  # 得到R（x）用于分类任务
        out = self.classifier(xi)  # 得到G（x）用于分类任务
        return out, hi, xi, mu, logvar
        # else:
        #     return xi


class CVAE_SAR_widerresnet(AbstractAutoEncoder):#vae+resnet50
    def __init__(self, d, z, num_classes=10, num_channels=1, **kwargs):
        super(CVAE_SAR_widerresnet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )

        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),

            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, 1, kernel_size=4, stride=2, padding=1, bias=False),
        )

        self.xi_bn = nn.BatchNorm2d(1)#3

        self.f = 32 #8
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f ** 2, self.z)
        self.fc12 = nn.Linear(d * self.f ** 2, self.z)
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)

        # self.classifier = resnet50(pretrained=True, num_classes=num_classes)
        self.classifier = WideResNet(34,num_classes)

        # self.with_classifier = with_classifier
        # if self.with_classifier:
        #     self.classifier = Wide_ResNet(28, 10, 0.3, num_classes=10)

    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(-1, self.d * self.f ** 2)
        return h, self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu  # mu是均值，数方差 logvar，std是标准差

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        _, mu, logvar = self.encode(x)
        hi = self.reparameterize(mu, logvar)  # + noise* torch.randn(mu.size()).cuda()  #潜在变量hi
        hi_projected = self.fc21(hi)
        xi = self.decode(hi_projected)  # 通过解码器恢复图像 xi，即获得重建图像
        xi = self.xi_bn(xi)

        # if self.with_classifier:
        out = self.classifier(x - xi)  # 得到R（x）用于分类任务
        return out, hi, xi, mu, logvar
        # else:
        #     return xi


class CVAE_SAR_PreActResNet(AbstractAutoEncoder):#vae+resnet50
    def __init__(self, d, z, num_classes=10, num_channels=1, **kwargs):
        super(CVAE_SAR_PreActResNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )

        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),

            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, 1, kernel_size=4, stride=2, padding=1, bias=False),
        )

        self.xi_bn = nn.BatchNorm2d(1)#3

        self.f = 32 #8
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f ** 2, self.z)
        self.fc12 = nn.Linear(d * self.f ** 2, self.z)
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)

        self.classifier = get_PreActResNet(50,num_classes)


    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(-1, self.d * self.f ** 2)
        return h, self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu  # mu是均值，数方差 logvar，std是标准差

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        _, mu, logvar = self.encode(x)
        hi = self.reparameterize(mu, logvar)  # + noise* torch.randn(mu.size()).cuda()  #潜在变量hi
        hi_projected = self.fc21(hi)
        xi = self.decode(hi_projected)  # 通过解码器恢复图像 xi，即获得重建图像
        xi = self.xi_bn(xi)

        # if self.with_classifier:
        out = self.classifier(x - xi)  # 得到R（x）用于分类任务
        return out, hi, xi, mu, logvar
        # else:
        #     return xi


class CVAE_SAR_mobile(AbstractAutoEncoder):  # vae+resnet50
    def __init__(self, d, z, num_classes=10, num_channels=1, **kwargs):
        super(CVAE_SAR_mobile, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )

        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),

            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, 1, kernel_size=4, stride=2, padding=1, bias=False),
        )

        self.xi_bn = nn.BatchNorm2d(1)  # 3

        self.f = 32  # 8
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f ** 2, self.z)
        self.fc12 = nn.Linear(d * self.f ** 2, self.z)
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)

        self.classifier = get_Mobilenet(num_classes)

    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(-1, self.d * self.f ** 2)
        return h, self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu  # mu是均值，数方差 logvar，std是标准差

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        _, mu, logvar = self.encode(x)
        hi = self.reparameterize(mu, logvar)  # + noise* torch.randn(mu.size()).cuda()  #潜在变量hi
        hi_projected = self.fc21(hi)
        xi = self.decode(hi_projected)  # 通过解码器恢复图像 xi，即获得重建图像
        xi = self.xi_bn(xi)

        # if self.with_classifier:
        out = self.classifier(x - xi)  # 得到R（x）用于分类任务
        return out, hi, xi, mu, logvar
        # else:
        #     return xi


class CVAE_SAR_densenet(AbstractAutoEncoder):  # vae+resnet50
    def __init__(self, d, z, num_classes=10, num_channels=1, **kwargs):
        super(CVAE_SAR_densenet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )

        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),

            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, 1, kernel_size=4, stride=2, padding=1, bias=False),
        )

        self.xi_bn = nn.BatchNorm2d(1)  # 3

        self.f = 32  # 8
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f ** 2, self.z)
        self.fc12 = nn.Linear(d * self.f ** 2, self.z)
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)

        self.classifier = get_Densenet(121,num_classes)

    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(-1, self.d * self.f ** 2)
        return h, self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu  # mu是均值，数方差 logvar，std是标准差

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        _, mu, logvar = self.encode(x)
        hi = self.reparameterize(mu, logvar)  # + noise* torch.randn(mu.size()).cuda()  #潜在变量hi
        hi_projected = self.fc21(hi)
        xi = self.decode(hi_projected)  # 通过解码器恢复图像 xi，即获得重建图像
        xi = self.xi_bn(xi)

        # if self.with_classifier:
        out = self.classifier(x - xi)  # 得到R（x）用于分类任务
        return out, hi, xi, mu, logvar
        # else:
        #     return xi

