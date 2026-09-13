from __future__ import print_function
import abc
import os
import math

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

from utils.SAR_loader import get_model
from .mobilenet import get_Mobilenet
from .resnet import resnet50
from .nearest_embed import NearestEmbed
from .preactresnet import get_PreActResNet
# from torchvision.models import  vgg13_bn, densenet121,  alexnet, efficientnet_b0, shufflenet_v2_x2_0, mobilenet_v3_large, wide_resnet50_2
from .vgg import get_VGG
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


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, norm=False):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor

        print('| Wide-Resnet %dx%d' % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)
        # self.normalize = CIFARNORMALIZE(32)
        # self.norm = norm
        self.normalize = SARNORMALIZE(128)
        self.norm = norm

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.norm:
            x = self.normalize(x)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        print(f"Shape after pooling: {out.shape}")
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out




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

# ------------------- 模型定义 -------------------
#以下模型用于adv_train_SAR以及本文件内的函数调用；函数的设置参考vae文件

#
# def wideresnet(num_class):
#     model = wide_resnet50_2(num_classes=num_class)
#     model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#     return model

# def wideresnet(num_classes, pretrained_path='./checkpoints/wide_resnet50_2-95faca4d.pth'):
#     # 创建模型并修改第一层卷积
#     model = wide_resnet50_2(num_classes=num_classes)  # 注意：此处会自动初始化分类头为 num_classes
#     model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

#     if pretrained_path is not None:
#         pretrained_dict = torch.load(pretrained_path)

#         # 需要移除的键：输入层和分类头的不匹配参数
#         keys_to_remove = [
#             'conv1.weight',  # 输入通道不同（原3通道，你改为1通道）
#             'fc.weight',  # 分类头输出维度不同（原1000类，你改为num_classes）
#             'fc.bias'  # 同上
#         ]
#         pretrained_dict = {
#             k: v for k, v in pretrained_dict.items()
#             if k not in keys_to_remove
#         }

#         # 加载可匹配的参数（非严格模式）
#         model.load_state_dict(pretrained_dict, strict=False)

#     return model


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
        out = self.classifier(x - gx)

        return out, gx, z_e, emb


# ------------------- 模型定义 -------------------
#以下模型用于adv_train_SAR以及CD调用；以及本文件内CD_VAE_SAR_xxx系列函数的调用

class CVAE_SAR_Res(AbstractAutoEncoder):
    def __init__(self, d, z, with_classifier=True, num_classes=10, num_channels=1, **kwargs):
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

        self.xi_bn = nn.BatchNorm2d(1)  # 3

        self.f = 32  # 8
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f ** 2, self.z)
        self.fc12 = nn.Linear(d * self.f ** 2, self.z)
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)

        # self.classifier = resnet50(pretrained=True, num_classes=num_classes)

        self.with_classifier = with_classifier
        if self.with_classifier:
            # self.classifier = Wide_ResNet(28, 10, 0.3, num_classes=10)
            self.classifier = resnet50(pretrained=False, num_classes=num_classes)

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
        hi = self.reparameterize(mu, logvar)
        hi_projected = self.fc21(hi)
        xi = self.decode(hi_projected)
        xi = self.xi_bn(xi)
        if self.with_classifier:
            out = self.classifier(torch.cat((xi, x-xi), dim=0))
            out_g = out[0:x.size(0)]
            out_r = out[x.size(0):]
            return out_g, out_r, hi, xi, mu, logvar
        else:
            return xi, mu, logvar



# class CVAE_SAR_Wid(AbstractAutoEncoder):
#     def __init__(self, d, z, with_classifier=False, num_classes=10, num_channels=1, **kwargs):
#         super(CVAE_SAR_Wid, self).__init__()

#         self.encoder = nn.Sequential(
#             nn.Conv2d(num_channels, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(d // 2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(d),
#             nn.ReLU(inplace=True),
#             ResBlock(d, d, bn=True),
#             nn.BatchNorm2d(d),
#             ResBlock(d, d, bn=True),
#         )

#         self.decoder = nn.Sequential(
#             ResBlock(d, d, bn=True),
#             nn.BatchNorm2d(d),
#             ResBlock(d, d, bn=True),
#             nn.BatchNorm2d(d),

#             nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(d // 2),
#             nn.LeakyReLU(inplace=True),
#             nn.ConvTranspose2d(d // 2, 1, kernel_size=4, stride=2, padding=1, bias=False),
#         )

#         self.xi_bn = nn.BatchNorm2d(1)  # 3

#         self.f = 32  # 8
#         self.d = d
#         self.z = z
#         self.fc11 = nn.Linear(d * self.f ** 2, self.z)
#         self.fc12 = nn.Linear(d * self.f ** 2, self.z)
#         self.fc21 = nn.Linear(self.z, d * self.f ** 2)

#         # self.classifier = resnet50(pretrained=True, num_classes=num_classes)

#         self.with_classifier = with_classifier
#         if self.with_classifier:
#             # self.classifier = Wide_ResNet(28, 10, 0.3, num_classes=10)
#             # self.classifier = wideresnet(num_classes)

#     def encode(self, x):
#         h = self.encoder(x)
#         h1 = h.view(-1, self.d * self.f ** 2)
#         return h, self.fc11(h1), self.fc12(h1)

#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std = logvar.mul(0.5).exp_()
#             eps = std.new(std.size()).normal_()
#             return eps.mul(std).add_(mu)
#         else:
#             return mu  # mu是均值，数方差 logvar，std是标准差

#     def decode(self, z):
#         z = z.view(-1, self.d, self.f, self.f)
#         h3 = self.decoder(z)
#         return torch.tanh(h3)


#     def forward(self, x):
#         _, mu, logvar = self.encode(x)
#         hi = self.reparameterize(mu, logvar)
#         hi_projected = self.fc21(hi)
#         xi = self.decode(hi_projected)
#         xi = self.xi_bn(xi)
#         if self.with_classifier:
#             out = self.classifier(torch.cat((xi, x-xi), dim=0))
#             out_g = out[0:x.size(0)]
#             out_r = out[x.size(0):]
#             return out_g, out_r, hi, xi, mu, logvar
#         else:
#             return xi, mu, logvar


class CVAE_SAR_vgg(AbstractAutoEncoder):
    def __init__(self, d, z, with_classifier=True, num_classes=10, num_channels=1, **kwargs):
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

        self.xi_bn = nn.BatchNorm2d(1)  # 3

        self.f = 32  # 8
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f ** 2, self.z)
        self.fc12 = nn.Linear(d * self.f ** 2, self.z)
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)

        # self.classifier = resnet50(pretrained=True, num_classes=num_classes)

        self.with_classifier = with_classifier
        if self.with_classifier:
            # self.classifier = Wide_ResNet(28, 10, 0.3, num_classes=10)
            # self.classifier = get_PreActResNet(50,num_classes)
            self.classifier = get_VGG(13, num_classes)

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
        hi = self.reparameterize(mu, logvar)
        hi_projected = self.fc21(hi)
        xi = self.decode(hi_projected)
        xi = self.xi_bn(xi)
        if self.with_classifier:
            out = self.classifier(torch.cat((xi, x-xi), dim=0))
            out_g = out[0:x.size(0)]
            out_r = out[x.size(0):]
            return out_g, out_r, hi, xi, mu, logvar
        else:
            return xi, mu, logvar



class CVAE_SAR_preres(AbstractAutoEncoder):
    def __init__(self, d, z, with_classifier=True, num_classes=10, num_channels=1, **kwargs):
        super(CVAE_SAR_preres, self).__init__()

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

        # self.classifier = resnet50(pretrained=True, num_classes=num_classes)

        self.with_classifier = with_classifier
        if self.with_classifier:
            # self.classifier = Wide_ResNet(28, 10, 0.3, num_classes=10)
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
        hi = self.reparameterize(mu, logvar)
        hi_projected = self.fc21(hi)
        xi = self.decode(hi_projected)
        xi = self.xi_bn(xi)
        if self.with_classifier:
            out = self.classifier(torch.cat((xi, x-xi), dim=0))
            out_g = out[0:x.size(0)]
            out_r = out[x.size(0):]
            return out_g, out_r, hi, xi, mu, logvar
        else:
            return xi, mu, logvar

class CVAE_SAR_mobile(AbstractAutoEncoder):
    def __init__(self, d, z, with_classifier=True, num_classes=10, num_channels=1, **kwargs):
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

        # self.classifier = resnet50(pretrained=True, num_classes=num_classes)

        self.with_classifier = with_classifier
        if self.with_classifier:
            self.classifier = get_Mobilenet( num_classes)

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
        hi = self.reparameterize(mu, logvar)
        hi_projected = self.fc21(hi)
        xi = self.decode(hi_projected)
        xi = self.xi_bn(xi)
        if self.with_classifier:
            out = self.classifier(torch.cat((xi, x-xi), dim=0))
            out_g = out[0:x.size(0)]
            out_r = out[x.size(0):]
            return out_g, out_r, hi, xi, mu, logvar
        else:
            return xi, mu, logvar


class CVAE_SAR_densenet(AbstractAutoEncoder):
    def __init__(self, d, z, with_classifier=True, num_classes=10, num_channels=1, **kwargs):
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

        # self.classifier = resnet50(pretrained=True, num_classes=num_classes)

        self.with_classifier = with_classifier
        if self.with_classifier:
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
        hi = self.reparameterize(mu, logvar)
        hi_projected = self.fc21(hi)
        xi = self.decode(hi_projected)
        xi = self.xi_bn(xi)
        if self.with_classifier:
            out = self.classifier(torch.cat((xi, x-xi), dim=0))
            out_g = out[0:x.size(0)]
            out_r = out[x.size(0):]
            return out_g, out_r, hi, xi, mu, logvar
        else:
            return xi, mu, logvar



class CD_VAE_SAR_Res(nn.Module):
    def __init__(self, vae_path, model_path, num_classes, dataset):
        super(CD_VAE_SAR_Res, self).__init__()
        self.vae = CVAE_SAR_Res(d=32, z=2048, with_classifier=False)
        self.model = resnet50(pretrained=True, num_classes =num_classes)
        self.vae.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(vae_path).items()})
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})
        # self.normalize = SARNORMALIZE(128)
        self.normalize = SARNORMALIZE(128, dataset)


    def forward(self, x):
        gx, _, _ = self.vae(self.normalize(x))
        out = self.model(gx)
        return out


# class CD_VAE_SAR_Wid(nn.Module):
#     def __init__(self, vae_path, model_path, num_classes, dataset):
#         super(CD_VAE_SAR_Wid, self).__init__()
#         self.vae = CVAE_SAR_Wid(d=32, z=2048, with_classifier=False)
#         # self.model = wideresnet(num_classes)
#         self.vae.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(vae_path).items()})
#         self.model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})
#         self.normalize = SARNORMALIZE(128, dataset)


#     def forward(self, x):
#         gx, _, _ = self.vae(self.normalize(x))
#         out = self.model(gx)
#         return out






class CD_VAE_SAR_Res(nn.Module):
    def __init__(self, vae_path, model_path, num_classes, dataset):
        super(CD_VAE_SAR_Res, self).__init__()
        self.vae = CVAE_SAR_Res(d=32, z=2048, with_classifier=False)
        self.model = resnet50(pretrained=True, num_classes =num_classes)
        self.vae.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(vae_path).items()})
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})
        # self.normalize = SARNORMALIZE(128)
        self.normalize = SARNORMALIZE(128, dataset)


    def forward(self, x):
        gx, _, _ = self.vae(self.normalize(x))
        out = self.model(gx)
        return out


# class CD_VAE_SAR_Wid(nn.Module):
#     def __init__(self, vae_path, model_path, num_classes, dataset):
#         super(CD_VAE_SAR_Wid, self).__init__()
#         self.vae = CVAE_SAR_Wid(d=32, z=2048, with_classifier=False)
#         # self.model = wideresnet(num_classes)
#         self.vae.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(vae_path).items()})
#         self.model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})
#         self.normalize = SARNORMALIZE(128, dataset)


#     def forward(self, x):
#         gx, _, _ = self.vae(self.normalize(x))
#         out = self.model(gx)
#         return out

class CD_VAE_SAR_vgg(nn.Module):
    def __init__(self, vae_path, model_path, num_classes, dataset):
        super(CD_VAE_SAR_vgg, self).__init__()
        self.vae = CVAE_SAR_vgg(d=32, z=2048, with_classifier=False)
        self.model = get_VGG(13,num_classes)
        self.vae.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(vae_path).items()})
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})
        self.normalize = SARNORMALIZE(128, dataset)


    def forward(self, x):
        gx, _, _ = self.vae(self.normalize(x))
        out = self.model(gx)
        return out


class CD_VAE_SAR_preres(nn.Module):
    def __init__(self, vae_path, model_path, num_classes, dataset):
        super(CD_VAE_SAR_preres, self).__init__()
        self.vae = CVAE_SAR_preres(d=32, z=2048, with_classifier=False)
        self.model = get_PreActResNet(50,num_classes)
        self.vae.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(vae_path).items()})
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})
        self.normalize = SARNORMALIZE(128, dataset)


    def forward(self, x):
        gx, _, _ = self.vae(self.normalize(x))
        out = self.model(gx)
        return out


class CD_VAE_SAR_mobile(nn.Module):
    def __init__(self, vae_path, model_path, num_classes, dataset):
        super(CD_VAE_SAR_mobile, self).__init__()
        self.vae = CVAE_SAR_mobile(d=32, z=2048, with_classifier=False)
        self.model = get_Mobilenet(num_classes)
        self.vae.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(vae_path).items()})
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})
        self.normalize = SARNORMALIZE(128, dataset)


    def forward(self, x):
        gx, _, _ = self.vae(self.normalize(x))
        out = self.model(gx)
        return out


class CD_VAE_SAR_densenet(nn.Module):
    def __init__(self, vae_path, model_path, num_classes, dataset):
        super(CD_VAE_SAR_densenet, self).__init__()
        self.vae = CVAE_SAR_densenet(d=32, z=2048, with_classifier=False)
        self.model = get_Densenet(121, num_classes)
        self.vae.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(vae_path).items()})
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})
        self.normalize = SARNORMALIZE(128, dataset)


    def forward(self, x):
        gx, _, _ = self.vae(self.normalize(x))
        out = self.model(gx)
        return out