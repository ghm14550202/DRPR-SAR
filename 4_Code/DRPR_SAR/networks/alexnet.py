import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, dataset='mstar'):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 第一层（修改为小卷积核适应小尺寸输入）
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),  # 使用BN代替原始LRN

            # 第二层
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(192),

            # 第三层
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 第四层
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 第五层
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 自适应池化解决尺寸匹配问题
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_AlexNet(num_classes, dataset='mstar'):
    return AlexNet(num_classes, dataset)


if __name__ == "__main__":
    # 测试样例（输入尺寸3x32x32）
    net = AlexNet(num_classes=10)
    y = net(torch.randn(2, 3, 32, 32))  # batch_size=2
    print(y.shape)  # 预期输出: torch.Size([2, 10])
