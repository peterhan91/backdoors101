import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter

from utils.utils import th, thp

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.lamda = 1

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

        # hook for the gradients

    def activations_hook(self, grad):
        self.gradient = grad

    def get_gradient(self):
        return self.gradient

    def get_activations(self, x):
        return self.features(x)

    def features(self, x):
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)

        return out5

    def forward(self, x):
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)

        # h = out5.register_hook(self.activations_hook)

        out = F.avg_pool2d(out5, 4)
        out_latent = out.view(out.size(0), -1)
        out = self.linear(out_latent)
        return out, out5
        # return out, out5
        # return out, out_latent # torch.cat([out1.view(-1), out2.view(-1), out3.view(-1), out4.view(-1), out5.view(-1),
                          #       out_latent.view(-1)])


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)

def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes)

def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes)

def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes)


# def test():
#     net = ResNet18()
#     y = net(torch.randn(1,3,self.size ,self.size ))
#     print(y.size())

# test()





class Mixed(nn.Module):

    def __init__(self, model, size):
        super().__init__()
        self.size = size
        self.pattern = torch.zeros([self.size , self.size ], requires_grad=True)\
                             + torch.normal(0, 0.5, [self.size , self.size ])
        self.mask = torch.zeros([self.size , self.size ], requires_grad=True)
                   # + torch.normal(0, 2, [self.size , self.size ])
        # self.mask[:, :, :22] = -1
        # self.mask[:, :, 25:] = -1
        # self.mask[:, 7:, :] = -1
        # self.pattern[:, :, :22] = -1
        # self.pattern[:, :, 25:] = -1
        # self.pattern[:, 7:, :] = -1
        self.mask = Parameter(self.mask)
        self.pattern = Parameter(self.pattern)
        self.resnet = model

    def forward(self, x):
        maskh = th(self.mask)
        patternh = thp(self.pattern)
        x = (1 - maskh) * x + maskh * patternh
        x, latent = self.resnet(x)

        return x, latent

    def grad_weights(self, mask=True, model=False):
        for i, n in self.named_parameters():
            if i == 'mask' or i == 'pattern':
                n.requires_grad_(mask)
            else:
                n.requires_grad_(model)

    def re_init(self, device):
        p = torch.zeros([self.size, self.size], requires_grad=False)

        self.pattern.data = p.to(device)
        m = torch.zeros([self.size, self.size], requires_grad=True) \
            + torch.normal(0, 0.5, [self.size, self.size])

        self.mask.data = m.to(device)

    def init_pattern(self):
        min_val = -2.2
        max_val = 2.2
        self.pattern.data[:,:,:] = min_val
        self.pattern.data[:, 2, 25] = max_val
        self.pattern.data[:, 2, 24] = min_val
        self.pattern.data[:, 2, 23] = max_val
        self.pattern.data[:, 6, 25] = max_val
        self.pattern.data[:, 6, 24] = min_val
        self.pattern.data[:, 6, 23] = max_val
        self.pattern.data[:, 5, 24] = max_val
        self.pattern.data[:, 4, 23] = min_val
        self.pattern.data[:, 3, 24] = max_val

class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means=_CIFAR10_MEAN, sds=_CIFAR10_STDDEV):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds
