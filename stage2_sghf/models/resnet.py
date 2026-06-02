import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from vit_model import make_fixed_roi_mask



__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def LSE_Pooling(inputs):
    B, C, H, W = inputs.size()
    lse_r = 6.
    inputs = inputs.view(B, C, -1)
    inputs = lse_r * inputs
    outputs = (torch.logsumexp(inputs, dim=2) - math.log(H * W)) / lse_r
    return outputs


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias), self.weight

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = conv3x3(inplanes, planes, stride, 1, dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class ResNet(nn.Module):

    def __init__(self, block, layers, stride_list=[1, 2, 2, 2], use_maxpooling=True, dilations=None,
                 norm_layer=None, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64,
                 c_roi_mode='soft', c_roi_floor=0.3, c_roi_tau=1.0, c_roi_thr=0.5):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.use_maxpooling = use_maxpooling

        self.c_roi_mode  = c_roi_mode
        self.c_roi_floor = c_roi_floor
        self.c_roi_tau   = c_roi_tau
        self.c_roi_thr   = c_roi_thr

        self.inplanes = 64
        if dilations is None:
            dilations = [1, 1, 1, 1]
        # print(dilations)
        if len(dilations) != 4:
            raise ValueError("dilations should be None "
                             "or a 4-element tuple, got {}".format(dilations))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        if self.use_maxpooling:
            self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=stride_list[0],
                                       dilate=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=stride_list[1],
                                       dilate=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride_list[2],
                                       dilate=dilations[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=stride_list[3],
                                       dilate=dilations[3])


        self.cls_head_global2 = nn.Sequential(
            conv3x3(512, 512, groups = 8),   # 1024
            norm_layer(512),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

        self.cls_head_global3 = nn.Sequential(
            conv3x3(1024, 512, groups = 8),   # 2048
            norm_layer(512),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

        self.cls_head_global4 = nn.Sequential(
            conv3x3(2048, 512, groups = 8),   # 4096
            norm_layer(512),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

        self.cls_head_global_final = nn.Sequential(
            conv3x3(1536, 512, groups = 8),
            norm_layer(512),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 权重初始化
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=1):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, dilate, norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=dilate,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, cam):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        if self.use_maxpooling:
            x1 = self.maxpool1(x1)

        e1 = self.layer1(x1)
        m1 = make_fixed_roi_mask(cam, target_hw=e1.shape[-2:],
                                 mode=self.c_roi_mode, floor=self.c_roi_floor,
                                 tau=self.c_roi_tau, thr=self.c_roi_thr)
        e1 = e1 * m1
        e2 = self.layer2(e1)
        m2 = make_fixed_roi_mask(cam, target_hw=e2.shape[-2:],
                                 mode=self.c_roi_mode, floor=self.c_roi_floor,
                                 tau=self.c_roi_tau, thr=self.c_roi_thr)
        e2 = e2 * m2

        e3 = self.layer3(e2)

        e4 = self.layer4(e3)

        e2 = self.cls_head_global2(e2)

        e3 = self.cls_head_global3(e3)

        e4 = self.cls_head_global4(e4)

        features = torch.cat([e2, e3, e4], dim=1)   # [e2_cat, e3_cat, e4_cat]
        features = self.cls_head_global_final(features)


        return features

def _resnet(arch, block, layers, stride_list, use_maxpooling, dilations, norm_layer, progress, **kwargs):
    model = ResNet(block, layers, stride_list, use_maxpooling, dilations, norm_layer, **kwargs)
    return model


def resnet18(stride_list, use_maxpooling, dilations, norm_layer, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], stride_list, use_maxpooling, dilations, norm_layer,
                   progress, **kwargs)


def resnet34(stride_list, use_maxpooling, dilations, norm_layer, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], stride_list, use_maxpooling, dilations, norm_layer,
                   progress, **kwargs)


def resnet50(stride_list, use_maxpooling, dilations, norm_layer, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], stride_list, use_maxpooling, dilations, norm_layer,
                   progress, **kwargs)


def resnet101(stride_list, use_maxpooling, dilations, norm_layer, progress=True, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], stride_list, use_maxpooling, dilations, norm_layer,
                   progress, **kwargs)


def resnet152(stride_list, use_maxpooling, dilations, norm_layer, progress=True, **kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], stride_list, use_maxpooling, dilations, norm_layer,
                   progress, **kwargs)


def resnext50_32x4d(stride_list, use_maxpooling, dilations, norm_layer, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], stride_list, use_maxpooling, dilations, norm_layer,
                   progress, **kwargs)


def resnext101_32x8d(stride_list, use_maxpooling, dilations, norm_layer, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], stride_list, use_maxpooling, dilations, norm_layer,
                   progress, **kwargs)


def resnet152(stride_list, use_maxpooling, dilations, norm_layer, progress=True, **kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], stride_list, use_maxpooling, dilations, norm_layer,
                   progress, **kwargs)


def resnext50_32x4d(stride_list, use_maxpooling, dilations, norm_layer, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], stride_list, use_maxpooling, dilations, norm_layer,
                   progress, **kwargs)


def resnext101_32x8d(stride_list, use_maxpooling, dilations, norm_layer, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], stride_list, use_maxpooling, dilations, norm_layer,
                   progress, **kwargs)