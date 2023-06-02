# -*- coding: utf-8 -*-
# @Time : 20-6-3 下午5:14
# @Author : zhuying
# @Company : Minivision
# @File : MultiFTNet.py
# @Software : PyCharm
from torch import nn
import torch.nn.functional as F
from torch._dynamo.variables import torch

from src.model_lib.MiniFASNet import MiniFASNetV1,MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE


class FTGenerator(nn.Module):
    def __init__(self, in_channels=48, out_channels=1):
        super(FTGenerator, self).__init__()

        self.ft = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.ft(x)


class MultiFTNet(nn.Module):
    def __init__(self, img_channel=3, num_classes=3, embedding_size=128, conv6_kernel=(5, 5)):
        super(MultiFTNet, self).__init__()
        self.img_channel = img_channel
        self.num_classes = num_classes
        self.model = MiniFASNetV2(embedding_size=embedding_size, conv6_kernel=conv6_kernel, num_classes=num_classes, img_channel=img_channel)
        self.FTGenerator = FTGenerator(in_channels=128)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # load model weight
        state_dict = torch.load('../../resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth', map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.conv2_dw(x)
        x = self.model.conv_23(x)
        x = self.model.conv_3(x)
        x = self.model.conv_34(x)
        x = self.model.conv_4(x)
        x1 = self.model.conv_45(x)
        x1 = self.model.conv_5(x1)
        x1 = self.model.conv_6_sep(x1)
        x1 = self.model.conv_6_dw(x1)
        x1 = self.model.conv_6_flatten(x1)
        x1 = self.model.linear(x1)
        x1 = self.model.bn(x1)
        x1 = self.model.drop(x1)
        cls = self.model.prob(x1)

        if self.training:
            ft = self.FTGenerator(x)
            return cls, ft
        else:
            return cls
