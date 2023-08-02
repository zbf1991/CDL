#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, print_function
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import _ConvBnReLU, _ResLayer, _Stem


class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        logits = 0
        for stage in self.children():
                logits += stage(x)

        return logits


class _AuxHead(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(_AuxHead, self).__init__()
        self.aux_conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias = True)
        self.aux_conv2 = nn.Conv2d(out_ch, in_ch, kernel_size=1, bias = True)
        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):

        new_x = self.aux_conv1(x)
        new_x = F.relu(new_x)
        new_x = self.drop(new_x)
        x = x + self.aux_conv2(new_x)
        return x


class _AffinityHead(nn.Module):
    def __init__(self, in2_ch, in3_ch, in4_ch, in5_ch, out_ch):
        super(_AffinityHead, self).__init__()
        self.aff_module = nn.ModuleList([])
        # self.aff_module = []
        self.aff_conv_2 = nn.Sequential(
                        nn.Conv2d(in2_ch, out_ch //4, kernel_size=1, bias=False),
                        nn.BatchNorm2d(out_ch//4),
                        nn.ReLU(),
                        nn.Dropout2d(0.5)
                        )
        self.aff_conv_3 = nn.Sequential(
                        nn.Conv2d(in3_ch, out_ch//4, kernel_size=1, bias=False),
                        nn.BatchNorm2d(out_ch//4),
                        nn.ReLU(),
                        nn.Dropout2d(0.5)
                        )
        self.aff_conv_4 = nn.Sequential(
                        nn.Conv2d(in4_ch, out_ch//2, kernel_size=1, bias=False),
                        nn.BatchNorm2d(out_ch//2),
                        nn.ReLU(),
                        nn.Dropout2d(0.5)
                        )
        self.aff_conv_5 = nn.Sequential(
                        nn.Conv2d(in5_ch, out_ch, kernel_size=1, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(),
                        nn.Dropout2d(0.5)
                        )
        self.aff_module.append(self.aff_conv_2)
        self.aff_module.append(self.aff_conv_3)
        self.aff_module.append(self.aff_conv_4)
        self.aff_module.append(self.aff_conv_5)

        self.reduce_dim = nn.Conv2d(out_ch//4+out_ch//4+out_ch//2+out_ch//1, out_ch, kernel_size=1, bias=False)

    def forward(self, fts):
        fts_all = []
        for num, fts_single in enumerate(fts):
            fts_single = self.aff_module[num](fts_single)
            fts_all.append(fts_single)

        aff_fts = torch.cat(fts_all, dim=1)
        aff_fts = self.reduce_dim(aff_fts)

        return aff_fts




class DeepLabV2(nn.Sequential):
    """
    DeepLab v2: Dilated ResNet + ASPP
    Output stride is fixed at 8
    """

    def __init__(self, n_classes, n_blocks, atrous_rates):
        super(DeepLabV2, self).__init__()
        ch = [64 * 2 ** p for p in range(6)]
        self.add_module("layer1", _Stem(ch[0]))
        self.add_module("layer2", _ResLayer(n_blocks[0], ch[0], ch[2], 1, 1))
        self.add_module("layer3", _ResLayer(n_blocks[1], ch[2], ch[3], 2, 1))
        self.add_module("layer4", _ResLayer(n_blocks[2], ch[3], ch[4], 1, 2))
        self.add_module("layer5", _ResLayer(n_blocks[3], ch[4], ch[5], 1, 4))
        self.add_module("aspp", _ASPP(ch[5], n_classes, atrous_rates))
        self.add_module("aspp_2nd", _ASPP(ch[5], n_classes, atrous_rates))
        self.add_module("aspp_3rd", _ASPP(ch[5], n_classes, atrous_rates))
        self.add_module("affinity_3rd", _AffinityHead(in2_ch=256, in3_ch=512, in4_ch=1024, in5_ch=2048, out_ch=128))


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x2 = x.clone().detach()
        x = self.layer3(x)
        x3 = x.clone().detach()
        x = self.layer4(x)
        x4 = x.clone().detach()
        x = self.layer5(x)
        x5 = x.clone().detach()

        fts = []
        b,c,h,w = x5.size()
        x2 = F.interpolate(x2, size=(h,w), mode='bilinear', align_corners=False)
        fts.append(x2)
        fts.append(x3)
        fts.append(x4)
        fts.append(x5)
        fts_distance = self.affinity_3rd(fts)

        b,c,h,w = x.size()
        x_main = x[0:b//2,:,:,:]
        x_aux = x[b//2:,:,:,:]

        logits = self.aspp(x_main)
        logits_2nd = self.aspp_2nd(x_main)
        logits_3rd = self.aspp_3rd(x_aux)
        return logits, logits_2nd, logits_3rd, fts_distance



    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, _ConvBnReLU.BATCH_NORM):
                m.eval()


if __name__ == "__main__":
    model = DeepLabV2(
        n_classes=21, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
    )
    model.eval()
    image = torch.randn(1, 3, 513, 513)

    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)
