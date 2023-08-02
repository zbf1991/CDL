#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2018-03-26

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_fts_distance(aff_fts):
    b, c, h, w = aff_fts.size()
    aff_fts = aff_fts.view(b, c, h * w)
    #fts_norm = torch.norm(aff_fts, 2, 1, True)
    fts_dis = torch.bmm(aff_fts.transpose(2, 1), aff_fts) #/ (torch.bmm(fts_norm.transpose(2, 1), fts_norm) + 1e-5)
    fts_dis = torch.sigmoid(fts_dis)
    return fts_dis


class MSC(nn.Module):
    """
    Multi-scale inputs
    """

    def __init__(self, base, scales=None):
        super(MSC, self).__init__()
        self.base = base
        if scales:
            self.scales = scales
        else:
            self.scales = [0.5, 0.75]

    def forward(self, x, x_aug):
        x = torch.cat([x, x_aug], dim=0)
        logits, logits_2nd, logits_3rd, features_3rd = self.base(x)
        features_3rd_dis = compute_fts_distance(features_3rd)

        _, _, H, W = logits.shape
        interp = lambda l: F.interpolate(
            l, size=(H, W), mode="bilinear", align_corners=False
        )

        # Scaled
        logits_pyramid = []
        logits_pyramid_2nd = []
        logits_pyramid_3rd = []
        features_pyramid_3rd = []
        features_pyramid_3rd_dis = []

        for p in self.scales:
            h = F.interpolate(x, scale_factor=p, mode="bilinear", align_corners=False)
            logits_pyramid_single, logits_pyramid_single_2nd, logits_pyramid_single_3rd, features_pyramid_single_3rd = self.base(h)
            logits_pyramid.append(logits_pyramid_single)
            logits_pyramid_2nd.append(logits_pyramid_single_2nd)
            logits_pyramid_3rd.append(logits_pyramid_single_3rd)
            features_pyramid_single_3rd_dis = compute_fts_distance(features_pyramid_single_3rd)
            features_pyramid_3rd.append(features_pyramid_single_3rd)
            features_pyramid_3rd_dis.append(features_pyramid_single_3rd_dis)



        # Pixel-wise max
        logits_all = [logits] + [interp(l) for l in logits_pyramid]
        logits_max = torch.max(torch.stack(logits_all), dim=0)[0]

        # 2nd Pixel-wise max
        logits_all_2nd = [logits_2nd] + [interp(l) for l in logits_pyramid_2nd]
        logits_max_2nd = torch.max(torch.stack(logits_all_2nd), dim=0)[0]

        # aux Pixel-wise max
        logits_all_3rd = [logits_3rd] + [interp(l) for l in logits_pyramid_3rd]
        logits_max_3rd = torch.max(torch.stack(logits_all_3rd), dim=0)[0]
        
        features_pyramid_all_3rd = [features_3rd] + [interp(l) for l in features_pyramid_3rd]
        features_mean_3rd = torch.mean(torch.stack(features_pyramid_all_3rd), dim=0)
        features_mean_3rd_dis = compute_fts_distance(features_mean_3rd)



        if self.training:
            return [logits] + logits_pyramid + [logits_max], [logits_2nd] + logits_pyramid_2nd + [logits_max_2nd], \
                   [logits_3rd] + logits_pyramid_3rd + [logits_max_3rd], [features_3rd_dis] + features_pyramid_3rd_dis + [features_mean_3rd_dis] 
        else:
            return (logits_max_3rd)/2.0