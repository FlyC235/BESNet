import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import ForwardRef
from torch._C import _logging_set_logger
from torch.nn.modules.container import Sequential

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
        logging.info("Global Average Pooling Initialized")

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)

class ConvBnReLU(nn.Sequential):
    def __init__(
        self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_ch))

        if relu:
            self.add_module("relu", nn.ReLU())
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def Upsample(x, size):
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=True)

def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                  dilation=atrous_rate, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True))
    return block

class AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AsppPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)
        return Upsample(pool, (h,w))

class ASPP_Module(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP_Module, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = AsppPooling(in_channels, out_channels)

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return y

class BE_Module(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch5, mid_ch, out_ch, n_class):
        super(BE_Module, self).__init__()
        
        self.convb_1 = ConvBnReLU(in_ch1, mid_ch, kernel_size=1, stride=1, padding=0, dilation=1)
        self.convb_2 = ConvBnReLU(in_ch2, mid_ch, kernel_size=1, stride=1, padding=0, dilation=1)
        self.convb_5 = ConvBnReLU(in_ch5, mid_ch, kernel_size=1, stride=1, padding=0, dilation=1)
        self.convbloss = nn.Conv2d(mid_ch, n_class, kernel_size=1, bias=False)
        boundary_ch = 3 * mid_ch
        self.boundaryconv = ConvBnReLU(boundary_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, l1, l2, l5):
        l1_b = self.convb_1(l1)
        l1_bl = self.convbloss(l1_b)

        l2_b = self.convb_2(l2)
        l2_bl = self.convbloss(l2_b)

        l5_b = self.convb_5(l5)
        l5_b = F.interpolate(l5_b, l1.size()[2:], mode='bilinear', align_corners=True)
        l5_bl = self.convbloss(l5_b)

        b = torch.cat((l1_b, l2_b, l5_b), dim=1)
        b = self.boundaryconv(b)

        c_boundaryloss = l1_bl + l2_bl + l5_bl

        return b, c_boundaryloss

class MSF_Module(nn.Module):
    def __init__(self, in_ch, mid_ch1, cat_ch, mid_ch2, out_ch):
        super(MSF_Module,self).__init__()

        self.input1 = ConvBnReLU(in_ch[0], mid_ch1, kernel_size=1, stride=1, padding=0, dilation=1)
        self.input2 = ConvBnReLU(in_ch[1], mid_ch1, kernel_size=1, stride=1, padding=0, dilation=1)
        self.input3 = ConvBnReLU(in_ch[2], mid_ch1, kernel_size=1, stride=1, padding=0, dilation=1)

        self.fusion1 = nn.Sequential(
            ConvBnReLU(cat_ch, mid_ch2, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.Conv2d(mid_ch2, mid_ch2, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Sigmoid(),
            GlobalAvgPool2d()
            )

        self.fusion2 = nn.Sequential(
            ConvBnReLU(cat_ch, mid_ch2, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.Conv2d(mid_ch2, out_ch, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Sigmoid(),
            GlobalAvgPool2d()
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, l3, l4, l5):

        x1 = self.input1(l3)
        x2 = self.input2(l4)
        x3 = self.input3(l5)

        w1 = torch.cat((x2, x3), dim=1)
        w1 = self.fusion1(w1).unsqueeze(2).unsqueeze(3).expand_as(x3)
        m1 = (1-w1)*x2 + w1*x3

        w2 = torch.cat((m1, x1), dim=1)
        w2 = self.fusion2(w2).unsqueeze(2).unsqueeze(3).expand_as(x3)
        m2 = (1-w2)*x1 + w2*m1

        return m2

class BES_Module(nn.Module):
    def __init__(self, f5_in, mul_ch):
        super(BES_Module, self).__init__()
        aspp_out = 5 * f5_in // 8
        self.aspp = ASPP_Module(f5_in, atrous_rates = [12, 24, 36])
        self.f5_out = ConvBnReLU(aspp_out, mul_ch, kernel_size=3, stride=1, padding=1, dilation=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, f5, fb, ff):
        aspp = self.aspp(f5)
        f5 = self.f5_out(aspp)
        f5 = F.interpolate(f5, fb.size()[2:], mode='bilinear', align_corners=True)
        f5_guide = torch.mul(f5, fb)
        ff_guide = torch.mul(ff, fb)
        fe = ff + ff_guide + f5_guide

        return fe
