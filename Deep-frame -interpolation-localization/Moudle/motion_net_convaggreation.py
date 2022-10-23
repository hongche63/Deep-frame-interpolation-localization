#after fusion,the size is not change
import torch
import torch.nn as nn
from torch.nn import Sequential, Flatten
import torch.nn.functional as F
# from multi_scale_aggreation import transformers
from motion_attention import guideattention1,guideattention2,guideattention3
# from convaggreation import conv_aggreation
import math
import torch.nn.functional as F
class downsamping(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(downsamping, self).__init__()
        self.block11 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1,
                      stride=2,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(num_features=out_channels))
        self.block12 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=True),
            nn.BatchNorm2d(num_features=out_channels))
        self.block13 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1,
                      stride=2,
                      padding=0,
                      bias=True),
            nn.Sigmoid())

    def forward(self, x):
        out11 = self.block11(x)
        out12 = self.block12(x)
        out1 = out11 + out12
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        out2 = self.block13(x)
        out = out1 * out2
        out = nn.SiLU(out)
        return out12

class fusion(nn.Module):#尺寸减半
    def __init__(self, in_channels, out_channels, **kwargs):
        super(fusion, self).__init__()
        self.bn1=nn.BatchNorm2d(num_features=in_channels)
        self.bn2=nn.BatchNorm2d(num_features=in_channels)
        self.block11 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=2*in_channels,
                       out_channels=out_channels,
                       kernel_size=1,
                       stride=2,
                       padding=0,
                       bias=True),
            nn.BatchNorm2d(num_features=out_channels))
        self.block12 = nn.Sequential(
            nn.Conv2d(in_channels=2*in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=True),
            nn.BatchNorm2d(num_features=out_channels))
        self.block13=nn.Sequential(
            nn.Conv2d(in_channels=2*in_channels,
                      out_channels=out_channels,
                      kernel_size=1,
                      stride=2,
                      padding=0,
                      bias=True),
            nn.Sigmoid())

    def forward(self,x,y):
        x1=self.bn1(x)
        x2=self.bn2(y)
        x=torch.cat((x, y), 1)
        out11=self.block11(x)
        out12=self.block12(x)
        out1=out11+out12
        x=torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        out2=self.block13(x)
        out=out1*out2
        nn.SiLU(out)
        return out

class block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(block, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.block21 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                       out_channels=out_channels,
                       kernel_size=1,
                       padding=0,
                       bias=True),
            nn.BatchNorm2d(num_features=out_channels))
        self.block22 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=True),
            nn.BatchNorm2d(num_features=out_channels))
        self.sse=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1,
                      padding=0,
                      bias=True),
            nn.Sigmoid())
    def forward(self, x):
        out21=self.block21(x)
        out22=self.block22(x)
        out23=self.bn1(x)
        out231=torch.nn.functional.adaptive_avg_pool2d(out23, (1, 1))
        out23=out23*self.sse(out231)
        out=out21+out22+out23
        nn.SiLU(out)
        return out

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.guideattention1 = guideattention1
        self.guideattention2 = guideattention2
        self.guideattention3 = guideattention3

        self.downsamping1=downsamping(6,30)
        self.downsamping2 = downsamping(30, 60)
        self.downsamping3 = downsamping(60, 90)
        self.downsamping4 = downsamping(90, 180)
        self.downsamping8 = downsamping(60, 90)
        self.downsamping11 = downsamping(180, 360)#64不确定值

        self.block14=block(60,60)
        self.block15 = block(60,60)
        self.block16 = block(60,60)
        self.block17 = block(60,60)

        self.block24 = block(90, 90)
        self.block25 = block(90, 90)
        self.block26 = block(90, 90)
        self.block27 = block(90, 90)
        self.block28 = block(90, 90)

        self.block35 = block(180,180)
        self.block36 = block(180,180)
        self.block37 = block(180,180)
        self.block38 = block(180,180)
        self.block39 = block(180,180)

        self.fusion9=fusion(90,180)
        self.fusion10=fusion(180,180)

        self.flatten = Flatten()
        self.fn = nn.Linear(in_features=450, out_features=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x,EME):
        out31=self.downsamping1(x)
        out32=self.downsamping2(out31)
        out33=self.downsamping3(out32)
        out34=self.downsamping4(out33)

        out35=self.block35(out34)
        out36 = self.block36(out35)
        out37 = self.block37(out36)
        out38 = self.block38(out37)
        out39 = self.block39(out38)

        out24=self.block24(out33)
        out25 = self.block25(out24)
        out26 = self.block26(out25)
        out27 = self.block27(out26)
        out28 = self.block28(out27)

        out14=self.block14(out32)
        out15=self.block15(out14)
        out16=self.block16(out15)
        out17=self.block17(out16)

        out18=self.downsamping8(out17)
        out18=self.guideattention1(EME,out18)

        out1=self.fusion9(out18,out28)
        out1=self.guideattention2(EME,out1)

        out1_F = F.interpolate(out1, size=out18.shape[-2:],
                                   mode='bilinear', align_corners=True)
        out=self.fusion10(out1,out39)
        out=self.guideattention3(EME,out)

        out_F = F.interpolate(out, size=out18.shape[-2:],
                                   mode='bilinear', align_corners=True)
        out=torch.cat((out18,out1_F,out_F),dim=1)
        return out
device = torch.device('cuda:1')
parnet_motion = net()
parnet_motion=parnet_motion.to(device)


