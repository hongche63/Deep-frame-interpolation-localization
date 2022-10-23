from torch import nn
import torch
import torch.nn.functional as F
class GuidedAttention(nn.Module):
    """ Reconstruction Guided Attention. """
    def __init__(self, depth, drop_rate=0.2):
        super(GuidedAttention, self).__init__()
        self.depth = depth
        self.gated = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(3, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.h = nn.Sequential(
            nn.Conv2d(depth, depth, 1, 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.ReLU(True),
        )
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, EME, feature):
        EME = F.interpolate(EME, size=feature.shape[-2:],
                                   mode='bilinear', align_corners=True)
        EME = self.gated(EME)
        return EME * self.h(feature) + self.dropout(feature)

guideattention1=GuidedAttention(depth=90)
guideattention2=GuidedAttention(depth=180)
guideattention3=GuidedAttention(depth=180)
# feature1=torch.randn(3,192,13,13)
# feature2=torch.randn(3,384,7,7)
# feature3=torch.randn(3,384,4,4)
# EME=torch.randn(3,1,100,100)
# out1=guideattention1(EME,feature1)
# out2=guideattention2(EME,feature2)
# out3=guideattention3(EME,feature3)
# print(out1.shape,out2.shape,out3.shape)