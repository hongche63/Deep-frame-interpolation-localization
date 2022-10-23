from motion_net_convaggreation import parnet_motion
from stem_transformer_FFN import transformer
import torch.nn.functional as F
import torch
from torch import nn
device = torch.device('cuda:1')

class Moudle(nn.Module):
    def __init__(self,flatten=630,num_class=2):
        super(Moudle, self).__init__()
        self.transformer=transformer
        self.parnet_motion=parnet_motion
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classify = nn.Linear(flatten, num_class)
    def forward(self, flow,sequance_EME,single_EME):
        out_parnet_motion=self.parnet_motion(flow,sequance_EME)#(3,450,28,28)
        out_transformer=self.transformer(single_EME)#(3,180,56,56)
        out_transformer = F.interpolate(out_transformer, size=out_parnet_motion.shape[-2:],
                                   mode='bilinear', align_corners=True)
        out=torch.cat((out_parnet_motion,out_transformer),dim=1)
        out=self._avg_pooling(out).flatten(start_dim=1)
        out=self.classify(out)
        return out
net=Moudle()
net=net.to(device)
