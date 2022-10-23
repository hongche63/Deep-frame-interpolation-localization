from torch import nn
import numpy as np
from Utils import make_layers
import torch
import logging
from collections import OrderedDict
import argparse
from data.mm import MovingMNIST
device = torch.device('cuda:1')
from tqdm import tqdm
from MOUDLE import net
from torch.utils.data import DataLoader
from torchvision.utils import save_image
# from ConvTranspose import decode
# device = torch.device('cuda')

from torch import nn
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import cv2
import os
from torch.utils.data import Dataset
from torchvision import transforms

datapath1 = '/home/dingxl/totaldatabase/train_frame'
txtpath1='/home/dingxl/totaldatabase/EMEtrue_trainlist.txt'

datapath2 =  '/home/dingxl/totaldatabase/test_frame'
txtpath2='/home/dingxl/totaldatabase/EMEtrue_testlist.txt'

transform = transforms.Compose(
[transforms.ToTensor(),#将0-255映射成0-1
transforms.Normalize([0.5], [0.5])])#加快模型收敛(均值（每个通道对应1个均值），标准差（每个通道对应1个标准差)

def save_model(model,epoch_index):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join("./outputgru", "epoch%d_checkpoint.pkl" % epoch_index)
    torch.save(model_to_save.state_dict(), model_checkpoint)


class MyDataset1(Dataset):
    def __init__(self,txtpath):
        imgs = []
        datainfo = open(txtpath, 'r')
        for line in datainfo:
            line = line.strip('\n')
            words = line.split()
            temp = words[0].split('.')
            words0 = temp[0].split('i')
            n = int(words0[1])
            imagelist = []
            for i in range(5):
                s = '%03d' % n
                temp = words0[0] + 'i' + str(s) + '.jpg'
                imagelist.append(temp)
                n += 1
            imgs.append((imagelist, words[1]))

        self.imgs = imgs
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        piclist,label = self.imgs[index]

        name1=datapath1+'/'+piclist[0]
        picr1 = cv2.imread(name1)
        picr1= transform(picr1)
        name2= datapath1 + '/' + piclist[1]
        picr2= cv2.imread(name2)
        picr2= transform(picr2)

        name3 = datapath1 + '/' + piclist[2]
        picr3 = cv2.imread(name3)
        picr3= transform(picr3)

        name4 = datapath1 + '/' + piclist[3]
        picr4 = cv2.imread(name4)
        picr4 = transform(picr4)

        name5 = datapath1 + '/' + piclist[4]
        picr5 = cv2.imread(name5)
        picr5 = transform(picr5)


        label = torch.tensor(int(label))
        return picr1,picr2,picr3,picr4,picr5,piclist[0],piclist[1],piclist[2],piclist[3],piclist[4],label

class MyDataset2(Dataset):
    def __init__(self,txtpath):
        imgs = []
        datainfo = open(txtpath, 'r')
        for line in datainfo:
            line = line.strip('\n')
            words = line.split()
            temp = words[0].split('.')
            words0 = temp[0].split('i')
            n = int(words0[1])
            imagelist = []
            for i in range(5):
                s = '%03d' % n
                temp = words0[0] + 'i' + str(s) + '.jpg'
                imagelist.append(temp)
                n += 1
            imgs.append((imagelist, words[1]))

        self.imgs = imgs
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        piclist,label = self.imgs[index]

        name1=datapath2+'/'+piclist[0]
        picr1 = cv2.imread(name1)
        picr1= transform(picr1)

        name2= datapath2 + '/' + piclist[1]
        picr2= cv2.imread(name2)
        picr2= transform(picr2)

        name3 = datapath2 + '/' + piclist[2]
        picr3 = cv2.imread(name3)
        picr3= transform(picr3)

        name4 = datapath2 + '/' + piclist[3]
        picr4 = cv2.imread(name4,)
        picr4 = transform(picr4)

        name5 = datapath2 + '/' + piclist[4]
        picr5 = cv2.imread(name5)
        picr5 = transform(picr5)


        label = torch.tensor(int(label))
        return picr1,picr2,picr3,picr4,picr5,piclist[0],piclist[1],piclist[2],piclist[3],piclist[4],label


data1 = MyDataset1(txtpath1)
data_loader1 = DataLoader(data1,batch_size=4,shuffle=True)#32

data2 = MyDataset2(txtpath2)
data_loader2 = DataLoader(data2,batch_size=4,shuffle=False)#32

from torchvision.utils import save_image

parser = argparse.ArgumentParser()

parser.add_argument('-cgru',
                    '--convgru',
                    help='use convgru as base cell',
                    action='store_true')
parser.add_argument('--batch_size',
                    default=15,
                    type=int,
                    help='mini-batch size')
parser.add_argument('-lr', default=1e-4, type=float, help='G learning rate')
parser.add_argument('-frames_input',
                    default=4,
                    type=int,
                    help='sum of input frames')
parser.add_argument('-frames_output',
                    default=4,
                    type=int,
                    help='sum of predict frames')
parser.add_argument('-epochs', default=500, type=int, help='sum of epochs')
args = parser.parse_args()


from torch.optim import lr_scheduler
import torch.optim as optim
lossfunction1 = nn.MSELoss().to(device)
lossfunction2=nn.L1Loss().to(device)
optimizer = optim.Adam(net.parameters(), lr=args.lr)
pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                  factor=0.5,
                                                  patience=4,
                                                  verbose=True)
for i in range(100):

    train_step=0
    trainloss=0
    testloss=0

    train_loss_list = []
    test_loss_list = []

    for data in data_loader1:
        train_step+=1
        picr1, picr2, picr3, picr4, picr5,name1,name2,name3,name4,name5,targerts = data
        inputs = torch.stack([picr1, picr2, picr3, picr4], dim=1).to(device)
        label = picr5.to(device)  # B,S,C,H,W
        optimizer.zero_grad()
        net.train()
        pred2,pred3,pred4,pred5 = net(inputs)  # B,S,C,H,W
        loss1 = lossfunction1(pred5, label)
        loss2 = lossfunction2(pred5, label)
        loss = loss1+loss2
        # loss=lossfunction1(pred5, label)
        loss.backward()
        torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
        optimizer.step()
        if(train_step%100==0):
            print(loss)
        trainloss+=loss
        del pred2,pred3,pred4,pred5,loss1,loss2,loss
    avg_trainloss=trainloss/len(data_loader1)

    train_loss_list.append(avg_trainloss)

    print('train_loss: epoch{}:{}'.format(i, avg_trainloss))

    save_model(net,i)
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    with torch.no_grad():

        for data in data_loader2:
            picr1, picr2, picr3, picr4, picr5,name1,name2,name3,name4,name5,targerts = data
            inputs = torch.stack([picr1, picr2, picr3, picr4], dim=1).to(device)
            label = picr5.to(device)
            # net.eval()
            pred2, pred3, pred4, pred5 = net(inputs)  # B,S,C,H,W
            loss1 = lossfunction1(pred5, label)
            loss2 = lossfunction2(pred5, label)
            loss = loss1 + loss2
            # loss = lossfunction1(pred5, label)
            testloss += loss
            del pred2, pred3, pred4, pred5, loss1, loss2, loss
        # #
        #     save_decod_img2(pred2.cpu().data,name2)
        #     save_decod_img3(pred3.cpu().data,name3)
        #     save_decod_img4(pred4.cpu().data,name4)
        #     save_decod_img5(pred5.cpu().data,name5)

    avg_testloss=testloss/len(data_loader2)
    test_loss_list.append(avg_testloss)

    print('test_loss: epoch{}:{}'.format(i, avg_testloss))

