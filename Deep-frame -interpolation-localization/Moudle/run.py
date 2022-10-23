import torch
import csv
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from moudle import net
# from Net_convaggreation import net
device = torch.device('cuda:1')
from torch import nn
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import cv2
import os
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
#路径是自己电脑里所对应的路径
def save_model(model,epoch_index):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join("./output", "epoch%d_checkpoint.pkl" % epoch_index)
    torch.save(model_to_save.state_dict(), model_checkpoint)
# datapath1 = '/home/dingxl/whole_database/Adof/DANVIS_frame/train_flow/'
# datapath2= '/home/dingxl/whole_database/Adof/DANVIS_frame/total_EME/train_sequanceframe_EME/'
# datapath3= '/home/dingxl/whole_database/Adof/DANVIS_frame/total_EME/train_singleframe_EME/'
#
# datapath4 = '/home/dingxl/whole_database/Adof/DANVIS_frame/test_flow/'
# datapath5= '/home/dingxl/whole_database/Adof/DANVIS_frame/total_EME/test_sequanceframe_EME/'
# datapath6= '/home/dingxl/whole_database/Adof/DANVIS_frame/total_EME/test_singleframe_EME/'
#
# txtpath1 = '/home/dingxl/whole_database/List/DANVIS_trainlist.txt'
# txtpath2 = '/home/dingxl/whole_database/List/DANVIS_testlist.txt'
# datapath1 = '/home/dingxl/whole_database/Adof/diffent_frame_rate/frame_25_30/train_flow/'
# datapath2= '/home/dingxl/whole_database/Adof/diffent_frame_rate/frame_25_30/train_sequanceframe_EME/'
# datapath3= '/home/dingxl/whole_database/Adof/diffent_frame_rate/frame_25_30/train_singleframe_EME/'
#
# datapath4 = '/home/dingxl/whole_database/Adof/diffent_frame_rate/frame_25_30/test_flow/'
# datapath5= '/home/dingxl/whole_database/Adof/diffent_frame_rate/frame_25_30/test_sequanceframe_EME/'
# datapath6= '/home/dingxl/whole_database/Adof/diffent_frame_rate/frame_25_30/test_singleframe_EME/'
#
# txtpath1 = '/home/dingxl/whole_database/Adof/diffent_frame_rate/list/25_30/trainlist.txt'
# txtpath2 = '/home/dingxl/whole_database/Adof/diffent_frame_rate/list/25_30/testlist.txt'



datapath1 = '/home/dingxl/whole_database/Adof/diffent_frame_rate/frame_20_25/train_flow/'
datapath2= '/home/dingxl/whole_database/Adof/diffent_frame_rate/frame_20_25/train_sequanceframe_EME/'
datapath3= '/home/dingxl/whole_database/Adof/diffent_frame_rate/frame_20_25/train_singleframe_EME/'

datapath4 = '/home/dingxl/whole_database/Adof/diffent_frame_rate/frame_20_25/test_flow/'
datapath5= '/home/dingxl/whole_database/Adof/diffent_frame_rate/frame_20_25/test_sequanceframe_EME/'
datapath6= '/home/dingxl/whole_database/Adof/diffent_frame_rate/frame_20_25/test_singleframe_EME/'

txtpath1 = '/home/dingxl/whole_database/Adof/diffent_frame_rate/List/20_25/trainlist1.txt'
txtpath2 = '/home/dingxl/whole_database/Adof/diffent_frame_rate/List/20_25/testlist1.txt'

class MyDataset1(Dataset):
    def __init__(self,txtpath1):
        imgs = []
        datainfo = open(txtpath1, 'r')
        for line in datainfo:
            line = line.strip('\n')
            words = line.split()
            temp = words[0].split('.')
            words0 = temp[0].split('i')
            n = int(words0[1])
            imagelist = []
            for i in range(3):
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

        flow1 = cv2.imread(datapath1+'/'+piclist[0])
        flow1=transform(flow1)
        single_EME1=cv2.imread(datapath3+'/'+piclist[0])
        single_EME1 = transform(single_EME1)

        flow2= cv2.imread(datapath1+'/'+piclist[1])
        flow2= transform(flow2)
        single_EME2=cv2.imread(datapath3+'/'+piclist[1])
        single_EME2 = transform(single_EME2)


        single_EME3=cv2.imread(datapath3+'/'+piclist[2])
        single_EME3 = transform(single_EME3)


        sequance_EME = cv2.imread(datapath2 + '/' + piclist[2])
        sequance_EME = transform(sequance_EME)


        label = torch.tensor(int(label))
        return flow1,flow2,sequance_EME,single_EME1,single_EME2,single_EME3,label

class MyDataset2(Dataset):
    def __init__(self,txtpath2):
        imgs = []
        datainfo = open(txtpath2, 'r')
        for line in datainfo:
            line = line.strip('\n')
            words = line.split()
            temp = words[0].split('.')
            words0 = temp[0].split('i')
            n = int(words0[1])
            imagelist = []
            for i in range(3):
                s = '%03d' % n
                temp = words0[0] + 'i' + str(s) + '.jpg'
                imagelist.append(temp)
                n += 1
            imgs.append((imagelist, words[1]))

        self.imgs = imgs
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        piclist, label = self.imgs[index]

        flow1 = cv2.imread(datapath4 + '/' + piclist[0])
        flow1 = transform(flow1)
        single_EME1 = cv2.imread(datapath6 + '/' + piclist[0])
        single_EME1 = transform(single_EME1)

        flow2 = cv2.imread(datapath4 + '/' + piclist[1])
        flow2 = transform(flow2)
        single_EME2 = cv2.imread(datapath6 + '/' + piclist[1])
        single_EME2 = transform(single_EME2)

        single_EME3 = cv2.imread(datapath6 + '/' + piclist[2])
        single_EME3 = transform(single_EME3)

        sequance_EME = cv2.imread(datapath5 + '/' + piclist[2])
        sequance_EME = transform(sequance_EME)


        label = torch.tensor(int(label))
        return flow1, flow2, sequance_EME, single_EME1, single_EME2, single_EME3, label


data1 = MyDataset1(txtpath1)
data_loader1 = DataLoader(data1,batch_size=16,shuffle=True)

data2 = MyDataset2(txtpath2)
data_loader2 = DataLoader(data2,batch_size=1)


net= net.to(device)
lossfunction=nn.CrossEntropyLoss()
lossfunction=lossfunction.to(device)

learning_rate=0.000001#0.0008345137614500873
optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=0.0001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.99)


# total_train_step=0
# total_test_step=0
# epoch=100

test_loss=[]
test_acc=[]
train_loss=[]
train_acc=[]
Epoch=[]


acc=0
acc1=0

k=100#控制训练次数和写入excel

T=len(data2)*0.5
F=len(data2)*0.5
for i in range(k):
    testloss = 0
    test_acc=0
    trian_acc = 0
    total_train_step = 0
    correct = 0
    total = 0
    running_loss = 0
    print("第{}轮训练开始：".format(i+1))
    start_time=time.time()
    #训练步骤开始
    Epoch.append(i)
    net.train()

    for data in data_loader1:
        flow1, flow2, sequance_EME, single_EME1, single_EME2, single_EME3,targerts = data

        flow = torch.cat((flow1, flow2), 1)
        flow = flow.to(device)

        sequance_EME=sequance_EME.to(device)

        single_EME_group=torch.cat((single_EME1, single_EME2, single_EME3), 1)
        single_EME_group=single_EME_group.to(device)

        targerts = targerts.to(device)

        outputs = net(flow,sequance_EME,single_EME_group)
        loss=lossfunction(outputs,targerts)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step=total_train_step+1
        if total_train_step%1000==0:
            print("训练次数：{}，loss:{}".format(total_train_step,loss.item()))
        y_pred = torch.argmax(outputs, dim=1)
        correct += (y_pred == targerts).sum().item()
        total += targerts.size(0)
        running_loss += loss.item()
    epoch_loss = running_loss / len(data_loader1)
    # epoch_acc = correct / total
    epoch_acc = correct / len(data1)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    end_time=time.time()
    print("训练第{}次的时间是{}".format(i + 1, end_time - start_time))
    print("train_loss：{}".format(epoch_loss ))
    print("train_acc：{}".format(epoch_acc))
    save_model(net,i+ 1)

    net.eval()
    total_test_loss=0
    total_accuracy=0
    with torch.no_grad():
        n1 = 0
        n2 = 0
        for data in data_loader2:
            flow1, flow2,sequance_EME, single_EME1, single_EME2, single_EME3,targerts = data

            flow = torch.cat((flow1, flow2), 1)
            flow = flow.to(device)

            sequance_EME = sequance_EME.to(device)

            single_EME_group = torch.cat((single_EME1, single_EME2, single_EME3), 1)
            single_EME_group = single_EME_group.to(device)

            targerts = targerts.to(device)

            outputs = net(flow, sequance_EME, single_EME_group)
            loss = lossfunction(outputs, targerts)
            testloss += loss
            if ((outputs.argmax(1).item() == 0) & (targerts.item() == 0)):
                n1 = n1 + 1
            if ((outputs.argmax(1).item() == 1) & (targerts.item() == 1)):
                n2 = n2 + 1
            accuracy = (outputs.argmax(1) == targerts).sum().item()
            test_acc += accuracy
    avg_testloss = testloss / len(data_loader2)
    avg_testacc = test_acc / len(data2)

    scheduler.step()

    print('test_loss: epoch{}:{}'.format(i, avg_testloss))
    print('test_accuracy: epoch{}:{}'.format(i, avg_testacc))
    PFACC = n1 / (T)
    FFACC = n2 / (F)
    FACC = (n1 + n2) / len(data2)
    Precision = n2 / (n2 + T- n1)
    Recall = n2 / (n2+F-n2)
    if Precision + Recall == 0:
        F1score = 0
    else:
        F1score = 2 * Precision * Recall / (Precision + Recall)
    print('n1,n2:', n1, n2)
    print('PFACC:', PFACC)
    print('FFACC:', FFACC)
    print('FACC:', FACC)
    print('Precision:', Precision)
    print('Recall:', Recall)
    print('F1score:', F1score)
