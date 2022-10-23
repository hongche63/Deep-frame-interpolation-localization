from torch import nn
import numpy as np
from Utils import make_layers
import torch
import logging
from collections import OrderedDict
import argparse
from data.mm import MovingMNIST
device = torch.device('cuda:1')
# from tqdm import tqdm

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
class CGRU_cell(nn.Module):
    """
    ConvGRU Cell
    """
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CGRU_cell, self).__init__()
        self.shape = shape
        self.input_channels = input_channels
        # kernel_size of input_to_state equals state_to_state
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      2 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(1, 2 * self.num_features))
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      self.num_features, self.filter_size, 1, self.padding),
            nn.GroupNorm(1, self.num_features))

    def forward(self, inputs=None, hidden_state=None, seq_len=4):
        # seq_len=10 for moving_mnist
        if hidden_state is None:
            htprev = torch.zeros(inputs.size(1), self.num_features,
                                 self.shape[0], self.shape[1]).to(device)
        else:
            htprev = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(htprev.size(0), self.input_channels,
                                self.shape[0], self.shape[1]).to(device)
            else:
                x = inputs[index, ...]
            # print(x.shape,htprev.shape)
            combined_1 = torch.cat((x, htprev), 1)  # X_t + H_t-1

            gates = self.conv1(combined_1)  # W * (X_t + H_t-1)

            zgate, rgate = torch.split(gates, self.num_features, dim=1)
            # zgate, rgate = gates.chunk(2, 1)
            z = torch.sigmoid(zgate)
            r = torch.sigmoid(rgate)

            combined_2 = torch.cat((x, r * htprev),
                                   1)  # h' = tanh(W*(x+r*H_t-1))
            ht = self.conv2(combined_2)
            ht = torch.tanh(ht)
            htnext = (1 - z) * htprev + z * ht
            output_inner.append(htnext)
            htprev = htnext
        return torch.stack(output_inner), htnext
class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            # index sign from 1
            setattr(self, 'stage' + str(index), make_layers(params))
            setattr(self, 'rnn' + str(index), rnn)

    def forward_by_stage(self, inputs, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        outputs_stage, state_stage = rnn(inputs, None)
        return outputs_stage, state_stage

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)  # to S,B,1,64,64
        hidden_states = []
        logging.debug(inputs.size())
        for i in range(1, self.blocks + 1):
            inputs, state_stage = self.forward_by_stage(
                inputs, getattr(self, 'stage' + str(i)),
                getattr(self, 'rnn' + str(i)))
            hidden_states.append(state_stage)
        return tuple(hidden_states)

class Decoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index),
                    make_layers(params))

    def forward_by_stage(self, inputs, state, subnet, rnn):
        inputs, state_stage = rnn(inputs, state, seq_len=4)
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        return inputs

        # input: 5D S*B*C*H*W

    def forward(self, hidden_states):
        inputs = self.forward_by_stage(None, hidden_states[-1],
                                       getattr(self, 'stage4'),
                                       getattr(self, 'rnn4'))
        for i in list(range(1, self.blocks))[::-1]:
            inputs = self.forward_by_stage(inputs, hidden_states[i - 1],
                                           getattr(self, 'stage' + str(i)),
                                           getattr(self, 'rnn' + str(i)))
        inputs = inputs.transpose(0, 1)  # to B,S,1,64,64
        return inputs

convgru_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [3, 16, 3, 2, 1]}),
        OrderedDict({'conv2_leaky_1': [16, 32, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [32, 64, 3, 2, 1]}),
        OrderedDict({'conv4_leaky_1': [64, 128, 3, 2, 1]}),
    ],

    [   CGRU_cell(shape=(112,112), input_channels=16, filter_size=3, num_features=16),
        CGRU_cell(shape=(56,56), input_channels=32, filter_size=3, num_features=32),
        CGRU_cell(shape=(28,28), input_channels=64, filter_size=3, num_features=64),
        CGRU_cell(shape=(14,14), input_channels=128, filter_size=3, num_features=128)
    ]
]

convgru_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [128, 64, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [64, 32, 4, 2, 1]}),
        OrderedDict({'deconv3_leaky_1': [32, 16, 4, 2, 1]}),
        OrderedDict({'deconv4_leaky_1': [16, 3, 4, 2, 1]}),
    ],

    [
        CGRU_cell(shape=(14,14), input_channels=128, filter_size=3, num_features=128),
        CGRU_cell(shape=(28,28), input_channels=64, filter_size=3, num_features=64),
        CGRU_cell(shape=(56,56), input_channels=32, filter_size=3, num_features=32),
        CGRU_cell(shape=(112,112), input_channels=16, filter_size=3, num_features=16)
    ]
]

class ED(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        ##########Convgru_warp
        batch_size,seq_number,input_channel, height, width = input.size()
        state_first = self.encoder(input)
        output_first = self.decoder(state_first)
        first_image=(input[:, 0 :, :, :]).to(device)
        temp_input = (input[:, 1:seq_number:1, :, :, :]).to(device)
        temp_output=(output_first[:,0:-1:1,:,:,:]).to(device)
        temp_n_1=(2*temp_input-temp_output).to(device)
        Numpy=torch.cat((first_image,temp_n_1), 1).to(device)
        state_second = self.encoder(Numpy)
        output_second = self.decoder(state_second)
        return output_second[:,0,:,:,:],output_second[:,1,:,:,:],output_second[:,2,:,:,:],output_second[:,3,:,:,:]



encoder = Encoder(convgru_encoder_params[0], convgru_encoder_params[1]).to(device)
decoder = Decoder(convgru_decoder_params[0], convgru_decoder_params[1]).to(device)
net = ED(encoder, decoder).to(device)
