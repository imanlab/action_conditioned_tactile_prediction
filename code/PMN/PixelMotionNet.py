# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import copy
import numpy as np

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

seed = 42
epochs = 100
batch_size = 32
learning_rate = 1e-3
context_frames = 10
sequence_length = 20

train_percentage = 0.9
validation_percentage = 0.1

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size, padding=self.padding, bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device).to(device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device).to(device))


class PixelMotionNet(nn.Module):
    def __init__(self):
        super(PixelMotionNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1).cuda()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1).cuda()
        self.convlstm1 = ConvLSTMCell(input_dim=64, hidden_dim=32, kernel_size=(3, 3), bias=True).cuda()
        self.convlstm2 = ConvLSTMCell(input_dim=32, hidden_dim=32, kernel_size=(3, 3), bias=True).cuda()
        self.upconv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1).cuda()
        self.upconv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1).cuda()
        self.outconv = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1).cuda()

        self.relu1 = nn.ReLU(True)
        self.relu2 = nn.ReLU(True)
        self.relu3 = nn.ReLU(True)
        self.relu4 = nn.ReLU(True)
        self.tanh = nn.Tanh()

        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )

    def forward(self, tactiles, actions):
        self.batch_size = actions.shape[1]
        state = actions[0]
        state.to(device)
        batch_size__ = tactiles.shape[1]
        hidden_1, cell_1 = self.convlstm1.init_hidden(batch_size=self.batch_size, image_size=(8,8))
        hidden_2, cell_2 = self.convlstm2.init_hidden(batch_size=self.batch_size, image_size=(8,8))
        outputs = []

        for index, (sample_tactile, sample_action) in enumerate(zip(tactiles[0:-1].squeeze(), actions[1:].squeeze())):
            # sample_tactile.to(device)
            # sample_action.to(device)

            if index > context_frames-1:
                out1 = self.maxpool1 (self.relu1 (self.conv1 (output)))
                out2 = self.maxpool2 (self.relu2 (self.conv2 (out1)))

                hidden_1, cell_1 = self.convlstm1 (input_tensor=out2, cur_state=[hidden_1, cell_1])
                hidden_2, cell_2 = self.convlstm2 (input_tensor=hidden_1, cur_state=[hidden_2, cell_2])

                out3 = self.upsample1 (self.relu3 (self.upconv1 (hidden_2)))
                skip_connection = torch.cat ((out1, out3), axis=1)  # skip connection
                out4 = self.upsample2 (self.relu4 (self.upconv2 (skip_connection)))

                PixelMotionMap = self.tanh(self.outconv(out4))
                output = PixelMotionMap + output  # Final addition layer:

                outputs.append(output)

            else:
                out1 = self.maxpool1(self.relu1(self.conv1(sample_tactile)))
                out2 = self.maxpool2(self.relu2(self.conv2(out1)))

                hidden_1, cell_1 = self.convlstm1(input_tensor=out2, cur_state=[hidden_1, cell_1])
                hidden_2, cell_2 = self.convlstm2(input_tensor=hidden_1, cur_state=[hidden_2, cell_2])

                out3 = self.upsample1(self.relu3(self.upconv1(hidden_2)))
                skip_connection = torch.cat((out1, out3), axis=1)  # skip connection
                out4 = self.upsample2(self.relu4(self.upconv2(skip_connection)))

                PixelMotionMap = self.tanh(self.outconv(out4))
                output = PixelMotionMap + sample_tactile  # Final addition layer:

                last_output = output

        outputs = [last_output] + outputs
        return torch.stack(outputs)