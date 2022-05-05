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

model_save_path = "/home/user/Robotics/tactile_prediction/tactile_prediction/models/PixelMotionNet-AC/saved_models/box_only_dataset_WITH_ADD_"
train_data_dir = "/home/user/Robotics/Data_sets/box_only_dataset/train_image_dataset_10c_10h/"
scaler_dir = "/home/user/Robotics/Data_sets/box_only_dataset/scalar_info/"

# unique save title:
# model_save_path = model_save_path + "model_" + datetime.now().strftime("%d_%m_%Y_%H_%M/")
# os.mkdir(model_save_path)

seed = 42
epochs = 25
batch_size = 32
learning_rate = 1e-3
context_frames = 10
sequence_length = 20

train_percentage = 0.9
validation_percentage = 0.1

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#  use gpu if available


class BatchGenerator:
    def __init__(self):
        self.data_map = []
        with open(train_data_dir + 'map.csv', 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                self.data_map.append(row)

    def load_full_data(self):
        dataset_train = FullDataSet(self.data_map, train=True)
        dataset_validate = FullDataSet(self.data_map, validation=True)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
        validation_loader = torch.utils.data.DataLoader(dataset_validate, batch_size=batch_size, shuffle=True, num_workers=4)
        self.data_map = []
        return train_loader, validation_loader


class FullDataSet:
    def __init__(self, data_map, train=False, validation=False):
        if train:
            self.samples = data_map[1:int((len(data_map) * train_percentage))]
        if validation:
            self.samples = data_map[int((len(data_map) * train_percentage)): -1]
        data_map = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        value = self.samples[idx]
        robot_data = np.load(train_data_dir + value[0])

        tactile_images = []
        for image_name in np.load(train_data_dir + value[2]):
            tactile_images.append(np.load(train_data_dir + image_name))

        experiment_number = np.load(train_data_dir + value[3])
        time_steps = np.load(train_data_dir + value[4])
        return [robot_data.astype(np.float32), np.array(tactile_images).astype(np.float32), experiment_number, time_steps]


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
        return(torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device).to(device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device).to(device))


class ACPixelMotionNet(nn.Module):
    def __init__(self):
        super(ACPixelMotionNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1).cuda()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1).cuda()
        self.convlstm1 = ConvLSTMCell(input_dim=64, hidden_dim=32, kernel_size=(3, 3), bias=True).cuda()
        self.convlstm2 = ConvLSTMCell(input_dim=44, hidden_dim=32, kernel_size=(3, 3), bias=True).cuda()
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
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16 , 10, 10
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
                out1 = self.maxpool1(self.relu1(self.conv1(output)))
                out2 = self.maxpool2(self.relu2(self.conv2(out1)))

                hidden_1, cell_1 = self.convlstm1(input_tensor=out2, cur_state=[hidden_1, cell_1])

                # Add in tiled action and state:
                state_action = torch.cat((state, sample_action), 1)
                robot_and_tactile = torch.cat((torch.cat(8*[torch.cat(8*[state_action.unsqueeze(2)], axis=2).unsqueeze(3)], axis=3), hidden_1.squeeze()), 1)

                hidden_2, cell_2 = self.convlstm2(input_tensor=robot_and_tactile, cur_state=[hidden_2, cell_2])

                out3 = self.upsample1(self.relu3(self.upconv1(hidden_2)))
                skip_connection = torch.cat((out1, out3), axis=1)  # skip connection
                out4 = self.upsample2(self.relu4(self.upconv2(skip_connection)))

                PixelMotionMap = self.tanh(self.outconv(out4))
                # Final addition layer:
                output = PixelMotionMap + output
                outputs.append(output)

            else:
                out1 = self.maxpool1(self.relu1(self.conv1(sample_tactile)))
                out2 = self.maxpool2(self.relu2(self.conv2(out1)))

                hidden_1, cell_1 = self.convlstm1(input_tensor=out2, cur_state=[hidden_1, cell_1])

                # Add in tiled action and state:
                state_action = torch.cat((state, sample_action), 1)
                robot_and_tactile = torch.cat((torch.cat(8*[torch.cat(8*[state_action.unsqueeze(2)], axis=2).unsqueeze(3)], axis=3), hidden_1.squeeze()), 1)

                hidden_2, cell_2 = self.convlstm2(input_tensor=robot_and_tactile, cur_state=[hidden_2, cell_2])

                out3 = self.upsample1(self.relu3(self.upconv1(hidden_2)))
                skip_connection = torch.cat((out1, out3), axis=1)  # skip connection
                out4 = self.upsample2(self.relu4(self.upconv2(skip_connection)))

                PixelMotionMap = self.tanh(self.outconv(out4))
                # Final addition layer:
                output = PixelMotionMap + sample_tactile

                last_output = output

        outputs = [last_output] + outputs
        return torch.stack(outputs)


class ModelTrainer:
    def __init__(self):
        self.train_full_loader, self.valid_full_loader = BG.load_full_data()
        self.full_model = ACPixelMotionNet()
        self.criterion = nn.L1Loss()
        self.criterion1 = nn.L1Loss()
        self.optimizer = optim.Adam(self.full_model.parameters(), lr=learning_rate)

    def train_full_model(self):
        plot_training_loss = []
        plot_validation_loss = []
        previous_val_mean_loss = 100.0
        best_val_loss = 100.0
        early_stop_clock = 0
        progress_bar = tqdm(range(0, epochs), total=(epochs*len(self.train_full_loader)))
        for epoch in progress_bar:
            losses = 0.0
            for index, batch_features in enumerate(self.train_full_loader):
                tactile = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
                action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)

                tactile_predictions = self.full_model.forward(tactiles=tactile, actions=action)  # Step 3. Run our forward pass.
                self.optimizer.zero_grad()
                loss = self.criterion(tactile_predictions, tactile[context_frames:])
                loss.backward()
                self.optimizer.step()

                losses += loss.item()
                if index:
                    mean = losses / index
                else:
                    mean = 0

                progress_bar.set_description("epoch: {}, ".format(epoch) + "loss: {:.4f}, ".format(float(loss.item())) + "mean loss: {:.4f}, ".format(mean))
                progress_bar.update()

            plot_training_loss.append(mean)

            # Validation checking:
            val_losses = 0.0
            with torch.no_grad():
                for index__, batch_features in enumerate(self.valid_full_loader):
                    tactile = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
                    action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)

                    tactile_predictions = self.full_model.forward(tactiles=tactile, actions=action)
                    self.optimizer.zero_grad()
                    val_loss = self.criterion1(tactile_predictions.to(device), tactile[context_frames:])
                    val_losses += val_loss.item()

            plot_validation_loss.append(val_losses / index__)
            print("Validation mean loss: {:.4f}, ".format(val_losses / index__))

            # save the train/validation performance data
            np.save(model_save_path + "plot_validation_loss", np.array(plot_validation_loss))
            np.save(model_save_path + "plot_training_loss", np.array(plot_training_loss))

            # Early stopping:
            if previous_val_mean_loss < val_losses / index__:
                early_stop_clock += 1
                previous_val_mean_loss = val_losses / index__
                if early_stop_clock == 4:
                    print("Early stopping")
                    break
            else:
                if best_val_loss > val_losses / index__:
                    print("saving model")
                    torch.save(self.full_model, model_save_path + "ACPixelMotionNet_with_addition_model")
                    best_val_loss = val_losses / index__
                early_stop_clock = 0
                previous_val_mean_loss = val_losses / index__


if __name__ == "__main__":
    BG = BatchGenerator()
    MT = ModelTrainer()
    MT.train_full_model()

