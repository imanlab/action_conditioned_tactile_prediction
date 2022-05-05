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

model_save_path = "/home/user/Robotics/tactile_prediction/tactile_prediction/models/simple_baseline_ACMLP/saved_models/box_only_"
train_data_dir = "/home/user/Robotics/Data_sets/box_only_dataset/train_linear_dataset_10c_10h/"
scaler_dir = "/home/user/Robotics/Data_sets/box_only_dataset/scalar_info/"

# unique save title:
model_save_path = model_save_path + "model_" + datetime.now().strftime("%d_%m_%Y_%H_%M/")
os.mkdir(model_save_path)

seed = 42
epochs = 50
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
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=6)
        validation_loader = torch.utils.data.DataLoader(dataset_validate, batch_size=batch_size, shuffle=True, num_workers=6)
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
        tactile_data = np.load(train_data_dir + value[1])
        experiment_number = np.load(train_data_dir + value[3])
        time_steps = np.load(train_data_dir + value[4])
        return [robot_data.astype(np.float32), tactile_data.astype(np.float32), experiment_number, time_steps]


class simple_ACMLP(nn.Module):
    def __init__(self):
        super(simple_ACMLP, self).__init__()
        self.fc1 = nn.Linear((48*10)+(6*20), (48*10)).to(device)
        self.tan_activation = nn.Tanh().to(device)

    def forward(self, tactiles, actions):
        tactile = tactiles[:context_frames].permute(1, 0, 2)
        tactile = tactile.flatten(start_dim=1)
        actions = actions.permute(1, 0, 2).flatten(start_dim=1)
        tactile_action_and_state = torch.cat((actions, tactile), 1)
        output = self.tan_activation(self.fc1(tactile_action_and_state))
        return output


class ModelTrainer:
    def __init__(self):
        self.train_full_loader, self.valid_full_loader = BG.load_full_data()
        self.simple_ACMLP = simple_ACMLP()
        self.criterion = nn.L1Loss()
        self.criterion1 = nn.L1Loss()
        self.optimizer = optim.Adam(self.simple_ACMLP.parameters(), lr=learning_rate)

    def train_simple_ACMLP(self):
        plot_training_loss = []
        plot_validation_loss = []
        previous_val_mean_loss = 100.0
        best_val_loss = 100.0
        early_stop_clock = 0
        progress_bar = tqdm(range(0, epochs), total=(epochs*len(self.train_full_loader)))
        for epoch in progress_bar:
            losses = 0.0
            for index, batch_features in enumerate(self.train_full_loader):
                action  = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
                tactile = torch.flatten(batch_features[1], start_dim=2).permute(1, 0, 2).to(device)

                tactile_predictions = self.simple_ACMLP.forward(tactiles=tactile, actions=action)  # Step 3. Run our forward pass.
                self.optimizer.zero_grad()
                loss = self.criterion(tactile_predictions, tactile[context_frames:].permute(1, 0, 2).flatten(start_dim=1))
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
                    action  = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
                    tactile = torch.flatten(batch_features[1], start_dim=2).permute(1, 0, 2).to(device)

                    tactile_predictions = self.simple_ACMLP.forward(tactiles=tactile, actions=action)
                    self.optimizer.zero_grad()
                    val_loss = self.criterion1(tactile_predictions, tactile[context_frames:].permute(1, 0, 2).flatten(start_dim=1))
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
                    torch.save(self.simple_ACMLP, model_save_path + "ACTP_model")
                    best_val_loss = val_losses / index__
                early_stop_clock = 0
                previous_val_mean_loss = val_losses / index__


if __name__ == "__main__":
    BG = BatchGenerator()
    MT = ModelTrainer()
    MT.train_simple_ACMLP()

