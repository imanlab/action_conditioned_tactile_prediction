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

model_save_path = "/home/user/Robotics/tactile_prediction/tactile_prediction/models/ACTP/saved_models/box_only_MSEloss_200hidden_"
train_data_dir = "/home/user/Robotics/Data_sets/box_only_dataset/train_linear_dataset_10c_10h/"
scaler_dir = "/home/user/Robotics/Data_sets/box_only_dataset/scalar_info/"

# unique save title:
# model_save_path = model_save_path + "model_" + datetime.now().strftime("%d_%m_%Y_%H_%M/")
# os.mkdir(model_save_path)

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


class ACTP(nn.Module):
    def __init__(self):
        super(ACTP, self).__init__()
        self.lstm1 = nn.LSTM(48, 200).to(device)  # tactile
        self.lstm2 = nn.LSTM(200 + 48, 200).to(device)  # tactile
        self.fc1 = nn.Linear(200 + 48, 200).to(device)  # tactile + pos
        self.fc2 = nn.Linear(200, 48).to(device)  # tactile + pos
        self.tan_activation = nn.Tanh().to(device)
        self.relu_activation = nn.ReLU().to(device)

    def forward(self, tactiles, actions):
        state = actions[0]
        state.to(device)
        batch_size__ = tactiles.shape[1]
        outputs = []
        hidden1 = (torch.zeros(1, batch_size__, 200, device=torch.device('cuda')), torch.zeros(1, batch_size__, 200, device=torch.device('cuda')))
        hidden2 = (torch.zeros(1, batch_size__, 200, device=torch.device('cuda')), torch.zeros(1, batch_size__, 200, device=torch.device('cuda')))

        for index, (sample_tactile, sample_action,) in enumerate(zip(tactiles.squeeze()[:-1], actions.squeeze()[1:])):
            # 2. Run through lstm:
            if index > context_frames-1:
                out4 = out4.squeeze()
                out1, hidden1 = self.lstm1(out4.unsqueeze(0), hidden1)
                tiled_action_and_state = torch.cat((sample_action, state, sample_action, state, sample_action, state, sample_action, state), 1)
                action_and_tactile = torch.cat((out1.squeeze(), tiled_action_and_state), 1)
                out2, hidden2 = self.lstm2(action_and_tactile.unsqueeze(0), hidden2)
                lstm_and_prev_tactile = torch.cat((out2.squeeze(), out4), 1)
                out3 = self.tan_activation(self.fc1(lstm_and_prev_tactile))
                out4 = self.tan_activation(self.fc2(out3))
                outputs.append(out4.squeeze())
            else:
                out1, hidden1 = self.lstm1(sample_tactile.unsqueeze(0), hidden1)
                tiled_action_and_state = torch.cat((sample_action, state, sample_action, state, sample_action, state, sample_action, state), 1)
                action_and_tactile = torch.cat((out1.squeeze(), tiled_action_and_state), 1)
                out2, hidden2 = self.lstm2(action_and_tactile.unsqueeze(0), hidden2)
                lstm_and_prev_tactile = torch.cat((out2.squeeze(), sample_tactile), 1)
                out3 = self.tan_activation(self.fc1(lstm_and_prev_tactile))
                out4 = self.tan_activation(self.fc2(out3))
                last_output = out4

        outputs = [last_output] + outputs
        return torch.stack(outputs)

    # def forward(self, tactiles, actions):
    #     state = actions[0]
    #     state.to(device)
    #     batch_size__ = tactiles.shape[1]
    #     outputs = []
    #     hidden1 = (torch.zeros(1, batch_size__, 200, device=torch.device('cuda')), torch.zeros(1, batch_size__, 200, device=torch.device('cuda')))
    #     hidden2 = (torch.zeros(1, batch_size__, 200, device=torch.device('cuda')), torch.zeros(1, batch_size__, 200, device=torch.device('cuda')))
    #
    #     for index, (sample_tactile, sample_action,) in enumerate(zip(tactiles.squeeze()[:-1], actions.squeeze()[1:])):
    #         # 2. Run through lstm:
    #         if index > context_frames-1:
    #             out4 = out4.squeeze()
    #             tiled_action_and_state = torch.cat((actions.squeeze()[index+1], state), 1)
    #             action_and_tactile = torch.cat((out4, tiled_action_and_state), 1)
    #             out1, hidden1 = self.lstm1(action_and_tactile.unsqueeze(0), hidden1)
    #             out2, hidden2 = self.lstm2(out1, hidden2)
    #             lstm_and_prev_tactile = torch.cat((out2.squeeze(), out4), 1)
    #             out3 = self.tan_activation(self.fc1(lstm_and_prev_tactile))
    #             out4 = self.tan_activation(self.fc2(out3))
    #             outputs.append(out4.squeeze())
    #         else:
    #             out1, hidden1 = self.lstm1(sample_tactile, hidden1)
    #             tiled_action_and_state = torch.cat((actions.squeeze()[index+1], state), 1)
    #             action_and_tactile = torch.cat((out1, tiled_action_and_state), 1)
    #             out2, hidden2 = self.lstm2(action_and_tactile, hidden2)
    #             lstm_and_prev_tactile = torch.cat((out2.squeeze(), sample_tactile), 1)
    #             out3 = self.tan_activation(self.fc1(lstm_and_prev_tactile))
    #             out4 = self.tan_activation(self.fc2(out3))
    #             last_output = out4
    #
    #     outputs = [last_output] + outputs
    #     return torch.stack(outputs)

class ModelTrainer:
    def __init__(self):
        self.train_full_loader, self.valid_full_loader = BG.load_full_data()
        self.full_model = ACTP()
        self.criterion  = nn.MSELoss()  # nn.L1Loss()
        self.criterion1 = nn.MSELoss()  # nn.L1Loss()
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
                action  = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
                tactile = torch.flatten(batch_features[1], start_dim=2).permute(1, 0, 2).to(device)

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
                    action  = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
                    tactile = torch.flatten(batch_features[1], start_dim=2).permute(1, 0, 2).to(device)

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
                    torch.save(self.full_model, model_save_path + "ACTP_model")
                    best_val_loss = val_losses / index__
                early_stop_clock = 0
                previous_val_mean_loss = val_losses / index__


if __name__ == "__main__":
    BG = BatchGenerator()
    MT = ModelTrainer()
    MT.train_full_model()

