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

from ACTVP.ACTVP_model import ACTVP

seed = 42

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#  use gpu if available


class BatchGenerator:
    def __init__(self, train_percentage, train_data_dir, batch_size, image_size):
        self.batch_size = batch_size
        self.image_size = image_size
        self.train_data_dir = train_data_dir
        self.train_percentage = train_percentage
        self.data_map = []
        with open(train_data_dir + 'map.csv', 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                self.data_map.append(row)

    def load_full_data(self):
        dataset_train = FullDataSet(self.data_map, self.train_data_dir, self.train_percentage, self.image_size, train=True)
        dataset_validate = FullDataSet(self.data_map, self.train_data_dir, self.train_percentage, self.image_size, validation=True)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=6)
        validation_loader = torch.utils.data.DataLoader(dataset_validate, batch_size=self.batch_size, shuffle=True, num_workers=6)
        self.data_map = []
        return train_loader, validation_loader


class FullDataSet:
    def __init__(self, data_map, train_percentage, train_data_dir, image_size, train=False, validation=False):
        self.train_data_dir = train_data_dir
        self.image_size = image_size
        if train:
            self.samples = data_map[1:int((len(data_map) * train_percentage))]
        if validation:
            self.samples = data_map[int((len(data_map) * train_percentage)): -1]
        data_map = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        value = self.samples[idx]
        robot_data = np.load(self.train_data_dir + value[0])

        if self.image_size == 0:
            tactile_data = np.load(self.train_data_dir + value[1])
        else:
            tactile_data = []
            for image_name in np.load(self.train_data_dir + value[2]):
                tactile_data.append(np.load(self.train_data_dir + image_name))
            np.array(tactile_data)

        experiment_number = np.load(self.train_data_dir + value[3])
        time_steps = np.load(self.train_data_dir + value[4])
        return [robot_data.astype(np.float32), tactile_data.astype(np.float32), experiment_number, time_steps]


class UniversalModelTrainer:
    def __init__(self, model, TL, VL, criterion, image_size, model_save_path, model_name, epochs, batch_size,
                 learning_rate, context_frames, sequence_length, train_percentage, validation_percentage):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.context_frames = context_frames
        self.sequence_length = sequence_length
        self.train_percentage = train_percentage
        self.validation_percentage = validation_percentage
        self.model_name = model_name
        self.model_save_path = model_save_path
        self.model = model
        self.image_size = image_size
        self.train_full_loader = TL
        self.valid_full_loader = VL
        if criterion == "L1":
            self.criterion = nn.L1Loss()
        if criterion == "L2":
            self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.full_model.parameters(), lr=learning_rate)

    def train_full_model(self):
        plot_training_loss = []
        best_training_loss = 0.000
        progress_bar = tqdm(range(0, self.epochs), total=(self.epochs*(len(self.train_full_loader) + len(self.valid_full_loader))))
        for epoch in progress_bar:
            self.train_loss = 0.0
            self.val_loss = 0.0
            for index, batch_features in enumerate(self.train_full_loader):
                self.optimizer.zero_grad()
                action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
                if self.image_size > 0:
                    tactile = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
                else:
                    tactile = torch.flatten(batch_features[1], start_dim=2).permute(1, 0, 2).to(device)
                loss = self.run_batch(tactile, action, train=True)
                progress_bar.set_description("epoch: {}, ".format(epoch) + "loss: {:.4f}, ".format(float(loss.item())) + "mean loss: {:.4f}, ".format(self.train_loss / (index+1)))
                train_max_index = index

            for index, batch_features in enumerate(self.valid_full_loader):
                self.optimizer.zero_grad()
                action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
                if self.image_size > 0:
                    tactile = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
                else:
                    tactile = torch.flatten(batch_features[1], start_dim=2).permute(1, 0, 2).to(device)
                loss = self.run_batch(tactile, action, train=False)
                progress_bar.set_description("epoch: {}, ".format(epoch) + "VAL loss: {:.4f}, ".format(float(loss.item())) + "VAL mean loss: {:.4f}, ".format(self.val_loss / (index+1)))
                val_max_index = index

            print("Training mean loss: {:.4f} || Validation mean loss: {:.4f}".format(self.train_loss/(train_max_index+1), self.val_loss/(val_max_index+1)))

            # early stopping and saving:
            if best_training_loss < self.val_loss/(val_max_index+1):
                best_training_loss = self.val_loss/(val_max_index+1)
                torch.save(self.full_model, self.model_save_path + self.model_name)

    def run_batch(self, tactile, action, train=True):
        tactile_predictions = self.model.forward(tactiles=tactile, actions=action)  # Step 3. Run our forward pass.
        loss = self.criterion(tactile_predictions, tactile[self.context_frames:])

        if train:
            loss.backward()
            self.optimizer.step()
            self.train_loss += loss.item()
        else:
            self.val_loss += loss.item()

        return loss.item()

def main():
    model_save_path = "/home/user/Robotics/tactile_prediction/tactile_prediction/models/ACTVP/saved_models/box_only_THDloss_"
    train_data_dir = "/home/user/Robotics/Data_sets/box_only_dataset/train_image_dataset_10c_10h/"

    # unique save title:
    model_save_path = model_save_path + "model_" + datetime.now().strftime("%d_%m_%Y_%H_%M/")
    os.mkdir(model_save_path)

    epochs = 50
    batch_size = 32
    learning_rate = 1e-3
    context_frames = 10
    sequence_length = 20
    train_percentage = 0.9
    validation_percentage = 0.1
    image_size = 32
    criterion = "L1"

    model = ACTVP()

    model_name = "ACTVP"
    BG = BatchGenerator()
    train_loader, val_loader = BG.load_full_data(train_percentage, train_data_dir)
    UniversalModelTrainer(model, train_loader, val_loader, criterion, image_size, model_save_path, model_name,
                          epochs, batch_size, learning_rate, context_frames, sequence_length,
                          train_percentage, validation_percentage)  # if not an image set image size to 0


if __name__ == "__main__":
    main()
