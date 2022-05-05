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

from ACTP.test_ACTP import ACTP
from ACTVP.test_model import ACTVP
from ACTVP.test_model import ConvLSTMCell
# from PixelMotionNet.PixelMotionNet import PixelMotionNet
# from PixelMotionNet_AC.AC_PMN import ACPixelMotionNet
# from simple_baseline.simple_MLP_model import simple_MLP
# from simple_baseline_ACMLP.simple_ACMLP_model import simple_ACMLP
# import SVG.models.lstm as lstm_models
# import SVG.models.dcgan_32 as model
#
# # SV2P & CDNA:
# import SV2P.sv2p.cdna as cdna
# from SV2P.sv2p.ssim import DSSIM
# from SV2P.sv2p.ssim import Gaussian2d
# from SV2P.sv2p.model import PosteriorInferenceNet, LatentModel
# from SV2P.sv2p.criteria import RotationInvarianceLoss

seed = 42

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available


class BatchGenerator:
    def __init__(self, test_data_dir, batch_size, image_size, trial_number):
        self.batch_size = batch_size
        self.image_size = image_size
        self.trial_number = trial_number
        self.test_data_dir = test_data_dir + 'test_trial_' + str(self.trial_number) + '/'
        self.data_map = []
        with open(self.test_data_dir + 'map_' + str(self.trial_number) + '.csv', 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                self.data_map.append(row)

    def load_full_data(self):
        dataset_test = FullDataSet(self.data_map, self.test_data_dir, self.image_size)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False)
        self.data_map = []
        return test_loader


class FullDataSet:
    def __init__(self, data_map, test_data_dir, image_size):
        self.image_size = image_size
        self.test_data_dir = test_data_dir
        self.samples = data_map[1:]
        data_map = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        value = self.samples[idx]
        robot_data = np.load(self.test_data_dir + value[0])

        if self.image_size == 0:
            tactile_data = np.load(self.test_data_dir + value[1])
        else:
            tactile_data = []
            for image_name in np.load(self.test_data_dir + value[2]):
                tactile_data.append(np.load(self.test_data_dir + image_name))

        experiment_number = np.load(self.test_data_dir + value[3])
        time_steps = np.load(self.test_data_dir + value[4])
        return [robot_data.astype(np.float32),  np.array(tactile_data).astype(np.float32), experiment_number, time_steps]


class UniversalModelTester:
    def __init__(self, model, number_of_trials, test_data_dir, image_size, criterion, model_path, data_save_path,
                 scaler_dir, model_name, epochs, batch_size, learning_rate, context_frames, sequence_length):
        self.number_of_trials = number_of_trials
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.context_frames = context_frames
        self.sequence_length = sequence_length
        self.model_name = model_name
        self.model_save_path = model_path
        self.data_save_path = data_save_path
        self.test_data_dir = test_data_dir
        self.scaler_dir = scaler_dir
        self.model = model
        self.image_size = image_size
        if criterion == "L1":
            self.criterion = nn.L1Loss()
        if criterion == "L2":
            self.criterion = nn.MSELoss()

    def test_model(self):
        for trial in range(self.number_of_trials):
            print(trial)
            BG = BatchGenerator(self.test_data_dir, self.batch_size, self.image_size, trial)
            test_full_loader = BG.load_full_data()
            prediction_data = []

            for index, batch_features in enumerate(test_full_loader):
                tactile = batch_features[1].permute(1, 0, 4, 3, 2).to(device)
                action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
                tactile_predictions, loss = self.run_batch(tactile, action)

                # tactile_predictions = self.model.forward(tactiles=tactile, actions=action)  # Step 3. Run our forward pass.
                # experiment_number = batch_features[2].permute(1, 0)[self.context_frames:]
                # time_steps = batch_features[3].permute(1, 0)[self.context_frames:]
                # self.meta = batch_features[4][0]

                prediction_data.append(tactile_predictions)

            prediction_data_descaled = self.scale_back(prediction_data)

    def scale_back(self, tactile_data):
        pass
        return tactile_data

    def run_batch(self, tactile, action):
        if self.image_size == 0:
            action = action.squeeze(-1).permute(1, 0, 2).to(device)
            tactile = torch.flatten(tactile, start_dim=2).permute(1, 0, 2).to(device)
        else:
            tactile = tactile.permute(1, 0, 2, 3, 4).to(device)
            action = action.squeeze(-1).permute(1, 0, 2).to(device)
        print(tactile.shape)

        tactile_predictions = self.model.forward(tactiles=tactile, actions=action)  # Step 3. Run our forward pass.
        loss = self.criterion(tactile_predictions, tactile[self.context_frames:])
        return tactile_predictions, loss.item()


def main():
    model_path_1 = "/home/user/Robotics/tactile_prediction/tactile_prediction/models/ACTVP/saved_models/box_only_model_30_11_2021_10_58/ACPixelMotionNet_model"
    model_path_2 = "/home/user/Robotics/tactile_prediction/tactile_prediction/models/ACTP/saved_models/box_only_model_30_11_2021_11_40/ACTP_model.zip"

    data_save_path_1 = "/home/user/Robotics/tactile_prediction/tactile_prediction/models/ACTVP/saved_models/box_only_model_30_11_2021_10_58/"
    data_save_path_2 = "/home/user/Robotics/tactile_prediction/tactile_prediction/models/ACTP/saved_models/box_only_model_30_11_2021_11_40/"

    test_data_dir_1 = "/home/user/Robotics/Data_sets/box_only_dataset/test_image_dataset_10c_10h_32_universal/"
    test_data_dir_2 = "/home/user/Robotics/Data_sets/box_only_dataset/test_linear_dataset_10c_10h_universal/"
    scaler_dir = "/home/user/Robotics/Data_sets/box_only_dataset/scalar_info_universal/"

    number_of_trials = 23  # 0-22
    epochs = 50
    batch_size = 32
    learning_rate = 1e-3
    context_frames = 10
    sequence_length = 20
    train_percentage = 0.9
    validation_percentage = 0.1
    image_size = 32
    criterion = "L1"

    # model = ACTVP()
    model = torch.load(model_path_1)
    model.eval()

    model_name = "ACTVP"
    UMT = UniversalModelTester(model, number_of_trials, test_data_dir_1, image_size, criterion, model_path_1,
                         data_save_path_1, scaler_dir, model_name, epochs, batch_size,
                         learning_rate, context_frames, sequence_length)  # if not an image set image size to 0
    UMT.test_model()


    # epochs = 50
    # batch_size = 32
    # learning_rate = 1e-3
    # context_frames = 10
    # sequence_length = 20
    # train_percentage = 0.9
    # validation_percentage = 0.1
    # image_size = 0
    # criterion = "L1"
    #
    # model = ACTP()
    # model = torch.load(model_path_2)
    # model.eval()
    #
    # model_name = "ACTP"


if __name__ == "__main__":
    main()
