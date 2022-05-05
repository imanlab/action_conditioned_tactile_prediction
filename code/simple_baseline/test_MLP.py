# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import cv2
import copy
import math
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.ticker import(AutoMinorLocator, MultipleLocator)
from matplotlib.animation import FuncAnimation

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from tqdm import tqdm
from pickle import load
from datetime import datetime
from torch.utils.data import Dataset

model_path      = "/home/user/Robotics/tactile_prediction/tactile_prediction/models/simple_baseline/saved_models/box_only_full_model_09_12_2021_15_57/simple_MLP_model"
data_save_path  = "/home/user/Robotics/tactile_prediction/tactile_prediction/models/simple_baseline/saved_models/box_only_full_model_09_12_2021_15_57/"
test_data_dir   = "/home/user/Robotics/Data_sets/box_only_dataset/test_linear_dataset_10c_10h/"
scaler_dir      = "/home/user/Robotics/Data_sets/box_only_dataset/scalar_info/"

train = False

seed = 42
epochs = 100
batch_size = 32
learning_rate = 1e-3
context_frames = 10
sequence_length = 20
ts_to_test = 9  # 0=ts1

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available


class BatchGenerator:
    def __init__(self):
        self.data_map = []
        with open(test_data_dir + 'map.csv', 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                self.data_map.append(row)

    def load_full_data(self):
        dataset_test = FullDataSet(self.data_map)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        self.data_map = []
        return test_loader


class FullDataSet:
    def __init__(self, data_map):
        self.samples = data_map[1:]
        data_map = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        value = self.samples[idx]
        robot_data = np.load(test_data_dir + value[0])
        tactile_data = np.load(test_data_dir + value[1])
        experiment_number = np.load(test_data_dir + value[3])
        time_steps = np.load(test_data_dir + value[4])
        meta = test_data_dir + value[5]
        return [robot_data.astype(np.float32), tactile_data.astype(np.float32), experiment_number, time_steps, meta]


class simple_MLP(nn.Module):
    def __init__(self):
        super(simple_MLP, self).__init__()
        self.fc1 = nn.Linear((48*10), (48*10)).to(device)
        self.tan_activation = nn.Tanh().to(device)

    def forward(self, tactiles, actions):
        tactile = tactiles[:context_frames].permute(1, 0, 2)
        tactile = tactile.flatten(start_dim=1)
        output = self.tan_activation(self.fc1(tactile))
        return output


class ModelTester:
    def __init__(self):
        self.test_full_loader = BG.load_full_data()

        # load model:
        self.full_model = simple_MLP()
        self.full_model = torch.load(model_path)
        self.full_model.eval()

        self.criterion = nn.L1Loss()
        self.optimizer = optim.Adam(self.full_model.parameters(), lr=learning_rate)

        self.load_scalars()

    def test_full_model(self):
        self.objects = []
        self.performance_data = []
        self.prediction_data = []
        self.tg_back_scaled = []
        self.tp1_back_scaled = []
        self.tp5_back_scaled = []
        self.tp10_back_scaled = []
        self.tg = []
        self.tp1 = []
        self.tp5 = []
        self.tp10 = []
        self.current_exp = 0

        for index, batch_features in enumerate (self.test_full_loader):
            action = batch_features[0].squeeze (-1).permute (1, 0, 2).to (device)
            tactile = torch.flatten (batch_features[1], start_dim=2).permute (1, 0, 2).to (device)
            tactile_predictions = self.full_model.forward (tactiles=tactile,
                                                           actions=action)  # Step 3. Run our forward pass.
            tactile_predictions = tactile_predictions.reshape (tactile_predictions.shape[0], 10, 48).permute (1, 0,
                                                                                                              2)

            experiment_number = batch_features[2].permute (1, 0)[context_frames:]
            time_steps = batch_features[3].permute (1, 0)[context_frames:]
            self.meta = batch_features[4][0]

            current_batch = 0
            new_batch = 0
            for index, exp in enumerate (experiment_number.T):
                if exp[0] == self.current_exp:
                    current_batch += 1
                else:
                    new_batch += 1

            for i in [0, 1]:
                if i == 0:
                    tactile_cut = tactile[:, 0:current_batch, :]
                    tactile_predictions_cut = tactile_predictions[:, 0:current_batch]
                    experiment_number_cut = experiment_number[:, 0:current_batch]
                    time_steps_cut = time_steps[:, 0:current_batch]
                if i == 1:
                    tactile_cut = tactile[:, current_batch:, :]
                    tactile_predictions_cut = tactile_predictions[:, current_batch:]
                    experiment_number_cut = experiment_number[:, current_batch:]
                    time_steps_cut = time_steps[:, current_batch:]

                self.prediction_data.append (
                    [tactile_predictions_cut.cpu ().detach (), tactile_cut[context_frames:].cpu ().detach (),
                     experiment_number_cut.cpu ().detach (), time_steps_cut.cpu ().detach ()])

                # convert back to 48 feature tactile readings for plotting:
                gt = np.array (tactile_cut[context_frames - 1].cpu ().detach ())
                p1 = np.array (tactile_predictions_cut[0].cpu ().detach ())
                p5 = np.array (tactile_predictions_cut[4].cpu ().detach ())
                p10 = np.array (tactile_predictions_cut[9].cpu ().detach ())
                descalled_data = []

                # print(gt.shape, p1.shape, p5.shape, p10.shape, gt.shape[0])
                if gt.shape[0] != 0:
                    for data in [gt, p1, p5, p10]:
                        (tx, ty, tz) = np.split (data, 3, axis=1)
                        xela_x_inverse_minmax = self.min_max_scalerx_full_data.inverse_transform (tx)
                        xela_y_inverse_minmax = self.min_max_scalery_full_data.inverse_transform (ty)
                        xela_z_inverse_minmax = self.min_max_scalerz_full_data.inverse_transform (tz)
                        xela_x_inverse_full = self.scaler_tx.inverse_transform (xela_x_inverse_minmax)
                        xela_y_inverse_full = self.scaler_ty.inverse_transform (xela_y_inverse_minmax)
                        xela_z_inverse_full = self.scaler_tz.inverse_transform (xela_z_inverse_minmax)
                        descalled_data.append (
                            np.concatenate ((xela_x_inverse_full, xela_y_inverse_full, xela_z_inverse_full),
                                            axis=1))

                    self.tg_back_scaled.append (descalled_data[0])
                    self.tp1_back_scaled.append (descalled_data[1])
                    self.tp5_back_scaled.append (descalled_data[2])
                    self.tp10_back_scaled.append (descalled_data[3])

                    self.tg.append (gt)
                    self.tp1.append (p1)
                    self.tp5.append (p5)
                    self.tp10.append (p10)

                if i == 0 and new_batch != 0:
                    print ("currently testing trial number: ", str (self.current_exp))

                    # if train:
                    #     self.calc_train_trial_performance()
                    # else:
                    #     self.calc_trial_performance()

                    self.save_predictions (self.current_exp, scaled=True)
                    self.tg_back_scaled = self.tg
                    self.tp1_back_scaled = self.tp1
                    self.tp5_back_scaled = self.tp5
                    self.tp10_back_scaled = self.tp10
                    self.save_predictions (self.current_exp, scaled=False)

                    self.prediction_data = []
                    self.tg_back_scaled = []
                    self.tp1_back_scaled = []
                    self.tp5_back_scaled = []
                    self.tp10_back_scaled = []
                    self.tg = []
                    self.tp1 = []
                    self.tp5 = []
                    self.tp10 = []

                    self.current_exp += 1
                if i == 0 and new_batch == 0:
                    break

        print ("Hello :D ")

        # if train:
        #     self.calc_train_performance()
        # else:
        #     self.calc_test_performance()

    def save_predictions(self, experiment_to_test, scaled):
        '''
        - Plot the descaled 48 feature tactile vector for qualitative analysis
        - Save plots in a folder with name being the trial number.
        '''
        trial_groundtruth_data = []
        trial_predicted_data_t1 = []
        trial_predicted_data_t5 = []
        trial_predicted_data_t10 = []
        for index in range (len(self.tg_back_scaled)):
            for batch_number in range (len (self.tp10_back_scaled[index])):
                if experiment_to_test == self.prediction_data[index][2].T[batch_number][0]:
                    trial_predicted_data_t1.append (self.tp1_back_scaled[index][batch_number])
                    trial_predicted_data_t5.append (self.tp5_back_scaled[index][batch_number])
                    trial_predicted_data_t10.append (self.tp10_back_scaled[index][batch_number])
                    trial_groundtruth_data.append (self.tg_back_scaled[index][batch_number])

        if scaled:
            add = "SCALED_"
        else:
            add = "DESCALED_"

        if train:
            tadd = "train_"
        else:
            tadd = "test_"

        plot_save_dir = data_save_path + tadd + add + "test_plots_" + str(experiment_to_test)
        try:
            os.mkdir(plot_save_dir)
        except:
            "directory already exists"

        np.save(plot_save_dir + '/trial_groundtruth_data', np.array(trial_groundtruth_data))
        np.save(plot_save_dir + '/prediction_data_t1', np.array(trial_predicted_data_t1))
        np.save(plot_save_dir + '/prediction_data_t5', np.array(trial_predicted_data_t5))
        np.save(plot_save_dir + '/prediction_data_t10', np.array(trial_predicted_data_t10))

        meta_data_file_name = str(np.load(self.meta)[0]) + "/meta_data.csv"
        meta_data = []
        with open(meta_data_file_name, 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                meta_data.append(row)
        np.save(plot_save_dir + '/meta_data', np.array(meta_data))
        np.save(plot_save_dir + '/meta_name', np.array(meta_data_file_name))

    def calc_trial_performance(self):
        mae_loss, mae_loss_1, mae_loss_5, mae_loss_10, mae_loss_x, mae_loss_y, mae_loss_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        meta_data_file_name = str (np.load (self.meta)[0]) + "/meta_data.csv"
        meta_data = []
        with open (meta_data_file_name, 'r') as f:  # rb
            reader = csv.reader (f)
            for row in reader:
                meta_data.append (row)
        seen = meta_data[1][2]
        object = meta_data[1][0]

        index = 0
        index_ssim = 0
        with torch.no_grad ():
            for batch_set in self.prediction_data:
                index += 1

                mae_loss_check = self.criterion (batch_set[0], batch_set[1]).item ()
                if math.isnan (mae_loss_check):
                    index -= 1
                else:
                    ## MAE:
                    mae_loss += mae_loss_check
                    mae_loss_1 += self.criterion (batch_set[0][0], batch_set[1][0]).item ()
                    mae_loss_5 += self.criterion (batch_set[0][4], batch_set[1][4]).item ()
                    mae_loss_10 += self.criterion (batch_set[0][9], batch_set[1][9]).item ()
                    mae_loss_x += self.criterion (batch_set[0][:, :, 0:16], batch_set[1][:, :, 0:16]).item ()
                    mae_loss_y += self.criterion (batch_set[0][:, :, 16:32], batch_set[1][:, :, 16:32]).item ()
                    mae_loss_z += self.criterion (batch_set[0][:, :, 32:], batch_set[1][:, :, 32:]).item ()

        self.performance_data.append (
            [mae_loss / index, mae_loss_1 / index, mae_loss_5 / index, mae_loss_10 / index,
             mae_loss_x / index, mae_loss_y / index, mae_loss_z / index, seen, object])

        print ("object: ", object)
        self.objects.append (object)

    def calc_test_performance(self):
        '''
        - Calculates PSNR, SSIM, MAE for ts1, 5, 10 and x,y,z forces
        - Save Plots for qualitative analysis
        - Slip classification test
        '''
        performance_data_full = []
        performance_data_full.append (
            ["test loss MAE(L1): ", (sum ([i[0] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["test loss MAE(L1) pred ts 1: ",
                                       (sum ([i[1] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["test loss MAE(L1) pred ts 5: ",
                                       (sum ([i[2] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["test loss MAE(L1) pred ts 10: ",
                                       (sum ([i[3] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["test loss MAE(L1) pred force x: ",
                                       (sum ([i[4] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["test loss MAE(L1) pred force y: ",
                                       (sum ([i[5] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["test loss MAE(L1) pred force z: ",
                                       (sum ([i[6] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["test loss MAE(L1) Seen objects: ", (
                    sum ([i[0] for i in self.performance_data if i[-2] == '1']) / len (
                [i[0] for i in self.performance_data if i[-2] == '1']))])
        performance_data_full.append (["test loss MAE(L1) Novel objects: ", (
                    sum ([i[0] for i in self.performance_data if i[-2] == '0']) / len (
                [i[0] for i in self.performance_data if i[-2] == '0']))])

        self.objects = list (set (self.objects))
        for object in self.objects:
            performance_data_full.append (["test loss MAE(L1) Trained object " + str (object) + ": ", (
                        sum ([i[0] for i in self.performance_data if i[-1] == str (object)]) / len (
                    [i[0] for i in self.performance_data if i[-1] == str (object)]))])

        [print (i) for i in performance_data_full]
        np.save (data_save_path + 'model_performance_loss_data', np.asarray (performance_data_full))

    def calc_train_performance(self):
        '''
        - Calculates PSNR, SSIM, MAE for ts1, 5, 10 and x,y,z forces
        - Save Plots for qualitative analysis
        - Slip classification test
        '''
        performance_data_full = []
        performance_data_full.append (
            ["train loss MAE(L1): ", (sum ([i[0] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["train loss MAE(L1) pred ts 1: ",
                                       (sum ([i[1] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["train loss MAE(L1) pred ts 5: ",
                                       (sum ([i[2] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["train loss MAE(L1) pred ts 10: ",
                                       (sum ([i[3] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["train loss MAE(L1) pred force x: ",
                                       (sum ([i[4] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["train loss MAE(L1) pred force y: ",
                                       (sum ([i[5] for i in self.performance_data]) / len (self.performance_data))])
        performance_data_full.append (["train loss MAE(L1) pred force z: ",
                                       (sum ([i[6] for i in self.performance_data]) / len (self.performance_data))])

        self.objects = list (set (self.objects))
        for object in self.objects:
            performance_data_full.append (["train loss MAE(L1) Trained object " + str (object) + ": ", (
                        sum ([i[0] for i in self.performance_data if i[-1] == str (object)]) / len (
                    [i[0] for i in self.performance_data if i[-1] == str (object)]))])

        [print (i) for i in performance_data_full]
        np.save (data_save_path + 'TRAIN_model_performance_loss_data', np.asarray (performance_data_full))

    def load_scalars(self):
        self.scaler_tx = load (open (scaler_dir + "tactile_standard_scaler_x.pkl", 'rb'))
        self.scaler_ty = load (open (scaler_dir + "tactile_standard_scaler_y.pkl", 'rb'))
        self.scaler_tz = load (open (scaler_dir + "tactile_standard_scaler_z.pkl", 'rb'))
        self.min_max_scalerx_full_data = load (open (scaler_dir + "tactile_min_max_scalar_x.pkl", 'rb'))
        self.min_max_scalery_full_data = load (open (scaler_dir + "tactile_min_max_scalar_y.pkl", 'rb'))
        self.min_max_scalerz_full_data = load (open (scaler_dir + "tactile_min_max_scalar.pkl", 'rb'))

if __name__ == "__main__":
    BG = BatchGenerator ()
    MT = ModelTester ()
    MT.test_full_model()