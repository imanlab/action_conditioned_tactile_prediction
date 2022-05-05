# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import cv2
import copy
import math
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from matplotlib.animation import FuncAnimation

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from tqdm import tqdm
from pickle import load
from datetime import datetime
from torch.utils.data import Dataset

import os
import csv
import copy
import utils
import numpy as np

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

model_path      = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/SVG/saved_models/box_only_4layers_WORKING_model_14_02_2022_11_00/SVG_model"
data_save_path  = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/SVG/saved_models/box_only_4layers_WORKING_model_14_02_2022_11_00/"
test_data_dir   = "/home/user/Robotics/Data_sets/box_only_dataset/test_image_dataset_10c_10h_32_universal/"
scaler_dir      = "/home/user/Robotics/Data_sets/box_only_dataset/scalar_info_universal/"

context_frames = 10
sequence_length = 20

(lr, beta1, batch_size, log_dir, model_dir, name, data_root, optimizer, niter, seed, epoch_size,
image_width, channels, out_channels, dataset, n_past, n_future, n_eval, rnn_size, prior_rnn_layers,
posterior_rnn_layers, predictor_rnn_layers, z_dim, g_dim, beta, data_threads, num_digits,
last_frame_skip, epochs, _, __) = torch.load(model_path)["features"]

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#  use gpu if available


class BatchGenerator:
    def __init__(self, test_data_dir, batch_size, image_size, trial_number):
        self.batch_size = batch_size
        self.image_size = image_size
        self.trial_number = trial_number
        self.test_data_dir = test_data_dir + 'test_trial_' + str(self.trial_number) + '/'
        self.data_map = []
        print(self.test_data_dir + 'map_' + str(self.trial_number) + '.csv')
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
            experiment_number = np.load(self.test_data_dir + value[2])
            time_steps = np.load(self.test_data_dir + value[3])
        else:
            tactile_data = []
            for image_name in np.load(self.test_data_dir + value[2]):
                tactile_data.append(np.load(self.test_data_dir + image_name))
            tactile_data = np.array(tactile_data)
            experiment_number = np.load(self.test_data_dir + value[3])
            time_steps = np.load(self.test_data_dir + value[4])

        return [robot_data.astype(np.float32), tactile_data.astype(np.float32), experiment_number, time_steps]


class ModelTester:
    def __init__(self):
        self.full_model = torch.load(model_path)

        (lr, beta1, batch_size, log_dir, model_dir, name, data_root, optimizer, niter, seed, epoch_size,
        image_width, channels, out_channels, dataset, n_past, n_future, n_eval, rnn_size, prior_rnn_layers,
        posterior_rnn_layers, predictor_rnn_layers, z_dim, g_dim, beta, data_threads, num_digits,
        last_frame_skip, epochs, _, __) = self.full_model["features"]

        self.test_full_loader = BG.load_full_data()

        self.optimizer = optim.Adam
        self.frame_predictor = self.full_model["frame_predictor"]
        self.posterior = self.full_model["posterior"]
        self.prior = self.full_model["prior"]
        self.encoder = self.full_model["encoder"]
        self.decoder = self.full_model["decoder"]

        self.mae_criterion = nn.L1Loss()

        self.frame_predictor.cuda()
        self.posterior.cuda()
        self.prior.cuda()
        self.encoder.cuda()
        self.decoder.cuda()
        self.mae_criterion.cuda()

        self.criterion = nn.L1Loss()
        self.load_scalars()

    def SVG_pass_through(self, scene, actions, test=False):
        # if not batch size == add a bunch of zeros and then remove them from the predictions:
        cut_point = self.batch_size
        if scene.shape[1] != self.batch_size:
            cut_point = scene.shape[1]
            scene = torch.cat((scene, torch.zeros([20, (self.batch_size - scene.shape[1]), 3, self.image_size,self.image_size]).to(device)), axis=1)
            actions = torch.cat((actions, torch.zeros([20, (self.batch_size - actions.shape[1]), 6]).to(device)), axis=1)

        mae, kld = 0, 0
        outputs = []

        self.frame_predictor.zero_grad ()
        self.posterior.zero_grad ()
        self.prior.zero_grad ()
        self.encoder.zero_grad ()
        self.decoder.zero_grad ()

        self.frame_predictor.hidden = self.frame_predictor.init_hidden ()
        self.posterior.hidden = self.posterior.init_hidden ()
        self.prior.hidden = self.prior.init_hidden ()

        state = actions[0].to (device)
        for index, (sample_sscene, sample_action) in enumerate (zip (scene[:-1], actions[1:])):
            state_action = torch.cat ((state, actions[index]), 1)

            if index > self.n_past - 1:  # horizon
                h, skip = self.encoder (x_pred)
                h_target = self.encoder (scene[index + 1])[0]

                if test:
                    _, mu, logvar = self.posterior (h_target)  # learned prior
                    z_t, mu_p, logvar_p = self.prior (h)  # learned prior
                else:
                    z_t, mu, logvar = self.posterior (h_target)  # learned prior
                    _, mu_p, logvar_p = self.prior (h)  # learned prior

                h_pred = self.frame_predictor (torch.cat ([h, z_t, state_action], 1))  # prediction model
                x_pred = self.decoder ([h_pred, skip])  # prediction model

                mae += self.criterion(x_pred, scene[index + 1])  # prediction model
                kld += self.kl_criterion(mu, logvar, mu_p, logvar_p)  # learned prior

                outputs.append (x_pred)
            else:  # context
                h, skip = self.encoder (scene[index])
                h_target = self.encoder (scene[index + 1])[0]

                if test:
                    _, mu, logvar = self.posterior (h_target)  # learned prior
                    z_t, mu_p, logvar_p = self.prior (h)  # learned prior
                else:
                    z_t, mu, logvar = self.posterior (h_target)  # learned prior
                    _, mu_p, logvar_p = self.prior (h)  # learned prior

                h_pred = self.frame_predictor (torch.cat ([h, z_t, state_action], 1))  # prediction model
                x_pred = self.decoder ([h_pred, skip])  # prediction model

                mae += self.criterion(x_pred, scene[index + 1])  # prediction model
                kld += self.kl_criterion(mu, logvar, mu_p, logvar_p)  # learned prior

                last_output = x_pred

        outputs = [last_output] + outputs

        return mae.data.cpu ().numpy () / (self.n_past + self.n_future), kld.data.cpu ().numpy () / (self.n_future + self.n_past), torch.stack(outputs)[:, :cut_point, :,:,:]

    def test_full_model(self):
        self.objects = []
        self.performance_data = []
        self.prediction_data  = []
        self.tg_back_scaled   = []
        self.tp1_back_scaled = []
        self.tp5_back_scaled = []
        self.tp10_back_scaled = []
        self.current_exp      = 0

        for index, batch_features in enumerate(self.test_full_loader):
            if batch_features[1].shape[0] == batch_size:
                tactile = batch_features[1].permute (1, 0, 4, 3, 2).to(device)
                action = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
                _, __, tactile_predictions = self.SVG_pass_through(scene=tactile, actions=action)

                experiment_number = batch_features[2].permute(1, 0)[context_frames:]
                time_steps = batch_features[3].permute(1, 0)[context_frames:]
                self.meta = batch_features[4][0]

                current_batch = 0
                new_batch = 0
                for index, exp in enumerate(experiment_number.T):
                    if exp[0] == self.current_exp:
                        current_batch += 1
                    else:
                        new_batch += 1

                for i in [0,1]:
                    if i == 0:
                        tactile_cut             = tactile[:, 0:current_batch, :, :, :]
                        tactile_predictions_cut = tactile_predictions[:, 0:current_batch, :, :, :]
                        experiment_number_cut   = experiment_number[:, 0:current_batch]
                        time_steps_cut          = time_steps[:, 0:current_batch]
                    if i == 1:
                        tactile_cut             = tactile[:, current_batch:, :, :, :]
                        tactile_predictions_cut = tactile_predictions[:, current_batch:, :, :, :]
                        experiment_number_cut   = experiment_number[:, current_batch:]
                        time_steps_cut          = time_steps[:, current_batch:]

                    self.prediction_data.append([tactile_predictions_cut.cpu().detach(), tactile_cut[context_frames:].cpu().detach(),
                                                 experiment_number_cut.cpu().detach(), time_steps_cut.cpu().detach()])

                    # convert back to 48 feature tactile readings for plotting:
                    gt = []
                    p1 = []
                    p5 = []
                    p10 = []
                    for batch_value in range(tactile_predictions_cut.shape[1]):
                        gt.append(cv2.resize(tactile_cut[context_frames-1][batch_value].permute(1, 2, 0).cpu().detach().numpy(), dsize=(4, 4), interpolation=cv2.INTER_CUBIC).flatten())
                        p1.append(cv2.resize(tactile_predictions_cut[0][batch_value].permute(1, 2, 0).cpu().detach().numpy(), dsize=(4, 4), interpolation=cv2.INTER_CUBIC).flatten())
                        p5.append(cv2.resize(tactile_predictions_cut[4][batch_value].permute(1, 2, 0).cpu().detach().numpy(), dsize=(4, 4), interpolation=cv2.INTER_CUBIC).flatten())
                        p10.append(cv2.resize(tactile_predictions_cut[9][batch_value].permute(1, 2, 0).cpu().detach().numpy(), dsize=(4, 4), interpolation=cv2.INTER_CUBIC).flatten())

                    gt = np.array(gt)
                    p1 = np.array(p1)
                    p5 = np.array(p5)
                    p10 = np.array(p10)
                    descalled_data = []

                    # print(gt.shape, p1.shape, p5.shape, p10.shape, gt.shape[0])
                    if gt.shape[0] != 0:
                        for data in [gt, p1, p5, p10]:
                            (tx, ty, tz) = np.split(data, 3, axis=1)
                            xela_x_inverse_minmax = self.min_max_scalerx_full_data.inverse_transform(tx)
                            xela_y_inverse_minmax = self.min_max_scalery_full_data.inverse_transform(ty)
                            xela_z_inverse_minmax = self.min_max_scalerz_full_data.inverse_transform(tz)
                            xela_x_inverse_full = self.scaler_tx.inverse_transform(xela_x_inverse_minmax)
                            xela_y_inverse_full = self.scaler_ty.inverse_transform(xela_y_inverse_minmax)
                            xela_z_inverse_full = self.scaler_tz.inverse_transform(xela_z_inverse_minmax)
                            descalled_data.append(np.concatenate((xela_x_inverse_full, xela_y_inverse_full, xela_z_inverse_full), axis=1))

                        self.tg_back_scaled.append(descalled_data[0])
                        self.tp1_back_scaled.append(descalled_data[1])
                        self.tp5_back_scaled.append(descalled_data[2])
                        self.tp10_back_scaled.append(descalled_data[3])

                    if i == 0 and new_batch != 0:
                        print("currently testing trial number: ", str(self.current_exp))
                        # self.calc_train_trial_performance()
                        self.calc_trial_performance()
                        # self.save_predictions(self.current_exp)
                        # self.create_test_plots(self.current_exp)
                        # self.create_difference_gifs(self.current_exp)
                        self.prediction_data = []
                        self.tg_back_scaled = []
                        self.tp1_back_scaled = []
                        self.tp5_back_scaled = []
                        self.tp10_back_scaled = []
                        self.current_exp += 1
                    if i== 0 and new_batch == 0:
                        break

        print("Hello :D ")

        self.calc_test_performance()
        # self.calc_train_performance()

    def calc_train_trial_performance(self):
        mae_loss, mae_loss_1, mae_loss_5, mae_loss_10, mae_loss_x, mae_loss_y, mae_loss_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        meta_data_file_name = str(np.load(self.meta)[0]) + "/meta_data.csv"
        meta_data = []
        with open(meta_data_file_name, 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                meta_data.append(row)
        seen = meta_data[1][2]
        object = meta_data[1][0]

        index = 0
        with torch.no_grad():
            for batch_set in self.prediction_data:
                index += 1

                mae_loss_check = self.criterion (batch_set[0], batch_set[1]).item ()
                if math.isnan(mae_loss_check):
                    index -= 1
                else:
                    ## MAE:
                    mae_loss    += mae_loss_check
                    mae_loss_1  += self.criterion(batch_set[0][0], batch_set[1][0]).item()
                    mae_loss_5  += self.criterion(batch_set[0][4], batch_set[1][4]).item()
                    mae_loss_10 += self.criterion(batch_set[0][9], batch_set[1][9]).item()
                    mae_loss_x  += self.criterion(batch_set[0][:,:,0], batch_set[1][:,:,0]).item()
                    mae_loss_y  += self.criterion(batch_set[0][:,:,1], batch_set[1][:,:,1]).item()
                    mae_loss_z  += self.criterion(batch_set[0][:,:,2], batch_set[1][:,:,2]).item()

        self.performance_data.append([mae_loss/index, mae_loss_1/index, mae_loss_5/index, mae_loss_10/index, mae_loss_x/index,
                                mae_loss_y/index, mae_loss_z/index, seen, object])

        print("object: ", object)
        self.objects.append(object)

    def calc_train_performance(self):
        '''
        - Calculates PSNR, SSIM, MAE for ts1, 5, 10 and x,y,z forces
        - Save Plots for qualitative analysis
        - Slip classification test
        '''
        performance_data_full = []
        performance_data_full.append(["test loss MAE(L1): ", (sum([i[0] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 1: ", (sum([i[1] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 5: ", (sum([i[2] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 10: ", (sum([i[3] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred force x: ", (sum([i[4] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred force y: ", (sum([i[5] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred force z: ", (sum([i[6] for i in self.performance_data]) / len(self.performance_data))])

        self.objects = list(set(self.objects))
        for object in self.objects:
            performance_data_full.append(["test loss MAE(L1) Trained object " + str(object) + ": ", (sum([i[0] for i in self.performance_data if i[-1] == str(object)]) / len([i[0] for i in self.performance_data if i[-1] == str(object)]))])

        [print (i) for i in performance_data_full]
        np.save(data_save_path + 'TRAIN_model_performance_loss_data', np.asarray(performance_data_full))

    def save_predictions(self, experiment_to_test):
        '''
        - Plot the descaled 48 feature tactile vector for qualitative analysis
        - Save plots in a folder with name being the trial number.
        '''
        trial_groundtruth_data = []
        trial_predicted_data_t1 = []
        trial_predicted_data_t5 = []
        trial_predicted_data_t10 = []
        for index in range(len(self.tg_back_scaled)):
            for batch_number in range(len(self.tp10_back_scaled[index])):
                if experiment_to_test == self.prediction_data[index][2].T[batch_number][0]:
                    trial_predicted_data_t1.append(self.tp1_back_scaled[index][batch_number])
                    trial_predicted_data_t5.append(self.tp5_back_scaled[index][batch_number])
                    trial_predicted_data_t10.append(self.tp10_back_scaled[index][batch_number])
                    trial_groundtruth_data.append(self.tg_back_scaled[index][batch_number])

        plot_save_dir = data_save_path + "test_plots_" + str(experiment_to_test)
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

    def calc_trial_performance(self):
        mae_loss, mae_loss_1, mae_loss_5, mae_loss_10, mae_loss_x, mae_loss_y, mae_loss_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ssim_loss, ssim_loss_1, ssim_loss_5, ssim_loss_10, ssim_loss_x, ssim_loss_y, ssim_loss_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        psnr_loss, psnr_loss_1, psnr_loss_5, psnr_loss_10, psnr_loss_x, psnr_loss_y, psnr_loss_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        ssim_calc = SSIM(window_size=32)
        psnr_calc = PSNR()

        meta_data_file_name = str(np.load(self.meta)[0]) + "/meta_data.csv"
        meta_data = []
        with open(meta_data_file_name, 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                meta_data.append(row)
        seen = meta_data[1][2]
        object = meta_data[1][0]

        index = 0
        index_ssim = 0
        with torch.no_grad():
            for batch_set in self.prediction_data:
                index += 1

                mae_loss_check = self.criterion (batch_set[0], batch_set[1]).item ()
                if math.isnan(mae_loss_check):
                    index -= 1
                else:
                    ## MAE:
                    mae_loss    += mae_loss_check
                    mae_loss_1  += self.criterion(batch_set[0][0], batch_set[1][0]).item()
                    mae_loss_5  += self.criterion(batch_set[0][4], batch_set[1][4]).item()
                    mae_loss_10 += self.criterion(batch_set[0][9], batch_set[1][9]).item()
                    mae_loss_x  += self.criterion(batch_set[0][:,:,0], batch_set[1][:,:,0]).item()
                    mae_loss_y  += self.criterion(batch_set[0][:,:,1], batch_set[1][:,:,1]).item()
                    mae_loss_z  += self.criterion(batch_set[0][:,:,2], batch_set[1][:,:,2]).item()
                    ## SSIM:
                    for i in range(len(batch_set)):
                        index_ssim += 1
                        ssim_loss += ssim_calc(batch_set[0][i], batch_set[1][i])
                    ssim_loss_1  += ssim_calc(batch_set[0][0], batch_set[1][0])
                    ssim_loss_5  += ssim_calc(batch_set[0][4], batch_set[1][4])
                    ssim_loss_10 += ssim_calc(batch_set[0][9], batch_set[1][9])
                    ssim_loss_x  += ssim_calc(batch_set[0][:,:,0], batch_set[1][:,:,0])
                    ssim_loss_y  += ssim_calc(batch_set[0][:,:,1], batch_set[1][:,:,1])
                    ssim_loss_z  += ssim_calc(batch_set[0][:,:,2], batch_set[1][:,:,2])
                    ## PSNR:
                    psnr_loss    += psnr_calc(batch_set[0], batch_set[1])
                    psnr_loss_1  += psnr_calc(batch_set[0][0], batch_set[1][0])
                    psnr_loss_5  += psnr_calc(batch_set[0][4], batch_set[1][4])
                    psnr_loss_10 += psnr_calc(batch_set[0][9], batch_set[1][9])
                    psnr_loss_x  += psnr_calc(batch_set[0][:,:,0], batch_set[1][:,:,0])
                    psnr_loss_y  += psnr_calc(batch_set[0][:,:,1], batch_set[1][:,:,1])
                    psnr_loss_z  += psnr_calc(batch_set[0][:,:,2], batch_set[1][:,:,2])

        self.performance_data.append([mae_loss/index, mae_loss_1/index, mae_loss_5/index, mae_loss_10/index, mae_loss_x/index,
                                mae_loss_y/index, mae_loss_z/index, ssim_loss/index_ssim, ssim_loss_1/index,
                                ssim_loss_5/index, ssim_loss_10/index, ssim_loss_x/index, ssim_loss_y/index,
                                ssim_loss_z/index, psnr_loss/index, psnr_loss_1/index, psnr_loss_5/index,
                                psnr_loss_10/index, psnr_loss_x/index, psnr_loss_y/index, psnr_loss_z/index, seen, object])


        print("object: ", object)
        self.objects.append(object)

    def calc_test_performance(self):
        '''
        - Calculates PSNR, SSIM, MAE for ts1, 5, 10 and x,y,z forces
        - Save Plots for qualitative analysis
        - Slip classification test
        '''
        performance_data_full = []
        performance_data_full.append(["test loss MAE(L1): ", (sum([i[0] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 1: ", (sum([i[1] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 5: ", (sum([i[2] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 10: ", (sum([i[3] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred force x: ", (sum([i[4] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred force y: ", (sum([i[5] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred force z: ", (sum([i[6] for i in self.performance_data]) / len(self.performance_data))])

        performance_data_full.append (["test loss MAE(L1) Seen objects: ", (sum([i[0] for i in self.performance_data if i[-2] == '1']) / len([i[0] for i in self.performance_data if i[-2] == '1']))])
        performance_data_full.append (["test loss MAE(L1) Novel objects: ", (sum([i[0] for i in self.performance_data if i[-2] == '0']) / len([i[0] for i in self.performance_data if i[-2] == '0']))])

        self.objects = list(set(self.objects))
        for object in self.objects:
            performance_data_full.append(["test loss MAE(L1) Trained object " + str(object) + ": ", (sum([i[0] for i in self.performance_data if i[-1] == str(object)]) / len([i[0] for i in self.performance_data if i[-1] == str(object)]))])


        performance_data_full.append(["test loss SSIM: ", (sum([i[7] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss SSIM pred ts 1: ", (sum([i[8] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss SSIM pred ts 5: ", (sum([i[9] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss SSIM pred ts 10: ", (sum([i[10] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss SSIM pred force x: ", (sum([i[11] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss SSIM pred force y: ", (sum([i[12] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss SSIM pred force z: ", (sum([i[13] for i in self.performance_data]) / len(self.performance_data))])

        performance_data_full.append (["test loss SSIM Seen objects: ", (sum([i[7] for i in self.performance_data if i[-2] == '1']) / len([i[7] for i in self.performance_data if i[-2] == '1']))])
        performance_data_full.append (["test loss SSIM Novel objects: ", (sum([i[7] for i in self.performance_data if i[-2] == '0']) / len([i[7] for i in self.performance_data if i[-2] == '0']))])

        performance_data_full.append(["test loss PSNR: ", (sum([i[14] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss PSNR pred ts 1: ", (sum([i[15] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss PSNR pred ts 5: ", (sum([i[16] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss PSNR pred ts 10: ", (sum([i[17] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss PSNR pred force x: ", (sum([i[18] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss PSNR pred force y: ", (sum([i[19] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss PSNR pred force z: ", (sum([i[20] for i in self.performance_data]) / len(self.performance_data))])

        performance_data_full.append (["test loss PSNR Seen objects: ", (sum([i[14] for i in self.performance_data if i[-2] == '1']) / len([i[14] for i in self.performance_data if i[-2] == '1']))])
        performance_data_full.append (["test loss PSNR Novel objects: ", (sum([i[14] for i in self.performance_data if i[-2] == '0']) / len([i[14] for i in self.performance_data if i[-2] == '0']))])

        [print (i) for i in performance_data_full]
        np.save(data_save_path + 'model_performance_loss_data', np.asarray(performance_data_full))

    def load_scalars(self):
        self.scaler_tx = load(open(scaler_dir + "tactile_standard_scaler_x.pkl", 'rb'))
        self.scaler_ty = load(open(scaler_dir + "tactile_standard_scaler_y.pkl", 'rb'))
        self.scaler_tz = load(open(scaler_dir + "tactile_standard_scaler_z.pkl", 'rb'))
        self.min_max_scalerx_full_data = load(open(scaler_dir + "tactile_min_max_scalar_x.pkl", 'rb'))
        self.min_max_scalery_full_data = load(open(scaler_dir + "tactile_min_max_scalar_y.pkl", 'rb'))
        self.min_max_scalerz_full_data = load(open(scaler_dir + "tactile_min_max_scalar.pkl", 'rb'))


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze (0).unsqueeze (0)
    window = Variable (_2D_window.expand (channel, 1, window_size, window_size).contiguous ())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d (img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d (img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow (2)
    mu2_sq = mu2.pow (2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d (img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d (img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d (img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean ()
    else:
        return ssim_map.mean (1).mean (1).mean (1)


class SSIM (torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super (SSIM, self).__init__ ()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window (window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size ()

        if channel == self.channel and self.window.data.type () == img1.data.type ():
            window = self.window
        else:
            window = create_window (self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda (img1.get_device ())
            window = window.type_as (img1)

            self.window = window
            self.channel = channel

        return _ssim (img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size ()
    window = create_window (window_size, channel)

    if img1.is_cuda:
        window = window.cuda (img1.get_device ())
    window = window.type_as (img1)

    return _ssim (img1, img2, window, window_size, channel, size_average)

if __name__ == "__main__":
    BG = BatchGenerator()
    MT = ModelTester()
    MT.test_full_model()

