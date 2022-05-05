# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import cv2
import math
import numpy as np

from tqdm import tqdm
from pickle import load
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torchvision

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

        meta = self.test_data_dir + value[5]

        return [robot_data.astype(np.float32), tactile_data.astype(np.float32), experiment_number, time_steps, meta]

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



class UniversalModelTester:
    def __init__(self, model, number_of_trials, test_data_dir, image_size, criterion, model_path, data_save_path,
                 scaler_dir, model_name, batch_size, context_frames, sequence_length, latent_channels):
        self.number_of_trials = number_of_trials
        self.batch_size = batch_size
        self.context_frames = context_frames
        self.sequence_length = sequence_length
        self.model_name = model_name
        self.model_save_path = model_path
        self.data_save_path = data_save_path
        self.test_data_dir = test_data_dir
        self.scaler_dir = scaler_dir
        self.model = model
        self.image_size = image_size
        self.latent_channels = latent_channels
        self.objects = []
        self.performance_data = []
        if criterion == "L1":
            self.criterion = nn.L1Loss()
        if criterion == "L2":
            self.criterion = nn.MSELoss()
        self.load_scalars()

        if self.model_name == "SVG":
            (self.lr, self.beta1, self.batch_size, self.log_dir, self.model_dir, self.name, self.data_root, self.optimizer, self.niter, self.seed, self.epoch_size,
             self.image_width, self.channels, self.out_channels, self.dataset, self.n_past, self.n_future, self.n_eval, self.rnn_size, self.prior_rnn_layers,
             self.posterior_rnn_layers, self.predictor_rnn_layers, self.z_dim, self.g_dim, self.beta, self.data_threads, self.num_digits,
             self.last_frame_skip, self.epochs, _, __) = self.model["features"]
            self.frame_predictor = self.model["frame_predictor"].to(device)
            self.posterior = self.model["posterior"].to(device)
            self.prior = self.model["prior"].to(device)
            self.encoder = self.model["encoder"].to(device)
            self.decoder = self.model["decoder"].to(device)

    def test_model(self, save_trial=False):
        for trial in self.number_of_trials:
            print(trial)
            BG = BatchGenerator(self.test_data_dir, self.batch_size, self.image_size, trial)
            self.test_full_loader = BG.load_full_data()


            for index, batch_features in tqdm(enumerate(self.test_full_loader)):
                tactile_predictions, tactile_groundtruth, loss = self.run_batch(batch_features[1], batch_features[0])
                self.meta = batch_features[-1]

                tactile_predictions = tactile_predictions.cpu().detach()
                tactile_groundtruth = tactile_groundtruth.cpu().detach()

                pt_descalled_data = []
                gt_descalled_data = []

                for batch_index in range(tactile_predictions.shape[1]):
                    sequence_p = []
                    sequence_g = []
                    for ps in range(tactile_predictions.shape[0]):
                        sequence_p.append(cv2.resize(torch.tensor(tactile_predictions)[ps][batch_index].permute(1, 2, 0).numpy(), dsize=(4, 4), interpolation=cv2.INTER_CUBIC).flatten())
                        sequence_g.append(cv2.resize(torch.tensor(tactile_groundtruth)[ps][batch_index].permute(1, 2, 0).numpy(), dsize=(4, 4), interpolation=cv2.INTER_CUBIC).flatten())
                    pt_descalled_data.append(sequence_p)
                    gt_descalled_data.append(sequence_g)

                if index == 0:
                    if self.image_size == 0:
                        prediction_data = np.array(tactile_predictions.permute(1, 0, 2).cpu().detach())
                        groundtruth_data = np.array(tactile_groundtruth.permute(1, 0, 2).cpu().detach())
                    else:
                        prediction_data = np.array(pt_descalled_data)
                        groundtruth_data = np.array(gt_descalled_data)
                        # prediction_data = np.array(tactile_predictions.permute(1, 0, 2, 3, 4).cpu().detach())
                        # groundtruth_data = np.array (tactile_groundtruth.permute(1, 0, 2, 3, 4).cpu().detach())
                else:
                    if self.image_size == 0:
                        prediction_data = np.concatenate((prediction_data, np.array(tactile_predictions.permute(1, 0, 2).cpu().detach())), axis=0)
                        groundtruth_data = np.concatenate((groundtruth_data, np.array(tactile_groundtruth.permute(1, 0, 2).cpu().detach())), axis=0)
                    else:
                        prediction_data = np.concatenate((prediction_data, np.array(pt_descalled_data)), axis=0)
                        groundtruth_data = np.concatenate((groundtruth_data, np.array(gt_descalled_data)), axis=0)
                        # prediction_data = np.concatenate((prediction_data, np.array(tactile_predictions.permute(1, 0, 2, 3, 4).cpu().detach())), axis=0)
                        # groundtruth_data = np.concatenate((groundtruth_data, np.array(tactile_groundtruth.permute(1, 0, 2, 3, 4).cpu().detach())), axis=0)

            # self.calc_losses(torch.tensor(np.array(prediction_data)), torch.tensor(np.array(groundtruth_data)), trial_number=trial)
            if save_trial:
                # prediction_data_descaled, groundtruth_data_descaled = self.scale_back(prediction_data, groundtruth_data)
                self.save_trial(np.array(prediction_data), np.array(groundtruth_data), trial_number=trial)
        # self.save_performance()

    def scale_back(self, tactile_data, groundtruth_data):
        pt_descalled_data = []
        gt_descalled_data = []
        if self.image_size == 0:
            (ptx, pty, ptz) = np.split(tactile_data, 3, axis=2)
            (gtx, gty, gtz) = np.split(groundtruth_data, 3, axis=2)
            for time_step in range(tactile_data.shape[0]):
                xela_ptx_inverse_minmax = self.min_max_scalerx_full_data.inverse_transform(ptx[time_step])
                xela_pty_inverse_minmax = self.min_max_scalery_full_data.inverse_transform(pty[time_step])
                xela_ptz_inverse_minmax = self.min_max_scalerz_full_data.inverse_transform(ptz[time_step])
                xela_ptx_inverse_full = self.scaler_tx.inverse_transform(xela_ptx_inverse_minmax)
                xela_pty_inverse_full = self.scaler_ty.inverse_transform(xela_pty_inverse_minmax)
                xela_ptz_inverse_full = self.scaler_tz.inverse_transform(xela_ptz_inverse_minmax)
                pt_descalled_data.append(np.concatenate((xela_ptx_inverse_full, xela_pty_inverse_full, xela_ptz_inverse_full), axis=1))

                xela_gtx_inverse_minmax = self.min_max_scalerx_full_data.inverse_transform(gtx[time_step])
                xela_gty_inverse_minmax = self.min_max_scalery_full_data.inverse_transform(gty[time_step])
                xela_gtz_inverse_minmax = self.min_max_scalerz_full_data.inverse_transform(gtz[time_step])
                xela_gtx_inverse_full = self.scaler_tx.inverse_transform(xela_gtx_inverse_minmax)
                xela_gty_inverse_full = self.scaler_ty.inverse_transform(xela_gty_inverse_minmax)
                xela_gtz_inverse_full = self.scaler_tz.inverse_transform(xela_gtz_inverse_minmax)
                gt_descalled_data.append(np.concatenate((xela_gtx_inverse_full, xela_gty_inverse_full, xela_gtz_inverse_full), axis=1))
        else:
            for time_step in range(tactile_data.shape[0]):
                # convert the image back to the 48 taxel features:
                sequence_p = []
                sequence_g = []
                for ps in range(tactile_data.shape[1]):
                    sequence_p.append(cv2.resize(torch.tensor(tactile_data)[time_step][ps].permute(1, 2, 0).numpy(), dsize=(4, 4), interpolation=cv2.INTER_CUBIC).flatten())
                    sequence_g.append(cv2.resize(torch.tensor(groundtruth_data)[time_step][ps].permute(1, 2, 0).numpy(), dsize=(4, 4), interpolation=cv2.INTER_CUBIC).flatten())

                pt_descalled_data.append(sequence_p)
                gt_descalled_data.append(sequence_g)

                # (ptx, pty, ptz) = np.split(np.array(sequence_p), 3, axis=1)
                # (gtx, gty, gtz) = np.split(np.array(sequence_g), 3, axis=1)
                #
                # xela_ptx_inverse_minmax = self.min_max_scalerx_full_data.inverse_transform(ptx)
                # xela_pty_inverse_minmax = self.min_max_scalery_full_data.inverse_transform(pty)
                # xela_ptz_inverse_minmax = self.min_max_scalerz_full_data.inverse_transform(ptz)
                # xela_ptx_inverse_full = self.scaler_tx.inverse_transform(xela_ptx_inverse_minmax)
                # xela_pty_inverse_full = self.scaler_ty.inverse_transform(xela_pty_inverse_minmax)
                # xela_ptz_inverse_full = self.scaler_tz.inverse_transform(xela_ptz_inverse_minmax)
                # pt_descalled_data.append(np.concatenate((xela_ptx_inverse_full, xela_pty_inverse_full, xela_ptz_inverse_full), axis=1))
                #
                # xela_gtx_inverse_minmax = self.min_max_scalerx_full_data.inverse_transform(gtx)
                # xela_gty_inverse_minmax = self.min_max_scalery_full_data.inverse_transform(gty)
                # xela_gtz_inverse_minmax = self.min_max_scalerz_full_data.inverse_transform(gtz)
                # xela_gtx_inverse_full = self.scaler_tx.inverse_transform(xela_gtx_inverse_minmax)
                # xela_gty_inverse_full = self.scaler_ty.inverse_transform(xela_gty_inverse_minmax)
                # xela_gtz_inverse_full = self.scaler_tz.inverse_transform(xela_gtz_inverse_minmax)
                # gt_descalled_data.append(np.concatenate((xela_gtx_inverse_full, xela_gty_inverse_full, xela_gtz_inverse_full), axis=1))

        return np.array(pt_descalled_data), np.array(gt_descalled_data)

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

        return mae.data.cpu().numpy() / (self.n_past + self.n_future), kld.data.cpu ().numpy () / (self.n_future + self.n_past), torch.stack(outputs)[:, :cut_point, :,:,:]

    def kl_criterion(self, mu1, logvar1, mu2, logvar2):
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2 / sigma1) +(torch.exp(logvar1) +(mu1 - mu2) ** 2) /(2 * torch.exp(logvar2)) - 1 / 2
        return kld.sum() / self.batch_size

    def run_batch(self, tactile, action):
        if self.image_size == 0:
            action = action.squeeze(-1).permute(1, 0, 2).to(device)
            tactile = torch.flatten(tactile, start_dim=2).permute(1, 0, 2).to(device)
        else:
            tactile = tactile.permute(1, 0, 4, 3, 2).to(device)
            action = action.squeeze(-1).permute(1, 0, 2).to(device)

        if self.model_name == "CDNA":
            tactile_predictions = self.CDNA_pass_through(tactiles=tactile, actions=action)
        elif self.model_name == "SV2P":
            tactile_predictions = self.SV2P_pass_through(tactiles=tactile, actions=action)
        elif self.model_name == "SVG":
            __, _, tactile_predictions = self.SVG_pass_through(scene=tactile, actions=action, test=True)
        else:
            tactile_predictions = self.model.forward(tactiles=tactile, actions=action)  # Step 3. Run our forward pass.

        loss = self.criterion(tactile_predictions, tactile[self.context_frames:])
        return tactile_predictions, tactile[self.context_frames:], loss.item()

    def save_trial(self, data_prediction, data_gt, trial_number):
        meta_data = np.load(self.test_data_dir + 'test_trial_' + str(trial_number) + '/trial_meta_0.npy')
        path_save = self.data_save_path + "test_trial_" + str(trial_number) + '/'
        try:
            os.mkdir(path_save)
        except:
            pass
        np.save(path_save + "prediction_data", data_prediction)
        np.save(path_save + "groundtruth_data", data_gt)
        np.save(path_save + "meta_data", meta_data)

    def load_scalars(self):
        self.scaler_tx = load(open(self.scaler_dir + "tactile_standard_scaler_x.pkl", 'rb'))
        self.scaler_ty = load(open(self.scaler_dir + "tactile_standard_scaler_y.pkl", 'rb'))
        self.scaler_tz = load(open(self.scaler_dir + "tactile_standard_scaler_z.pkl", 'rb'))
        self.min_max_scalerx_full_data = load(open(self.scaler_dir + "tactile_min_max_scalar_x.pkl", 'rb'))
        self.min_max_scalery_full_data = load(open(self.scaler_dir + "tactile_min_max_scalar_y.pkl", 'rb'))
        self.min_max_scalerz_full_data = load(open(self.scaler_dir + "tactile_min_max_scalar.pkl", 'rb'))

    def calc_losses(self, prediction_data, groundtruth_data, trial_number):
        mae_loss, mae_loss_1, mae_loss_5, mae_loss_10, mae_loss_x, mae_loss_y, mae_loss_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ssim_loss, ssim_loss_1, ssim_loss_5, ssim_loss_10, ssim_loss_x, ssim_loss_y, ssim_loss_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        psnr_loss, psnr_loss_1, psnr_loss_5, psnr_loss_10, psnr_loss_x, psnr_loss_y, psnr_loss_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        ssim_calc = SSIM(window_size=32)
        psnr_calc = PSNR()

        meta_data = np.load(self.meta[0])
        seen = meta_data[1][2]
        object = meta_data[1][0]

        index = 0
        index_ssim = 0
        with torch.no_grad():
            for batch_set_pred, batch_set_gt in zip(prediction_data, groundtruth_data):
                index += 1

                mae_loss_check = self.criterion(batch_set_pred, batch_set_gt).item()
                if math.isnan (mae_loss_check):
                    index -= 1
                else:
                    ## MAE:
                    mae_loss += mae_loss_check
                    mae_loss_1 += self.criterion (batch_set_pred[0], batch_set_gt[0]).item ()
                    mae_loss_5 += self.criterion (batch_set_pred[4], batch_set_gt[4]).item ()
                    mae_loss_10 += self.criterion (batch_set_pred[9], batch_set_gt[9]).item ()
                    mae_loss_x += self.criterion (batch_set_pred[:, :, 0], batch_set_gt[:, :, 0]).item ()
                    mae_loss_y += self.criterion (batch_set_pred[:, :, 1], batch_set_gt[:, :, 1]).item ()
                    mae_loss_z += self.criterion (batch_set_pred[:, :, 2], batch_set_gt[:, :, 2]).item ()
                    ## SSIM:

                    # ssim_loss += ssim_calc (batch_set_pred, batch_set_gt)
                    # ssim_loss_1 += ssim_calc (batch_set_pred[0].unsqueeze(0), batch_set_gt[0].unsqueeze(0))
                    # ssim_loss_5 += ssim_calc (batch_set_pred[4].unsqueeze(0), batch_set_gt[4].unsqueeze(0))
                    # ssim_loss_10 += ssim_calc (batch_set_pred[9].unsqueeze(0), batch_set_gt[9].unsqueeze(0))
                    # ssim_loss_x += ssim_calc (batch_set_pred[:, 0:0+1, :, :], batch_set_gt[:, 0:0+1, :, :])
                    # ssim_loss_y += ssim_calc (batch_set_pred[:, 1:1+1, :, :], batch_set_gt[:, 1:1+1, :, :])
                    # ssim_loss_z += ssim_calc (batch_set_pred[:, 2:2+1, :, :], batch_set_gt[:, 2:2+1, :, :])
                    # ## PSNR:
                    # psnr_loss += psnr_calc (batch_set_pred, batch_set_gt)
                    # psnr_loss_1 += psnr_calc (batch_set_pred[0], batch_set_gt[0])
                    # psnr_loss_5 += psnr_calc (batch_set_pred[4], batch_set_gt[4])
                    # psnr_loss_10 += psnr_calc (batch_set_pred[9], batch_set_gt[9])
                    # psnr_loss_x += psnr_calc (batch_set_pred[:, 0:0+1, :, :], batch_set_gt[:, 0:0+1, :, :])
                    # psnr_loss_y += psnr_calc (batch_set_pred[:, 1:1+1, :, :], batch_set_gt[:, 1:1+1, :, :])
                    # psnr_loss_z += psnr_calc (batch_set_pred[:, 2:2+1, :, :], batch_set_gt[:, 2:2+1, :, :])

        self.performance_data.append (
            [mae_loss / index, mae_loss_1 / index, mae_loss_5 / index, mae_loss_10 / index, mae_loss_x / index,
             mae_loss_y / index, mae_loss_z / index, seen, object])
             # ssim_loss / index, ssim_loss_1 / index,
             # ssim_loss_5 / index, ssim_loss_10 / index, ssim_loss_x / index, ssim_loss_y / index,
             # ssim_loss_z / index, psnr_loss / index, psnr_loss_1 / index, psnr_loss_5 / index,
             # psnr_loss_10 / index, psnr_loss_x / index, psnr_loss_y / index, psnr_loss_z / index, seen, object])

        print(self.performance_data)

        print ("object: ", object)
        self.objects.append(object)

    def save_performance(self):
        self.performance_data = np.array([[float(0.025331925462546284), float(0.011709410127418206), float(0.025347584810874357), float(0.03562737875660937), float(0.029740294250454696),
          float(0.029850441709870022), float(0.029589793177277703), '1', '12'],
         [float(0.028718649070530566), float(0.02830865305236953), float(0.027992812933112378), float(0.029747702329746346), float(0.029547435751798766),
          float(0.029588835347611305), float(0.02879450208799585), '1', '10'],
         [float(0.19296999669349785), float(0.17234593228025288), float(0.19334307653789826), float(0.2050338971052081), float(0.17713492374809955),
          float(0.17441329782516046), float(0.17427653092098364), '1', '4'],
         [float(0.0592116969945731), float(0.06055505894876072), float(0.05765559362771455), float(0.0615120683086228), float(0.06035517753266236),
          float(0.06011823957155952), float(0.05907481062989545), '1', '13'],
         [float(0.03725137389734272), float(0.025810647380444432), float(0.03718757553589205), float(0.046694168301169264), float(0.03374227565937033),
          float(0.03318172190771546), float(0.03384020349169655), '1', '17'],
         [float(0.028925127898000166), float(0.0262102612728515), float(0.02828169998628498), float(0.03313613826541772), float(0.026042332347870047),
          float(0.02635164452386528), float(0.026913120385316812), '1', '7'],
         [float(0.0411143755429668), float(0.03320995366014195), float(0.041909600784293684), float(0.04652864363645799), float(0.02836687580366137),
          float(0.028469071649222296), float(0.02892426463234513), '1', '16'],
         [float(0.029978007968883356), float(0.03232730354241983), float(0.02749809539784829), float(0.03343019901428285), float(0.028704855494271573),
          float(0.027634299600491474), float(0.028264805170988967), '1', '7'],
         [float(0.029806839160621167), float(0.02835910625522956), float(0.028639294307678937), float(0.03365205851756036), float(0.029668452011421324),
          float(0.028640727480873464), float(0.029123504428192972), '1', '7'],
         [float(0.04361097234799418), float(0.038961987775685485), float(0.04343787210798541), float(0.04845006267085325), float(0.046581013686954977),
          float(0.04639030866064998), float(0.04481928988771383), '1', '10'],
         [float(0.04201912502815694), float(0.03466318244758396), float(0.039920316031777564), float(0.051530417474561635), float(0.04260954947245342),
          float(0.04198555103905591), float(0.04127554710448524), '1', '14'],
         [float(0.03946160225871396), float(0.03610291223352154), float(0.03868414410753242), float(0.04420962582296091), float(0.03437210277964672),
          float(0.03418802143675639), float(0.034182310130790615), '1', '17'],
         [float(0.07427404067357343), float(0.07133020275214767), float(0.07279417352148654), float(0.0776958799093328), float(0.07030746410761012),
          float(0.0721476366737621), float(0.07235409650809509), '1', '13'],
         [float(0.11015168089038274), float(0.10041243110244405), float(0.11011895702007836), float(0.11758863843526457), float(0.09551834910613023),
          float(0.09279616157832284), float(0.0933879527408175), '1', '4'],
         [float(0.03775462486747048), float(0.027543219567047365), float(0.03628972991423301), float(0.048997343834782145), float(0.03285169653613998),
          float(0.03208975017031762), float(0.03157849888761006), '1', '16'],
         [float(0.03267390568420958), float(0.03016555482354849), float(0.0314385969247391), float(0.037977679232274415), float(0.032156924585075075),
          float(0.031206645502416946), float(0.03126382050034938), '1', '7'],
         [float(0.024326892286858897), float(0.03634727118456876), float(0.02112005720365147), float(0.02473585313224546), float(0.02285445121743247),
          float(0.02242358573171975), float(0.023652665647241288), '1', '16'],
         [float(0.02855945400983454), float(0.03352717917981038), float(0.026012380915996898), float(0.03236709949915787), float(0.02926849235731914),
          float(0.028344813182605916), float(0.028026181752849596), '1', '4'],
         [float(0.03517200495432882), float(0.020329300765382145), float(0.03521569205032101), float(0.0470652099427523), float(0.036543004281674195),
          float(0.03619675546414355), float(0.035889703641166386), '1', '14'],
         [float(0.03831814350849262), float(0.03206285098145287), float(0.03632671151728078), float(0.04614620188780147), float(0.04046222318390602),
          float(0.0391993699750923), float(0.03801665619140166), '1', '10'],
         [float(0.04275885089699711), float(0.02822898675820657), float(0.042196100604321274), float(0.05565524555742741), float(0.04211341254413128),
          float(0.04127220229910953), float(0.040817624479532244), '1', '7'],
         [float(0.021422601571583096), float(0.02314970488120691), float(0.02031469072798665), float(0.024113818551877098), float(0.023067841507634225),
          float(0.021949267681827846), float(0.021624370419601097), '1', '5'],
         [float(0.03456761068038976), float(0.03566002821670243), float(0.033738021700215184), float(0.034217453958655074), float(0.03601892777312405),
          float(0.036150497137252276), float(0.035223060282901744), '1', '10'],
         [float(0.05495948493548292), float(0.046115878981262605), float(0.05412036748755108), float(0.06272021550583753), float(0.06715961445136479),
          float(0.06575210133192219), float(0.06410835224614495), '1', '13'],
         [float(0.035113279654963664), float(0.021884721962381525), float(0.03447667181639597), float(0.046594723935104464), float(0.03388800621349312),
          float(0.03369726011102159), float(0.03350939261581928), '1', '14'],
         [float(0.023340350061640744), float(0.026276913812945304), float(0.02197789621476253), float(0.025599476849000068), float(0.02326501869451748),
          float(0.022703660450642184), float(0.022441005061058207), '1', '5'],
         [float(0.032029514701696495), float(0.040949523452282165), float(0.028706209967747386), float(0.032024668366064095), float(0.03056600770168674),
          float(0.029774699688932482), float(0.03044413180360871), '1', '4'],
         [float(0.0951579755261362), float(0.09147528930104332), float(0.09345318556032674), float(0.10481686022378013), float(0.09818433017140912),
          float(0.09572331567188937), float(0.09451570970279549), '1', '13'],
         [float(0.04052847511211068), float(0.0362839903444888), float(0.0408804921186244), float(0.043794850776048035), float(0.029787971419567356),
          float(0.029461594588332335), float(0.029547492617133016), '1', '4'],
         [float(0.03548137186675943), float(0.02819693589743267), float(0.03489165179811979), float(0.043058558607534055), float(0.036622001499097834),
          float(0.03579578174137941), float(0.0348221610440442), '1', '4'],
         [float(0.05048554989607372), float(0.04700039231686587), float(0.0497278900843328), float(0.05382385609790796), float(0.04455290041483704),
          float(0.04351250838435758), float(0.04442022003771395), '1', '17'],
         [float(0.026403294040309163), float(0.011074844676221333), float(0.026379068756480193), float(0.0379344474634193), float(0.0329416147044306),
          float(0.032973498171340584), float(0.032883519768291763), '1', '12'],
         [float(0.06937497105848577), float(0.05597395327548304), float(0.07013668671309475), float(0.07905284201519357), float(0.0640792818850882),
          float(0.06177619925241858), float(0.06186646493595271), '1', '7'],
         [float(0.05192258599182141), float(0.043604493652331246), float(0.050469951329474474), float(0.05701417549267069), float(0.07645207071869538),
          float(0.0756565060616396), float(0.07082776081930289), '1', '10'],
         [float(0.08889712100952035), float(0.07499438096096532), float(0.08824960022078206), float(0.10188845770123105), float(0.09139675339166489),
          float(0.08897038841516608), float(0.0882483191607106), '1', '7'],
         [float(0.024780828617011513), float(0.025964299902614434), float(0.023111772435441366), float(0.028494801661962534), float(0.025122256803493802),
          float(0.024389758100854534), float(0.024709633610605367), '1', '7'],
         [float(0.03441574534422956), float(0.02820012265231547), float(0.034139821489969656), float(0.041708394771355184), float(0.03287135039152967),
          float(0.033162359381094575), float(0.03281736053894599), '1', '5'],
         [float(0.033972303207331106), float(0.023769196447969047), float(0.03368844054112176), float(0.04371602974726625), float(0.032071883399215226),
          float(0.031873406442585135), float(0.031332657571425136), '1', '5'],
         [float(0.0884786940469647), float(0.0710580934899927), float(0.0881906094720328), float(0.09795475181882941), float(0.08491279268082207),
          float(0.08422650921695082), float(0.0826494704734608), '1', '16'],
         [float(0.033721847554944874), float(0.03864982228100082), float(0.031705016013631494), float(0.03201665552838517), float(0.030333515081655753),
          float(0.03000191386812415), float(0.03067847932562512), '1', '16'],
         [float(0.029681999441878548), float(0.02223151035221486), float(0.0298304616184584), float(0.03672583176401155), float(0.029501251516671018),
          float(0.029162853898416306), float(0.029633288297416835), '1', '12'],
         [float(0.07905262333845761), float(0.07539167121549126), float(0.07863868313466824), float(0.0816798912551625), float(0.05547760087360579),
          float(0.0568161077103439), float(0.05880994581524644), '1', '13'],
         [float(0.04094042765100111), float(0.03446120230573646), float(0.039876630519470456), float(0.04837913188609958), float(0.04353553664649307),
          float(0.04218021903742743), float(0.041419962349023606), '1', '7'],
         [float(0.02903068098268345), float(0.03073484738275443), float(0.02711655850267831), float(0.03284518640951173), float(0.02795236111463358),
          float(0.027387604753366084), float(0.02761802650351411), '1', '7'],
         [float(0.027245216421626715), float(0.02844147927536395), float(0.025342141169733486), float(0.03040899633220366), float(0.02771167531696343),
          float(0.027128666415488157), float(0.02709363642075507), '1', '14'],
         [float(0.13528630691893764), float(0.12531693350802822), float(0.135007328380879), float(0.1417156187105434), float(0.12864242912204624),
          float(0.1270249091545889), float(0.12483422360767465), '1', '17'],
         [float(0.047855017848381524), float(0.030011016828295548), float(0.04752919475723934), float(0.06249343020163373), float(0.04230504672682505),
          float(0.0423062550170081), float(0.04154387233598578), '1', '13'],
         [float(0.038006862054385665), float(0.02540578766075789), float(0.03727546838586623), float(0.04851404210236275), float(0.038591114206696454),
          float(0.038831038795903605), float(0.03830699297546778), '1', '12'],
         [float(0.03424566630274058), float(0.01906384700643165), float(0.03391635722347668), float(0.04674531615206173), float(0.036079651532428605),
          float(0.03624874202268464), float(0.035208982543221544), '1', '17'],
         [float(0.04347299830713593), float(0.04438961926511029), float(0.04121064535051233), float(0.045827305207874314), float(0.045343972848680726),
          float(0.042973658179750844), float(0.041916426445022664), '1', '17'],
         [float(0.03995668526124401), float(0.026141959792195062), float(0.03844047540837189), float(0.05343942847774969), float(0.03813647790379899),
          float(0.03761932514031831), float(0.03683941751795331), '1', '13'],
         [float(0.03195459809134768), float(0.02072688228943195), float(0.030796171478068953), float(0.043406734252787384), float(0.030889661460919843),
          float(0.031103396047082246), float(0.030582401730822332), '1', '14']])

        self.performance_data = self.performance_data.astype(np.float32)
        performance_data_full = []
        performance_data_full.append(["train loss MAE(L1): ", (sum([i[0] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["train loss MAE(L1) pred ts 1: ", (sum([i[1] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["train loss MAE(L1) pred ts 5: ", (sum([i[2] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["train loss MAE(L1) pred ts 10: ", (sum([i[3] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["train loss MAE(L1) pred force x: ", (sum([i[4] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["train loss MAE(L1) pred force y: ", (sum([i[5] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["train loss MAE(L1) pred force z: ", (sum([i[6] for i in self.performance_data]) / len(self.performance_data))])

        performance_data_full.append (["train loss MAE(L1) Seen objects: ", (sum([i[0] for i in self.performance_data if i[-2] == 1]) / len([i[0] for i in self.performance_data if i[-2] == 1]))])
        # performance_data_full.append (["train loss MAE(L1) Novel objects: ", (sum([i[0] for i in self.performance_data if i[-2] == '0']) / len([i[0] for i in self.performance_data if i[-2] == '0']))])

        self.objects = [i for i in range(30)]

        self.objects = list(set(self.objects))
        for object in self.objects:
            try:
                performance_data_full.append(["train loss MAE(L1) Trained object " + str(object) + ": ", (sum([i[0] for i in self.performance_data if i[-1] == int(object)]) / len([i[0] for i in self.performance_data if i[-1] == int(object)]))])
            except:
                print("object {} doesnt exist".format(str(object)))
        # performance_data_full.append(["test loss SSIM: ", (sum([i[7] for i in self.performance_data]) / len(self.performance_data))])
        # performance_data_full.append(["test loss SSIM pred ts 1: ", (sum([i[8] for i in self.performance_data]) / len(self.performance_data))])
        # performance_data_full.append(["test loss SSIM pred ts 5: ", (sum([i[9] for i in self.performance_data]) / len(self.performance_data))])
        # performance_data_full.append(["test loss SSIM pred ts 10: ", (sum([i[10] for i in self.performance_data]) / len(self.performance_data))])
        # performance_data_full.append(["test loss SSIM pred force x: ", (sum([i[11] for i in self.performance_data]) / len(self.performance_data))])
        # performance_data_full.append(["test loss SSIM pred force y: ", (sum([i[12] for i in self.performance_data]) / len(self.performance_data))])
        # performance_data_full.append(["test loss SSIM pred force z: ", (sum([i[13] for i in self.performance_data]) / len(self.performance_data))])
        #
        # performance_data_full.append (["test loss SSIM Seen objects: ", (sum([i[7] for i in self.performance_data if i[-2] == '1']) / len([i[7] for i in self.performance_data if i[-2] == '1']))])
        # performance_data_full.append (["test loss SSIM Novel objects: ", (sum([i[7] for i in self.performance_data if i[-2] == '0']) / len([i[7] for i in self.performance_data if i[-2] == '0']))])
        #
        # performance_data_full.append(["test loss PSNR: ", (sum([i[14] for i in self.performance_data]) / len(self.performance_data))])
        # performance_data_full.append(["test loss PSNR pred ts 1: ", (sum([i[15] for i in self.performance_data]) / len(self.performance_data))])
        # performance_data_full.append(["test loss PSNR pred ts 5: ", (sum([i[16] for i in self.performance_data]) / len(self.performance_data))])
        # performance_data_full.append(["test loss PSNR pred ts 10: ", (sum([i[17] for i in self.performance_data]) / len(self.performance_data))])
        # performance_data_full.append(["test loss PSNR pred force x: ", (sum([i[18] for i in self.performance_data]) / len(self.performance_data))])
        # performance_data_full.append(["test loss PSNR pred force y: ", (sum([i[19] for i in self.performance_data]) / len(self.performance_data))])
        # performance_data_full.append(["test loss PSNR pred force z: ", (sum([i[20] for i in self.performance_data]) / len(self.performance_data))])
        #
        # performance_data_full.append (["test loss PSNR Seen objects: ", (sum([i[14] for i in self.performance_data if i[-2] == '1']) / len([i[14] for i in self.performance_data if i[-2] == '1']))])
        # performance_data_full.append (["test loss PSNR Novel objects: ", (sum([i[14] for i in self.performance_data if i[-2] == '0']) / len([i[14] for i in self.performance_data if i[-2] == '0']))])

        [print(i) for i in performance_data_full]
        np.save(self.data_save_path + 'TRAIN_model_performance_loss_data', np.asarray(performance_data_full))


def main():
    model_path = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/SVG/saved_models/box_only_4layers_WORKING_model_14_02_2022_11_00/SVG_model"
    data_save_path = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/SVG/saved_models/box_only_4layers_WORKING_model_14_02_2022_11_00/"
    test_data_dir = "/home/user/Robotics/Data_sets/box_only_dataset/test_image_dataset_10c_10h_32_universal/"
    # test_data_dir = "/home/user/Robotics/Data_sets/box_only_dataset/Train_formatted_image_32_by_trial_FORSVG/"
    scaler_dir = "/home/user/Robotics/Data_sets/box_only_dataset/scalar_info_universal/"

    number_of_trials = [i for i in range(52)]  # [0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22]  # 0-22
    batch_size = 16
    context_frames = 10
    sequence_length = 20
    image_size = 32
    latent_channels = 10
    criterion = "L1"
    save_plots = True

    model = torch.load(model_path)
    model_name = "SVG"

    UMT = UniversalModelTester(model, number_of_trials, test_data_dir, image_size, criterion, model_path,
                         data_save_path, scaler_dir, model_name, batch_size,
                         context_frames, sequence_length, latent_channels)  # if not an image set image size to 0
    UMT.test_model(save_trial=save_plots)
    # UMT.save_performance()

if __name__ == "__main__":
    main()

    # def SVG_pass_through(self, tactiles, actions):
    #     # if not batch size == add a bunch of zeros and then remove them from the predictions:
    #     appended = False
    #     if tactiles.shape[1] != self.batch_size:
    #         cut_point = tactiles.shape[1]
    #         tactiles = torch.cat((tactiles, torch.zeros([20, (self.batch_size - tactiles.shape[1]), 3, self.image_size,self.image_size]).to(device)), axis=1)
    #         actions = torch.cat((actions, torch.zeros([20, (self.batch_size - actions.shape[1]), 6]).to(device)), axis=1)
    #         appended = True
    #
    #
    #     states = actions[0, :, :]
    #     state_action = torch.cat((torch.cat(20*[states.unsqueeze(0)], 0), actions), 2)
    #     state_action_image = torch.cat(32*[torch.cat(32*[state_action.unsqueeze(3)], axis=3).unsqueeze(4)], axis=4)
    #     x = torch.cat((state_action_image, tactiles), 2)