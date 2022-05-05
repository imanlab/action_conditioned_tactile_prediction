# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import cv2
import copy
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.ticker import(AutoMinorLocator, MultipleLocator)
from matplotlib.animation import FuncAnimation

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset

data_save_path = "/home/user/Robotics/tactile_prediction/tactile_prediction/models/PixelMotionNet-AC/saved_models/box_only_dataset_model_25_11_2021_14_10/"

seed = 42
batch_size = 32
learning_rate = 1e-3
context_frames = 10
sequence_length = 20

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available

test_trial = 10
taxel = 31

trim_min = 0
trim_max = -1

def create_test_plots(experiment_to_test):
    trial_groundtruth_data = np.load(data_save_path + "test_plots_" + str(test_trial) + "/trial_groundtruth_data.npy")
    trial_predicted_data_t1 = np.load(data_save_path + "test_plots_" + str(test_trial) + "/prediction_data_t1.npy")
    trial_predicted_data_t5 = np.load(data_save_path + "test_plots_" + str(test_trial) + "/prediction_data_t5.npy")
    trial_predicted_data_t10 = np.load(data_save_path + "test_plots_" + str(test_trial) + "/prediction_data_t10.npy")
    meta = np.load(data_save_path + "test_plots_" + str(test_trial) + "/meta_data.npy")

    trial_groundtruth_data   = trial_groundtruth_data[trim_min:trim_max]
    trial_predicted_data_t1  = trial_predicted_data_t1[trim_min:trim_max]
    trial_predicted_data_t5  = trial_predicted_data_t5[trim_min:trim_max]
    trial_predicted_data_t10 = trial_predicted_data_t10[trim_min:trim_max]

    index = 0
    titles = ["sheerx", "sheery", "normal"]
    # for j in range(3):
    #     for i in range(16):
    index = taxel

    groundtruth_taxle = []
    predicted_taxel_t1 = []
    predicted_taxel_t5 = []
    predicted_taxel_t10 = []
    for k in range(len(trial_predicted_data_t10)):
        predicted_taxel_t1.append(trial_predicted_data_t1[k][index])
        predicted_taxel_t5.append(trial_predicted_data_t5[k][index])
        predicted_taxel_t10.append(trial_predicted_data_t10[k][index])
        groundtruth_taxle.append(trial_groundtruth_data[k][index])

    index += 1

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('time step')
    ax1.set_ylabel('tactile reading')
    line_5 = ax1.plot([i for i in predicted_taxel_t1], alpha=1.0, c="orange", label="Pred_t1")
    line_2 = ax1.plot([i for i in predicted_taxel_t5], alpha=1.0, c="g", label="Pred_t5")
    line_1 = ax1.plot([i for i in predicted_taxel_t10], alpha=1.0, c="b", label="Pred_t10")
    line_3 = ax1.plot(groundtruth_taxle, alpha=1.0, c="r", label="Gt")
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('MAE between gt+t10, pred_t10')  # we already handled the x-label with ax1
    line_4 = ax2.plot([None for i in range(10)] + [abs(pred - gt) for(gt, pred) in zip(groundtruth_taxle[10:], predicted_taxel_t10[:-10])], alpha=0.4, c="k", label="MAE")
    lines = line_1 + line_2 + line_3 + line_4 + line_5
    labs = [l.get_label() for l in lines]

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.subplots_adjust(top=0.90)
    ax1.legend(lines, labs, loc="upper right")
    if len(predicted_taxel_t5) < 150:
        ax1.xaxis.set_major_locator(MultipleLocator(10))
        ax1.xaxis.set_minor_locator(AutoMinorLocator(10))
    elif len(predicted_taxel_t5) > 150 and len(predicted_taxel_t5) < 1000:
        ax1.xaxis.set_major_locator(MultipleLocator(25))
        ax1.xaxis.set_minor_locator(AutoMinorLocator(25))
    elif len(predicted_taxel_t5) > 1000:
        ax1.xaxis.set_major_locator(MultipleLocator(100))
        ax1.xaxis.set_minor_locator(AutoMinorLocator(100))

    ax1.grid(which='minor')
    ax1.grid(which='major')
    plt.title("AV-PMN model taxel " + str(index) + ". Test " + str(test_trial) + ". Seen at training: " + meta[1][2] + ". Object: " + meta[1][0])
    # plt.savefig(plot_save_dir + '/test_plot_taxel_' + str(index) + '.png', dpi=300)
    plt.show()

create_test_plots(test_trial)