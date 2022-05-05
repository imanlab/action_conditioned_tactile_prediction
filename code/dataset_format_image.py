# -*- coding: utf-8 -*-
# RUN IN PYTHON 3

import os
import cv2
import csv
import glob
import torch
import numpy as np
import pandas as pd


from tqdm import tqdm
from pickle import dump
from sklearn import preprocessing
from datetime import datetime
from scipy.spatial.transform import Rotation as R

dataset_path = "/home/user/Robotics/Data_sets/box_only_dataset/"
# Hyper-parameters:
train_data_dir = dataset_path + 'train/'
test_data_dir  = dataset_path + 'test/'
# train_out_dir  = dataset_path + 'train_image_dataset_10c_10h_64_universal/'
train_out_dir  = dataset_path + 'train_linear_dataset_10c_10h/'
test_out_dir   = dataset_path + 'test_linear_dataset_10c_10h/'
scaler_out_dir = dataset_path + 'scalar_linear_test/'

smooth = True
image = False
image_height = 0
image_width = 0
context_length = 10
horrizon_length = 10


class data_formatter:
    def __init__(self):
        self.files_train = []
        self.files_test = []
        self.full_data_tactile = []
        self.full_data_robot = []
        self.smooth = smooth
        self.image = image
        self.image_height = image_height
        self.image_width = image_width
        self.context_length = context_length
        self.horrizon_length = horrizon_length

    def create_map(self):
        for stage in [train_out_dir, test_out_dir]:
            self.path_file = []
            index_to_save = 0
            print(stage)
            if stage == train_out_dir:
                files_to_run = self.files_train
            else:
                files_to_run = self.files_test

            for experiment_number, file in tqdm(enumerate(files_to_run)):
                # if stage == test_out_dir:
                #     path_save = stage + "test_trial_" + str(experiment_number) + '/'
                #     os.mkdir(path_save)
                #     self.path_file = []
                #     index_to_save = 0
                # else:
                #     path_save = stage

                path_save = stage + "test_trial_" + str(experiment_number) + '/'
                os.mkdir(path_save)
                self.path_file = []
                index_to_save = 0

                tactile, robot, meta = self.load_file_data(file)

                # scale the data
                for index, (standard_scaler, min_max_scalar) in enumerate(zip(self.tactile_standard_scaler, self.tactile_min_max_scalar)):
                    tactile[:, index] = standard_scaler.transform(tactile[:, index])
                    tactile[:, index] = min_max_scalar.transform(tactile[:, index])

                for index, min_max_scalar in enumerate(self.robot_min_max_scalar):
                    robot[:, index] = np.squeeze(min_max_scalar.transform(robot[:, index].reshape(-1, 1)))

                tactile_image_names = []
                if self.image:
                    current_images = []
                    for time_step in range(len(tactile)):
                        current_image = self.create_image(tactile[time_step][0], tactile[time_step][1], tactile[time_step][2])
                        image_name = "tactile_image_" + str(experiment_number) + "_time_step_" + str(time_step) + ".npy"
                        tactile_image_names.append(image_name)
                        np.save(path_save + image_name, current_image)
                        # current_images.append(cv2.resize(current_image, dsize=(4, 4), interpolation=cv2.INTER_CUBIC).flatten())

                    # tactile = np.array(current_images).astype(np.float64)

                sequence_length = self.context_length + self.horrizon_length
                for time_step in range(len(tactile) - sequence_length):
                    robot_data_euler_sequence = [robot[time_step + t] for t in range(sequence_length)]
                    tactile_data_sequence     = [tactile[time_step + t] for t in range(sequence_length)]
                    experiment_data_sequence  = experiment_number
                    time_step_data_sequence   = [time_step + t for t in range(sequence_length)]
                    if self.image:
                        tactile_image_name_sequence = [tactile_image_names[time_step + t] for t in range(sequence_length)]

                    ####################################### Save the data and add to the map ###########################################
                    np.save(path_save + 'robot_data_euler_' + str(index_to_save), robot_data_euler_sequence)
                    np.save(path_save + 'tactile_data_sequence_' + str(index_to_save), tactile_data_sequence)
                    if self.image:
                        np.save(path_save + 'tactile_image_name_sequence_' + str(index_to_save), tactile_image_name_sequence)
                    np.save(path_save + 'experiment_number_' + str(index_to_save), experiment_data_sequence)
                    np.save(path_save + 'time_step_data_' + str(index_to_save), time_step_data_sequence)
                    np.save(path_save + 'trial_meta_' + str(index_to_save), np.array(meta))
                    ref = []
                    ref.append('robot_data_euler_' + str(index_to_save) + '.npy')
                    ref.append('tactile_data_sequence_' + str(index_to_save) + '.npy')
                    if self.image:
                        ref.append('tactile_image_name_sequence_' + str(index_to_save) + '.npy')
                    ref.append('experiment_number_' + str(index_to_save) + '.npy')
                    ref.append('time_step_data_' + str(index_to_save) + '.npy')
                    ref.append('trial_meta_' + str(index_to_save) + '.npy')
                    self.path_file.append(ref)
                    index_to_save += 1

                # if stage == test_out_dir:
                #     self.test_no = experiment_number
                #     self.save_map(path_save, test=True)

                self.test_no = experiment_number
                self.save_map(path_save, test=True)

            self.save_map(path_save)

    def save_map(self, path, test=False):
        if test:
            with open(path + '/map_' + str(self.test_no) + '.csv', 'w') as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
                writer.writerow(['robot_data_path_euler', 'tactile_data_sequence', 'tactile_image_name_sequence', 'experiment_number', 'time_steps', 'meta'])
                for row in self.path_file:
                    writer.writerow(row)
        else:
            with open(path + '/map.csv', 'w') as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
                writer.writerow(['robot_data_path_euler', 'tactile_data_sequence', 'tactile_image_name_sequence', 'experiment_number', 'time_steps', 'meta'])
                for row in self.path_file:
                    writer.writerow(row)

    def scale_data(self):
        files = self.files_train + self.files_test
        for file in tqdm(files):
            tactile, robot, _ = self.load_file_data(file)
            self.full_data_tactile += list(tactile)
            self.full_data_robot += list(robot)

        self.full_data_robot = np.array(self.full_data_robot)
        self.full_data_tactile = np.array(self.full_data_tactile)

        self.robot_min_max_scalar = [preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(self.full_data_robot[:, feature].reshape(-1, 1)) for feature in range(6)]
        self.tactile_standard_scaler = [preprocessing.StandardScaler().fit(self.full_data_tactile[:, feature]) for feature in range(3)]
        self.tactile_min_max_scalar = [preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(self.tactile_standard_scaler[feature].transform(self.full_data_tactile[:, feature])) for feature in range(3)]

        self.save_scalars()

    def load_file_data(self, file):
        robot_state = np.array(pd.read_csv(file + '/robot_state.csv', header=None))
        xela_sensor = np.array(pd.read_csv(file + '/xela_sensor1.csv', header=None))
        meta_data = np.array(pd.read_csv(file + '/meta_data.csv', header=None))

        # convert orientation to euler, and remove column labels:
        robot_task_space = np.array([[state[-7], state[-6], state[-5]] + list(R.from_quat([state[-4], state[-3], state[-2], state[-1]]).as_euler('zyx', degrees=True)) for state in robot_state[1:]]).astype(float)

        # split tactile sensor into the three forces | find start value average for each force | find offsets for each taxel | take away start value average from the tactile data:
        tactile_data_split = [np.array(xela_sensor[1:, [i for i in range(feature, 48, 3)]]).astype(float) for feature in range(3)]
        tactile_mean_start_values = [int(sum(tactile_data_split[feature][0]) / len(tactile_data_split[feature][0])) for feature in range(3)]
        tactile_offsets = [[tactile_mean_start_values[feature] - tactile_starting_value for tactile_starting_value in tactile_data_split[feature][0]] for feature in range(3)]
        tactile_data = [[tactile_data_split[feature][ts] + tactile_offsets[feature] for feature in range(3)] for ts in range(tactile_data_split[0].shape[0])]

        if self.smooth:
            tactile_data = self.smooth_the_trial(np.array(tactile_data))
            tactile_data = tactile_data[3:-3, :, :]
            robot_task_space = robot_task_space[3:-3, :]

        return tactile_data, robot_task_space, meta_data

    def load_file_names(self):
        self.files_train = glob.glob(train_data_dir + '/*')
        self.files_test = glob.glob(test_data_dir + '/*')

    def smooth_the_trial(self, tactile_data):
        for force in range(tactile_data.shape[1]):
            for taxel in range(tactile_data.shape[2]):
                tactile_data[:, force, taxel] = [None for i in range(3)] + list(self.smooth_func(tactile_data[:, force, taxel], 6)[3:-3]) + [None for i in range(3)]

        return tactile_data

    def smooth_func(self, y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def save_scalars(self):
        # save the scalars
        dump(self.tactile_standard_scaler[0], open(scaler_out_dir + 'tactile_standard_scaler_x.pkl', 'wb'), protocol=2)
        dump(self.tactile_standard_scaler[1], open(scaler_out_dir + 'tactile_standard_scaler_y.pkl', 'wb'), protocol=2)
        dump(self.tactile_standard_scaler[2], open(scaler_out_dir + 'tactile_standard_scaler_z.pkl', 'wb'), protocol=2)
        dump(self.tactile_min_max_scalar[0], open(scaler_out_dir + 'tactile_min_max_scalar_x.pkl', 'wb'), protocol=2)
        dump(self.tactile_min_max_scalar[1], open(scaler_out_dir + 'tactile_min_max_scalar_y.pkl', 'wb'), protocol=2)
        dump(self.tactile_min_max_scalar[2], open(scaler_out_dir + 'tactile_min_max_scalar.pkl', 'wb'), protocol=2)

        dump(self.robot_min_max_scalar[0], open(scaler_out_dir + 'robot_min_max_scalar_px.pkl', 'wb'), protocol=2)
        dump(self.robot_min_max_scalar[1], open(scaler_out_dir + 'robot_min_max_scalar_py.pkl', 'wb'), protocol=2)
        dump(self.robot_min_max_scalar[2], open(scaler_out_dir + 'robot_min_max_scalar_pz.pkl', 'wb'), protocol=2)
        dump(self.robot_min_max_scalar[3], open(scaler_out_dir + 'robot_min_max_scalar_ex.pkl', 'wb'), protocol=2)
        dump(self.robot_min_max_scalar[4], open(scaler_out_dir + 'robot_min_max_scalar_ey.pkl', 'wb'), protocol=2)
        dump(self.robot_min_max_scalar[5], open(scaler_out_dir + 'robot_min_max_scalar_ez.pkl', 'wb'), protocol=2)

    def create_image(self, tactile_x, tactile_y, tactile_z):
        # convert tactile data into an image:
        image = np.zeros((4, 4, 3), np.float32)
        index = 0
        for x in range(4):
            for y in range(4):
                image[x][y] = [tactile_x[index], tactile_y[index], tactile_z[index]]
                index += 1
        reshaped_image = np.rot90(cv2.resize(image.astype(np.float32), dsize=(self.image_height, self.image_width), interpolation=cv2.INTER_CUBIC), k=1, axes=(0, 1))
        return reshaped_image


def main():
    df = data_formatter()
    df.load_file_names()
    df.scale_data()
    df.create_map()


if __name__ == "__main__":
    main()
