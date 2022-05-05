#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Predict the next n frames from a trained model
# ==============================================

import numpy as np
import chainer
import random
from chainer import cuda
import chainer.functions as F
from matplotlib import pyplot as plt
import cv2
import copy
# from model import Model
# from model import concat_examples
from SPOTS_PRI import Model
from SPOTS_PRI import concat_examples
from SPOTS_PRI import load_and_concat_examples
from pickle import load

try:
    import cupy
except:
    cupy = np
    pass

import click
import os
import csv
import logging
import glob
import subprocess

import six.moves.cPickle as pickle

from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageChops
import imageio

DNA_KERN_SIZE = 3  # could try 16 = 1 for every sensor? was 5


class get_data_info():
    def __init__(self, trial, data_dir, data_index, get_changing_data, process_channel):
        self.data_dir = data_dir
        self.data_map = []
        with open(data_dir + '/map_' + str(trial) + '.csv', 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                self.data_map.append(row)
        self.data_map = self.data_map[1:]

    def get_item(self, item):
        image = np.float32([np.load(self.data_dir + "/" + str(image_name)) for image_name in np.load(self.data_dir + '/' + self.data_map[item][2])])
        action = np.float32(np.load(self.data_dir + '/' + self.data_map[item][0]))
        state = np.float32(np.load(self.data_dir + '/' + self.data_map[item][0]))

        return image, action, state

    def get_data_len(self):
        return len(self.data_map)


class DataGenerator():
    def __init__(self, batch_size, logger, data_dir, generator_type, trial):
        data_map = []
        with open(data_dir + '/map_' + str(trial) + '.csv', 'rb') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                data_map.append(row)

        if len(data_map) <= 1:  # empty or only header
            logger.error("No file map found")
            exit()

        # Load the images, actions and states
        images = []
        actions = []
        states = []

        for i in xrange(1, len(data_map)):  # Exclude the header
            images.append(str(data_dir + '/' + data_map[i][2]))
            actions.append(str(data_dir + '/' + data_map[i][0]))
            states.append(str(data_dir + '/' + data_map[i][0]))

        self.images_training = np.asarray(images)
        self.actions_training = np.asarray(actions)
        self.states_training = np.asarray(states)

        # Group the images, actions and states
        self.grouped_set_training = []
        for idx in xrange(len(self.images_training)):
            group = []
            group.append(self.images_training[idx])
            group.append(self.actions_training[idx])
            group.append(self.states_training[idx])
            self.grouped_set_training.append(group)

    def return_Data(self):
        return [self.images_training, self.actions_training, self.states_training, self.images_validation, self.actions_validation, self.states_validation,
        self.grouped_set_training, self.grouped_set_validation]

    def train_iter(self, batch_size):
        test_iter = chainer.iterators.SerialIterator(self.grouped_set_training, batch_size, repeat=False, shuffle=False)
        return test_iter


# =================================================
# Main entry point of the training processes (main)
# =================================================
@click.command()
@click.option('--model_dir', type=click.STRING, default='20220223-111825-CDNA-8', help='Directory containing model.')
@click.option('--model_name', type=click.STRING, default='training-7', help='The name of the model.')
@click.option('--data_index', type=click.INT, default=0, help='Directory containing data.')
@click.option('--get_changing_data', type=click.INT, default=0, help='Should the program look for a test sample where there is change in the time step.')
@click.option('--models_dir', type=click.Path(exists=True), default='/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/CDNA_chainer/saved_models/', help='Directory containing the models.')
@click.option('--data_dir', type=click.Path(exists=True), default='/home/user/Robotics/Data_sets/box_only_dataset/train_image_dataset_10c_10h_64_universal_split', help='Directory containing data.')
@click.option('--time_step', type=click.INT, default=10, help='Number of time steps to predict.')
@click.option('--model_type', type=click.STRING, default='CDNA', help='Type of the trained model.')
@click.option('--schedsamp_k', type=click.FLOAT, default=900.0, help='The k parameter for schedules sampling. -1 for no scheduled sampling.')
@click.option('--context_frames', type=click.INT, default=2, help='Number of frames before predictions.')
@click.option('--use_state', type=click.INT, default=1, help='Whether or not to give the state+action to the model.')
@click.option('--num_masks', type=click.INT, default=10, help='Number of masks, usually 1 for DNA, 10 for CDNA, STP.')
@click.option('--image_height', type=click.INT, default=64, help='Height of one predicted frame.')
@click.option('--image_width', type=click.INT, default=64, help='Width of one predicted frame.')
@click.option('--original_image_height', type=click.INT, default=64, help='Height of one predicted frame.')
@click.option('--original_image_width', type=click.INT, default=64, help='Width of one predicted frame.')
@click.option('--downscale_factor', type=click.FLOAT, default=1, help='Downscale the image by this factor. (was 0.5)')
@click.option('--gpu', type=click.INT, default=0, help='ID of the gpu to use')
@click.option('--gif', type=click.INT, default=1, help='Create a GIF of the predicted result.')
@click.option('--process_channel', type=click.INT, default=0, help='if you want to train on a single channel')
def main(model_dir, model_name, data_index, get_changing_data, models_dir, data_dir, time_step, model_type, schedsamp_k, context_frames, use_state, num_masks, image_height, image_width, original_image_height, original_image_width, downscale_factor, gpu, gif, process_channel):
    """ Predict the next {time_step} frame based on a trained {model} """
    logger = logging.getLogger(__name__)
    trials = [i for i in range(1, 52)]
    batch_size = 4
    save_dir = "/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/CDNA_chainer/saved_models/20220223-111825-CDNA-8/"

    mae_loss = 0.0
    psnr_loss = 0.0
    data_len = 0.0

    # Extract the information about the model
    if model_type == '':
        split_name = model_dir.split("-")
        if len(split_name) != 4:
            raise ValueError("Model {} is not recognized, use --model_type to describe the type".format(model_dir))
        model_type = split_name[2]

    # Load the model for prediction
    model = Model(num_masks=10,
                  is_cdna="CDNA",
                  use_state=1,
                  scheduled_sampling_k=900.0,
                  num_frame_before_prediction=10,
                  prefix='predict')
    if gpu > -1:
        cuda.get_device(gpu).use()
        model.to_gpu()
        xp = cupy
        xp.cuda.alloc_pinned_memory(5 * 1024 ** 3)  # 8gb
    else:
        xp = np

    for trial in trials:
        print("trial: ", trial)
        path = models_dir + '/' + model_dir
        if not os.path.exists(path + '/' + model_name):
            raise ValueError("Directory {} does not exists".format(path))
        if not os.path.exists(data_dir):
            raise ValueError("Directory {} does not exists".format(data_dir))

        logger.info("Loading data {}".format(data_index))

        DG = DataGenerator(batch_size=batch_size, logger=logger, data_dir=data_dir+"/test_trial_"+str(trial), generator_type=1, trial=trial)
        train_iter = DG.train_iter(batch_size)

        data_gen = get_data_info(trial, data_dir+"/test_trial_"+str(trial), data_index, get_changing_data, process_channel)
        data_length = data_gen.get_data_len()
        data_gen = []

        gt = []
        pd = []
        INDEX = 0

        for time_step in range(0, data_length/batch_size):
            batch = train_iter.next()
            img_pred, act_pred, sta_pred = load_and_concat_examples(batch, data_dir+"/test_trial_"+str(trial)+'/')

            print(INDEX)
            INDEX += 1

            # img_pred, act_pred, sta_pred = concat_examples([data_gen.get_item(time_step)])

            # Predict the new images
            with chainer.using_config('train', False):
                loss = model(xp.asarray(img_pred, dtype=xp.float32),
                             xp.asarray(act_pred, dtype=xp.float32),
                             xp.asarray(sta_pred, dtype=xp.float32),
                             0)
                predicted_images = copy.deepcopy(model.gen_images)
            model.reset_state()

            gt_data = img_pred[10:].squeeze()
            pd_data = np.array([i.data for i in predicted_images[9:]]).squeeze()

            # convert back:

            for batch_indi in range(pd_data.shape[1]):
                sequence_p = []
                sequence_g = []
                for ps in range(len(gt_data)):
                    sequence_p.append(cv2.resize(np.swapaxes(np.float32(pd_data[ps][batch_indi]), 0, 2), dsize=(4, 4), interpolation=cv2.INTER_CUBIC).flatten())
                    sequence_g.append(cv2.resize(np.swapaxes(np.float32(gt_data[ps][batch_indi]), 0, 2), dsize=(4, 4), interpolation=cv2.INTER_CUBIC).flatten())
                gt.append(sequence_g)
                pd.append(sequence_p)

        print("saving1")
        np.save(save_dir + "train_gt_trial_"+str(trial), np.array(gt))
        print("saving2")
        np.save(save_dir + "train_pd_trial_"+str(trial), np.array(pd).squeeze())
        print("end of trial: ", str(trial))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt, stream=sys.stdout)
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
