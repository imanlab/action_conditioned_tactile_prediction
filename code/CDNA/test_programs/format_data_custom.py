# -*- coding: utf-8 -*-

import os
import cv2
import csv
import glob
import click
import logging


from PIL import Image 
from tqdm import tqdm
from dotenv import find_dotenv, load_dotenv

import numpy as np
import pandas as pd
import tensorflow as tf

from matplotlib import pyplot as plt


class DataFormatter():
    def __init__(self, data_set_length, data_dir, out_dir, sequence_length, image_original_width, image_original_height, image_original_channel, image_resize_width, image_resize_height, state_action_dimension, create_img, create_img_prediction):
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.sequence_length = sequence_length
        self.image_original_width = image_original_width
        self.image_original_height = image_original_height
        self.image_original_channel = image_original_channel
        self.image_resize_width = image_resize_width
        self.image_resize_height = image_resize_height
        self.state_action_dimension = state_action_dimension
        self.create_img = create_img
        self.create_img_prediction = create_img_prediction

        self.logger = logging.getLogger(__name__)
        self.logger.info('making final data set from raw data')

        with tf.Session() as sess:
            files = glob.glob(data_dir + '/*')
            if len(files) == 0:
                self.logger.error("No files found with extensions .tfrecords in directory {0}".format(self.out_dir))
                exit()

            robot_positions = []
            image_names = []
            image_names_labels = []
            data_set_length = data_set_length # 1000  # 1001
            frameRate = 1.0/10.0        # It will capture image in each 0.5 second

            for i in tqdm(range(0, data_set_length)):
                # robot_positions_new_sample = pd.read_csv(data_dir + 'robot_EE_pose_and_vel/data_set_' + str(i) + '.csv', header=None)
                robot_positions_new_sample = pd.read_csv(data_dir + 'robot_pos/data_set_' + str(i) + '_robot_data_store_position' + '.csv', header=None)
                vidcap = cv2.VideoCapture('/content/camera_1_video/rgb_video/sample_' + str(i) + '.mp4')
                sec = 0
                for j in range(0, len(robot_positions_new_sample) - sequence_length):
                    robot_positions__ = []
                    image_names__ = []
                    image_names_labels__ = []
                    for t in range(0, sequence_length):
                        robot_state = robot_positions_new_sample.iloc[j+t].values.flatten()
                        robot_positions__.append(robot_state[0:7])  # position = 0:3; pose = 0:
                        image_names__.append([data_dir + 'camera_1_video/rgb_video/sample_' + str(i) + '.mp4',((j+t) * frameRate)])  # [video location, frame]
                        image_names_labels__.append([data_dir + 'camera_1_video/rgb_video/sample_' + str(i) + '.mp4',((j+t+1) * frameRate)])  # [video location, frame]
                    # robot_positions.append([sublist for sublist in robot_positions__])
                    # robot_positions.append([item for sublist in robot_positions__ for item in sublist])  # original
                    # robot_positions.append([tuple(state) for state in robot_positions__])  # works for our priblem (but is 7 dimensional and theirs takes 5)
                    robot_positions.append([state for state in robot_positions__])  # this works for testing their system
                    image_names.append(image_names__)
                    image_names_labels.append(image_names_labels__)
            self.image_names = np.asarray(image_names)
            self.robot_positions = np.asarray(robot_positions)

            self.csv_ref = []

            self.process_data()

            self.save_data_to_map()


    def process_data(self):
        for j in tqdm(range(0, int(len(self.image_names) / 2))):
        # for j in tqdm(range(0, int(32*20))):
            raw = []
            vidcap = cv2.VideoCapture(self.image_names[j][0][0])

            for k in range(len(self.image_names[j])):
                tmp = Image.fromarray(self.getFrame(float(self.image_names[j][k][1]), vidcap), 'RGB')
                tmp = tmp.resize((self.image_resize_height, self.image_resize_width), Image.ANTIALIAS)
                tmp = np.fromstring(tmp.tobytes(), dtype=np.uint8)
                tmp = tmp.reshape((self.image_resize_height, self.image_resize_width, 3))
                tmp = tmp.astype(np.float32) / 255.0
                raw.append(tmp)
            raw = np.array(raw)

            ref = []
            ref.append(j)

            ### save png data
            if self.create_img == 1:
                for k in range(raw.shape[0]):
                    img = Image.fromarray(raw[k], 'RGB')
                    img.save(self.out_dir + '/image_batch_' + str(j) + '_' + str(k) + '.png')
                ref.append('image_batch_' + str(j) + '_*' + '.png')
            else:
                ref.append('')

            ### save np images
            np.save(self.out_dir + '/image_batch_' + str(j), raw)

            ### save np action
            # print("============================")
            # print(self.robot_positions[j][1:])
            np.save(self.out_dir + '/action_batch_' + str(j), self.robot_positions[j])

            ### save np states
            # print(self.robot_positions[j][0:-2])
            # print("============================")
            # np.save(self.out_dir + '/state_batch_' + str(j), self.robot_positions[j])
            np.save(self.out_dir + '/state_batch_' + str(j), self.robot_positions[j])  # original

            # save names for map file
            ref.append('image_batch_' + str(j) + '.npy')
            ref.append('action_batch_' + str(j) + '.npy')
            ref.append('state_batch_' + str(j) + '.npy')

            # Image used in prediction
            if self.create_img_prediction == 1:
                pred = []
                for k in range(len(self.image_names[j])):
                    img = Image.fromarray(self.getFrame(float(self.image_names[j][k][1]), vidcap), 'RGB')
                    img.save(self.out_dir + '/image_batch_pred_' + str(j) + '_' + str(k) + '.png')
                    pred.append(np.array(img))
                np.save(self.out_dir + '/image_batch_pred_' + str(j), np.array(pred))
                ref.append('image_batch_pred_' + str(j) + '_*' + '.png')
                ref.append('image_batch_pred_' + str(j) + '.npy')
            else:
                ref.append('')
                ref.append('')

            ### Append all file names for this sample to CSV file for training.
            self.csv_ref.append(ref)


    def getFrame(self, sec, vidcap):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        return image

    def save_data_to_map(self):
        self.logger.info("Writing the results into map file '{0}'".format('map.csv'))
        with open(self.out_dir + '/map.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            writer.writerow(['id', 'img_bitmap_path', 'img_np_path', 'action_np_path', 'state_np_path', 'img_bitmap_pred_path', 'img_np_pred_path'])
            for row in self.csv_ref:
                writer.writerow(row)


@click.command()
@click.option('--data_set_length', type=click.INT, default=500, help='size of dataset to format.')
@click.option('--data_dir', type=click.Path(exists=True), default='/home/user/Robotics/Data_sets/data_set_003/', help='Directory containing data.')
@click.option('--out_dir', type=click.Path(), default='/home/user/Robotics/Data_sets/CDNA_data/processed_custom_medium', help='Output directory of the converted data.')
@click.option('--sequence_length', type=click.INT, default=10, help='Sequence length, including context frames.')
@click.option('--image_original_width', type=click.INT, default=640, help='Original width of the images.')
@click.option('--image_original_height', type=click.INT, default=512, help='Original height of the images.')
@click.option('--image_original_channel', type=click.INT, default=3, help='Original channels amount of the images.')
@click.option('--image_resize_width', type=click.INT, default=64, help='Resize width of the the images.')
@click.option('--image_resize_height', type=click.INT, default=64, help='Resize height of the the images.')
@click.option('--state_action_dimension', type=click.INT, default=5, help='Dimension of the state and action.')
@click.option('--create_img', type=click.INT, default=0, help='Create the bitmap image along the numpy RGB values')
@click.option('--create_img_prediction', type=click.INT, default=1, help='Create the bitmap image used in the prediction phase')
def main(data_set_length, data_dir, out_dir, sequence_length, image_original_width, image_original_height, image_original_channel, image_resize_width, image_resize_height, state_action_dimension, create_img, create_img_prediction):
    data_formatter = DataFormatter(data_set_length, data_dir, out_dir, sequence_length, image_original_width, image_original_height, image_original_channel, image_resize_width, image_resize_height, state_action_dimension, create_img, create_img_prediction)

if __name__ == '__main__':
    main()

