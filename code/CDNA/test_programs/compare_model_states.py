#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Predict the next n frames from a trained model
# ==============================================

import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F

from custom_model_1 import Model
from custom_model_1 import concat_examples

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


model_dir = '20210113-163814-CDNA-20'
models_dir = 'models'
model_name = 'training-67'
model_type ='CDNA'
schedsamp_k = -1
context_frames = 2
use_state = 1
num_masks = 10

path = models_dir + '/' + model_dir
model = Model(
    num_masks=num_masks,
    is_cdna=model_type,
    use_state=use_state,
    scheduled_sampling_k=schedsamp_k,
    num_frame_before_prediction=context_frames,
    prefix='predict'
)

trained_model = chainer.serializers.load_npz(str(path + '/' + model_name), model)
print(trained_model)


global_losses = np.load("/home/user/Robotics/CDNA/CDNA/test_programs/models/20210113-163814-CDNA-20/training-global_losses.npy")
global_losses_valid = np.load("/home/user/Robotics/CDNA/CDNA/test_programs/models/20210113-163814-CDNA-20/training-global_losses_valid.npy")
global_losses_psnr = np.load("/home/user/Robotics/CDNA/CDNA/test_programs/models/20210113-163814-CDNA-20/training-global_psnr_all.npy")