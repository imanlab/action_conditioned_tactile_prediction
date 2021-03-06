#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Implementation in Chainer of https://github.com/tensorflow/models/tree/master/video_prediction
# ==============================================================================================

import math
import types
import random
import subprocess
from tqdm import tqdm
from math import floor, log
import numpy as np

try:
    import cupy
except:
    cupy = np
    pass

import chainer
from chainer import cuda
from chainer import variable
import chainer.functions as F
import chainer.links as L
from chainer.functions.connection import convolution_2d
from chainer import initializers
from chainer import serializers
from chainer.functions.math import square
from chainer.functions.activation import lstm

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import sys
import os
import time
import glob
import csv
import click
import logging

import matplotlib.pyplot as plt

# Amount to use when lower bounding Variables
RELU_SHIFT = 1e-12

# Kernel size for DNA and CDNA
DNA_KERN_SIZE = 3 ## could try 16 = 1 for every sensor? was 5

# =============================================
# Helpers functions used accross scripts (hlpe)
# =============================================
def concat_examples(batch):
    img_training_set, act_training_set, sta_training_set = [], [], []
    for idx in xrange(len(batch)):
        img_training_set.append(batch[idx][0])
        act_training_set.append(batch[idx][1])
        sta_training_set.append(batch[idx][2])

    img_training_set = np.array(img_training_set)
    act_training_set = np.array(act_training_set)
    sta_training_set = np.array(sta_training_set)

    # Split the actions, states and images into timestep
    act_training_set = np.split(ary=act_training_set, indices_or_sections=act_training_set.shape[1], axis=1)
    act_training_set = [np.squeeze(act, axis=1) for act in act_training_set]
    sta_training_set = np.split(ary=sta_training_set, indices_or_sections=sta_training_set.shape[1], axis=1)
    sta_training_set = [np.squeeze(sta, axis=1) for sta in sta_training_set]
    img_training_set = np.split(ary=img_training_set, indices_or_sections=img_training_set.shape[1], axis=1)
    # Reshape the img training set to a Chainer compatible tensor : batch x channel x height x width instead of Tensorflow's: batch x height x width x channel
    img_training_set = [np.rollaxis(np.squeeze(img, axis=1), 3, 1) for img in img_training_set]

    return np.array(img_training_set), np.array(act_training_set), np.array(sta_training_set)

def load_and_concat_examples(batch, data_dir):
    img_training_set, act_training_set, sta_training_set = [], [], []
    for idx in xrange(len(batch)):
        img_training_set.append(batch[idx][0])
        act_training_set.append(batch[idx][1])
        sta_training_set.append(batch[idx][2])

    images = []
    actions = []
    states = []

    for i in xrange(0, len(img_training_set)):
        images.append(np.float32([np.load(data_dir + "/" + str(image_name)) for image_name in np.load(img_training_set[i])]))
        actions.append(np.float32(np.load(act_training_set[i])))
        states.append(np.float32(np.load(sta_training_set[i])))

    img_training_set = np.asarray(images, dtype=np.float32)
    act_training_set = np.asarray(actions, dtype=np.float32)
    sta_training_set = np.asarray(states, dtype=np.float32)

    img_training_set = np.array(img_training_set)
    act_training_set = np.array(act_training_set)
    sta_training_set = np.array(sta_training_set)

    # Split the actions, states and images into timestep
    act_training_set = np.split(ary=act_training_set, indices_or_sections=act_training_set.shape[1], axis=1)
    act_training_set = [np.squeeze(act, axis=1) for act in act_training_set]
    sta_training_set = np.split(ary=sta_training_set, indices_or_sections=sta_training_set.shape[1], axis=1)
    sta_training_set = [np.squeeze(sta, axis=1) for sta in sta_training_set]
    img_training_set = np.split(ary=img_training_set, indices_or_sections=img_training_set.shape[1], axis=1)
    # Reshape the img training set to a Chainer compatible tensor : batch x channel x height x width instead of Tensorflow's: batch x height x width x channel
    img_training_set = [np.rollaxis(np.squeeze(img, axis=1), 3, 1) for img in img_training_set]

    return np.array(img_training_set), np.array(act_training_set), np.array(sta_training_set)

def scheduled_sample(ground_truth_x, generated_x, batch_size, num_ground_truth):
    """
        Sample batch with specified mix of ground truth and generated data points.
        e.g: the final matrix is a mix of vectors from the ground_truth (gt) and prediction (p)
            [gt1, gt2, gt3, gt4, gt5, gt6, gt7, gt8, gt9, gt10] = ground truth
            [p1, p2, p3, p4, p5, p6] = prediction
            [p1, gt2, gt3, gt4, p5, p6, gt7, gt8, gt9, gt10] = returns

        Args:
            ground_truth_x: tensor of ground-truth data point
            generated_x: tensor of generated data point
            batch_size: batch size
            num_ground_truth: number of ground-truth examples to include in batch
        Returns:
            New batch with num_ground_truth samples from ground_truth_x and the rest from generated_x
    """
    xp = chainer.cuda.get_array_module(generated_x.data)
    ground_truth_x = chainer.cuda.to_cpu(ground_truth_x)
    generated_x = chainer.cuda.to_cpu(generated_x.data)

    idx = np.arange(int(batch_size))
    np.random.shuffle(idx)
    ground_truth_idx = np.array(np.take(idx, np.arange(num_ground_truth)))
    generated_idx = np.array(np.take(idx, np.arange(num_ground_truth, int(batch_size))))

    reshaped_ground_truth_x = F.reshape(ground_truth_x, (int(batch_size), -1))
    reshaped_genetated_x = F.reshape(generated_x, (int(batch_size), -1))

    ground_truth_examps = np.take(reshaped_ground_truth_x.data, ground_truth_idx, axis=0)
    generated_examps = np.take(reshaped_genetated_x.data, generated_idx, axis=0)

    index_a = np.vstack((ground_truth_idx, np.zeros_like(ground_truth_idx)))
    index_b = np.vstack((generated_idx, np.ones_like(generated_idx)))
    ground_truth_generated_stacked = np.hstack((ground_truth_idx, generated_idx))
    ground_truth_generated_stacked_sorted = np.argsort(ground_truth_generated_stacked)
    order = np.hstack((index_a, index_b))[:, ground_truth_generated_stacked_sorted]

    stitched = []
    for i in xrange(len(order[0])):
        if order[1][i] == 0:
            pos = np.where(ground_truth_idx == i)
            stitched.append(ground_truth_examps[pos])
            continue
        else:
            pos = np.where(generated_idx == i)
            stitched.append(generated_examps[pos])
            continue
    stitched = np.array(stitched, dtype=np.float32)
    stitched = np.reshape(stitched, (ground_truth_x.shape[0], ground_truth_x.shape[1], ground_truth_x.shape[2], ground_truth_x.shape[3]))
    return xp.array(stitched)

def peak_signal_to_noise_ratio(true, pred):
    """
        Image quality metric based on maximal signal power vs. power of the noise

        Args:
            true: the ground truth image
            pred: the predicted image
        Returns:
            Peak signal to noise ratio (PSNR)
    """
    return 10.0 * F.log(1.0 / F.mean_squared_error(true, pred)) / log(10.0)


def broadcast_reshape(x, y, axis=0):
    """
        Reshape y to correspond to shape of x

        Args:
            x: the broadcasted
            y: the broadcastee
            axis: where the reshape will be performed
        Results:
            Output variable of same shape of x
    """
    y_shape = tuple([1] * axis + list(y.shape) +
                [1] * (len(x.shape) - axis - len(y.shape)))
    y_t = F.reshape(y, y_shape)
    y_t = F.broadcast_to(y_t, x.shape)
    return y_t

def broadcasted_division(x, y, axis=0):
    """
        Apply a division x/y where y is broadcasted to x to be able to complete the operation

        Args:
            x: the numerator
            y: the denominator
            axis: where the reshape will be performed
        Results:
            Output variable of same shape of x
    """
    y_t = broadcast_reshape(x, y, axis)
    return x / y_t

def broadcast_scale(x, y, axis=0):
    """
        Apply a multiplication x*y where y is broadcasted to x to be able to complete the operation

        Args:
            x: left hand operation
            y: right hand operation
            axis: where the reshape will be performed
        Resuts:
            Output variable of same shape of x
    """
    y_t = broadcast_reshape(x, y, axis)
    return x*y_t

# =============
# Chains (chns)
# =============

class LayerNormalizationConv2D(chainer.Chain):

    def __init__(self):
        super(LayerNormalizationConv2D, self).__init__()

        with self.init_scope():
            self.norm = L.LayerNormalization()


    """
        Apply a "layer normalization" on the result of a convolution

        Args:
            inputs: input tensor, 4D, batch x channel x height x width
        Returns:
            Output variable of shape (batch x channels x height x width)
    """
    def __call__(self, inputs):
        batch_size, channels, height, width = inputs.shape[0:4]
        inputs = F.reshape(inputs, (batch_size, -1))
        inputs = self.norm(inputs)
        inputs = F.reshape(inputs, (batch_size, channels, height, width))
        return inputs


# =============
# Models (mdls)
# =============


class BasicConvLSTMCell(chainer.Chain):
    """ Stateless convolutional LSTM, as seen in lstm_op.py from video_prediction model """

    def __init__(self, out_size=None, filter_size=5):
        super(BasicConvLSTMCell, self).__init__()

        with self.init_scope():
            # @TODO: maybe provide in channels because the concatenation
            self.conv = L.Convolution2D(4*out_size, (filter_size, filter_size), pad=filter_size/2)

        self.out_size = out_size
        self.filter_size = filter_size
        self.reset_state()

    def reset_state(self):
        self.c = None
        self.h = None

    def __call__(self, inputs, forget_bias=1.0):
        """Basic LSTM recurrent network cell, with 2D convolution connctions.

          We add forget_bias (default: 1) to the biases of the forget gate in order to
          reduce the scale of forgetting in the beginning of the training.

          It does not allow cell clipping, a projection layer, and does not
          use peep-hole connections: it is the basic baseline.

          Args:
            inputs: input Tensor, 4D, batch x channels x height x width
            forget_bias: the initial value of the forget biases.

          Returns:
             a tuple of tensors representing output and the new state.
        """
        # In Tensorflow: batch x height x width x channels
        # In Chainer: batch x channel x height x width
        # Create a state based on Finn's implementation
        xp = chainer.cuda.get_array_module(*inputs.data)
        if self.c is None:
            self.c = xp.zeros((inputs.shape[0], self.out_size, inputs.shape[2], inputs.shape[3]), dtype=inputs[0].data.dtype)
        if self.h is None:
            self.h = xp.zeros((inputs.shape[0], self.out_size, inputs.shape[2], inputs.shape[3]), dtype=inputs[0].data.dtype)

        #c, h = F.split_axis(state, indices_or_sections=2, axis=1)

        #inputs_h = np.concatenate((inputs, h), axis=1)
        inputs_h = F.concat((inputs, self.h), axis=1)

        # Parameters of gates are concatenated into one conv for efficiency
        #j_i_f_o = L.Convolution2D(in_channels=inputs_h.shape[1], out_channels=4*num_channels, ksize=(filter_size, filter_size), pad=filter_size/2)(inputs_h)
        j_i_f_o = self.conv(inputs_h)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        j, i, f, o = F.split_axis(j_i_f_o, indices_or_sections=4, axis=1)

        self.c = self.c * F.sigmoid(f + forget_bias) + F.sigmoid(i) * F.tanh(j)
        self.h = F.tanh(self.c) * F.sigmoid(o)

        #return new_h, np.concatenate((new_c, new_h), axis=1)
        #return new_h, F.concat((new_c, new_h), axis=1)
        return self.h

class StatelessCDNA(chainer.Chain):
    """
        Build convolutional lstm video predictor using CDNA
        * Because the CDNA does not keep states, it should be passed as a parameter if one wants to continue learning from previous states
    """

    def __init__(self, num_masks):
        super(StatelessCDNA, self).__init__()

        with self.init_scope():
            self.enc7 = L.Deconvolution2D(in_channels=64, out_channels=3, ksize=(1,1), stride=1)  # was in_channels=16  || 64
            self.cdna_kerns = L.Linear(in_size=None, out_size=DNA_KERN_SIZE * DNA_KERN_SIZE * num_masks)

        self.num_masks = num_masks

    def __call__(self, encs, hiddens, batch_size, prev_image, num_masks, color_channels):
        """
            Learn through StatelessCDNA.
            Args:
                encs: An array of computed transformation
                hiddens: An array of hidden layers
                batch_size: Size of mini batches
                prev_image: The image to transform
                num_masks: Number of masks to apply
                color_channels: Output color channels
            Returns:
                transformed: A list of masks to apply on the previous image
        """
        logger = logging.getLogger(__name__)

        enc0, enc1, enc2, enc3, enc4, enc5, enc6 = encs
        hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7 = hiddens

        img_height = prev_image.shape[2]
        img_width = prev_image.shape[3]

        # CDNA specific
        enc7 = self.enc7(enc6)
        enc7 = F.relu(enc7)
        transformed_list = list([F.sigmoid(enc7)])

        # CDNA specific
        # Predict kernels using linear function of last layer
        cdna_input = F.reshape(hidden5, (int(batch_size), -1))
        cdna_kerns = self.cdna_kerns(cdna_input)

        # Reshape and normalize
        # B x C x H x W => B x NUM_MASKS x 1 x H x W
        cdna_kerns = F.reshape(cdna_kerns, (int(batch_size), self.num_masks, 1, DNA_KERN_SIZE, DNA_KERN_SIZE))
        cdna_kerns = F.relu(cdna_kerns - RELU_SHIFT) + RELU_SHIFT
        norm_factor = F.sum(cdna_kerns, (2, 3, 4), keepdims=True)
        cdna_kerns = broadcasted_division(cdna_kerns, norm_factor)

        # Treat the color channel dimension as the batch dimension since the same
        # transformation is applied to each color channel.
        # Treat the batch dimension as the channel dimension so that
        # F.depthwise_convolution_2d can apply a different transformation to each sample.
        cdna_kerns = F.reshape(cdna_kerns, (int(batch_size), self.num_masks, DNA_KERN_SIZE, DNA_KERN_SIZE))
        cdna_kerns = F.transpose(cdna_kerns, (1, 0, 2, 3))
        # Swap the batch and channel dimension.
        prev_image = F.transpose(prev_image, (1, 0, 2, 3))

        # Transform the image.
        transformed = F.depthwise_convolution_2d(prev_image, cdna_kerns, stride=(1, 1), pad=DNA_KERN_SIZE/2)

        # Transpose the dimensions where they belong.
        transformed = F.reshape(transformed, (color_channels, int(batch_size), self.num_masks, img_height, img_width))
        transformed = F.transpose(transformed, (2, 1, 0, 3, 4))
        transformed = F.split_axis(transformed, indices_or_sections=self.num_masks, axis=0)
        transformed = [F.squeeze(t, axis=0) for t in transformed]

        transformed_list += transformed

        return transformed_list, enc7

class Model(chainer.Chain):
    """
        This Model wrap other models like CDNA, STP or DNA.
        It calls their training and get the generated images and states, it then compute the losses and other various parameters
    """

    def __init__(self, num_masks, is_cdna=True, use_state=True, scheduled_sampling_k=-1, num_frame_before_prediction=2, prefix=None):
        """
            Initialize a CDNA, STP or DNA through this 'wrapper' Model
            Args:
                is_cdna: if the model should be an extension of CDNA
                use_state: if the state should be concatenated
                scheduled_sampling_k: schedule sampling hyperparameter k
                num_frame_before_prediction: number of frame before prediction
                prefix: appended to the results to differentiate between training and validation
                learning_rate: learning rate
        """
        super(Model, self).__init__()

        with self.init_scope():
            self.enc0 = L.Convolution2D(32, (5, 5), stride=2, pad=2)  # was (5, 5)
            self.enc1 = L.Convolution2D(32, (3, 3), stride=2, pad=1)
            self.enc2 = L.Convolution2D(64, (3, 3), stride=2, pad=1)
            self.enc3 = L.Convolution2D(64, (1, 1), stride=1)

            self.enc4 = L.Deconvolution2D(128, (3, 3), stride=2, outsize=(16,16), pad=1)
            self.enc5 = L.Deconvolution2D(96, (3, 3), stride=2, outsize=(32,32), pad=1)
            self.enc6 = L.Deconvolution2D(64, (3, 3), stride=2, outsize=(64, 64), pad=1)

            self.lstm1 = BasicConvLSTMCell(32)
            self.lstm2 = BasicConvLSTMCell(32)
            self.lstm3 = BasicConvLSTMCell(64)
            self.lstm4 = BasicConvLSTMCell(64)
            self.lstm5 = BasicConvLSTMCell(128)
            self.lstm6 = BasicConvLSTMCell(64)
            self.lstm7 = BasicConvLSTMCell(32)

            self.norm_enc0 = LayerNormalizationConv2D()
            self.norm_enc6 = LayerNormalizationConv2D()
            self.hidden1 = LayerNormalizationConv2D()
            self.hidden2 = LayerNormalizationConv2D()
            self.hidden3 = LayerNormalizationConv2D()
            self.hidden4 = LayerNormalizationConv2D()
            self.hidden5 = LayerNormalizationConv2D()
            self.hidden6 = LayerNormalizationConv2D()
            self.hidden7 = LayerNormalizationConv2D()

        # self.ops = [
        #     [self.enc0, self.norm_enc0],
        #     [self.lstm1, self.hidden1, ops_save("hidden1"), self.lstm2, self.hidden2, ops_save("hidden2"), self.enc1],
        #     [self.lstm3, self.hidden3, ops_save("hidden3"), self.lstm4, self.hidden4, ops_save("hidden4"), self.enc2],
        #     [ops_smear(use_state), self.enc3],
        #     [self.lstm5, self.hidden5, ops_save("hidden5"), self.enc4],
        #     [self.lstm6, self.hidden6, ops_save("hidden6"), ops_skip_connection(1), self.enc5],
        #     [self.lstm7, self.hidden7, ops_save("hidden7"), ops_skip_connection(0), self.enc6, self.norm_enc6]
        # ]


        # : NEW VERSION :
        # with self.init_scope():
        #     self.enc0 = L.Convolution2D(in_channels=None, out_channels=8, ksize=(3, 3), stride=1, pad=1)
        #     self.enc1 = L.Convolution2D(in_channels=None, out_channels=8, ksize=(3, 3), stride=1, pad=1)
        #     self.enc2 = L.Convolution2D(in_channels=None, out_channels=8, ksize=(3, 3), stride=1, pad=1)
        #     self.enc3 = L.Convolution2D(in_channels=None, out_channels=8, ksize=(3, 3), stride=1, pad=1)
        #
        #     # self.enc4 = L.Deconvolution2D(in_channels=None, out_channels=8, ksize=(3, 3), stride=1, outsize=(4, 4), pad=2)
        #     # self.enc5 = L.Deconvolution2D(in_channels=None, out_channels=8, ksize=(3, 3), stride=1, outsize=(4, 4), pad=2)
        #     # self.enc6 = L.Deconvolution2D(in_channels=None, out_channels=8, ksize=(3, 3), stride=1, outsize=(4, 4), pad=2)
        #
        #     self.lstm1 = BasicConvLSTMCell(8)
        #     self.lstm2 = BasicConvLSTMCell(8)
        #     self.lstm3 = BasicConvLSTMCell(8)
        #     self.lstm4 = BasicConvLSTMCell(8)
        #     self.lstm5 = BasicConvLSTMCell(8)
        #     self.lstm6 = BasicConvLSTMCell(8)
        #     self.lstm7 = BasicConvLSTMCell(8)
        #
        #     self.norm_enc0 = LayerNormalizationConv2D()
        #     self.norm_enc6 = LayerNormalizationConv2D()
        #     self.hidden1 = LayerNormalizationConv2D()
        #     self.hidden2 = LayerNormalizationConv2D()
        #     self.hidden3 = LayerNormalizationConv2D()
        #     self.hidden4 = LayerNormalizationConv2D()
        #     self.hidden5 = LayerNormalizationConv2D()
        #     self.hidden6 = LayerNormalizationConv2D()
        #     self.hidden7 = LayerNormalizationConv2D()
        #
            self.masks = L.Deconvolution2D(num_masks+1, (1, 1), stride=1)

            self.current_state = L.Linear(6)  # change this from 5 to 7?

            model = None
            if is_cdna:
                model = StatelessCDNA(num_masks)
            if model is None:
                raise ValueError("No network specified")
            else:
                self.model = model

        self.num_masks = num_masks
        self.use_state = use_state
        self.scheduled_sampling_k = scheduled_sampling_k
        self.num_frame_before_prediction = num_frame_before_prediction
        self.prefix = prefix

        self.loss = 0.0
        self.psnr_all = 0.0
        self.summaries = []
        self.conv_res = []

        # Condition ops callback
        def ops_smear(use_state):
            def ops(args):
                x = args.get("x")
                if use_state:
                    state_action = args.get("state_action")
                    batch_size = args.get("batch_size")

                    smear = F.reshape(state_action, (int(batch_size), int(state_action.shape[1]), 1, 1))
                    smear = F.tile(smear, (1, 1, int(x.shape[2]), int(x.shape[3])))
                    x = F.concat((x, smear), axis=1)  # Previously axis=3 but our channel is on axis=1? ok
                return x
            return ops

        def ops_skip_connection(enc_idx):
            def ops(args):
                x = args.get("x")
                enc = args.get("encs")[enc_idx]
                # Skip connection (current input + target enc)
                x = F.concat((x, enc), axis=1)  # Previously axis=3 but our channel is on axis=1? ok!
                return x
            return ops

        def ops_save(name):
            def ops(args):
                x = args.get("x")
                save_map = args.get("map")
                save_map[name] = x
                return x
            return ops

        def ops_get(name):
            def ops(args):
                save_map = args.get("map")
                return save_map[name]
            return ops

        # Create an executable array containing all the transformations
        # self.ops = [
        #     [self.enc0, self.norm_enc0],
        #     [self.lstm1, self.hidden1, ops_save("hidden1"), self.lstm2, self.hidden2, ops_save("hidden2"), self.enc1],
        #     [self.lstm3, self.hidden3, ops_save("hidden3"), self.lstm4, self.hidden4, ops_save("hidden4"), self.enc2],
        #     [ops_smear(use_state), self.enc3],
        #     [self.lstm5, self.hidden5, ops_save("hidden5")],
        #     [self.lstm6, self.hidden6, ops_save("hidden6"), ops_skip_connection(1)],
        #     [self.lstm7, self.hidden7, ops_save("hidden7"), ops_skip_connection(0), self.norm_enc6]
        # ]

        # original one:
        self.ops = [
            [self.enc0, self.norm_enc0],
            [self.lstm1, self.hidden1, ops_save("hidden1"), self.lstm2, self.hidden2, ops_save("hidden2"), self.enc1],
            [self.lstm3, self.hidden3, ops_save("hidden3"), self.lstm4, self.hidden4, ops_save("hidden4"), self.enc2],
            [ops_smear(use_state), self.enc3],
            [self.lstm5, self.hidden5, ops_save("hidden5"), self.enc4],
            [self.lstm6, self.hidden6, ops_save("hidden6"), ops_skip_connection(1), self.enc5],
            [self.lstm7, self.hidden7, ops_save("hidden7"), ops_skip_connection(0), self.enc6, self.norm_enc6]
        ]

    def reset_state(self):
        """
            Reset the gradient of this model, but also the specific model
        """
        self.loss = 0.0
        self.psnr_all = 0.0
        self.summaries = []
        self.conv_res = []
        self.lstm1.reset_state()
        self.lstm2.reset_state()
        self.lstm3.reset_state()
        self.lstm4.reset_state()
        self.lstm5.reset_state()
        self.lstm6.reset_state()
        self.lstm7.reset_state()

    # def __call__(self, x, iter_num=-1.0):
    def __call__(self, images, actions, states, iter_num=-1.0):
        """
            Calls the training process
            Args:
                x: an array containing an array of:
                    images: an array of Tensor of shape batch x channels x height x width
                    actions: an array of Tensor of shape batch x action
                    states: an array of Tensor of shape batch x state
                iter_num: iteration (epoch) index
            Returns:
                loss, all the peak signal to noise ratio, summaries
        """
        logger = logging.getLogger(__name__)
        batch_size, color_channels, img_height, img_width = images[0].shape[0:4]

        # Generated robot states and images
        gen_states, gen_images = [], []
        current_state = states[0]

        # When validation/test, disable schedule sampling
        if not chainer.config.train or self.scheduled_sampling_k == -1:
            feedself = True
        else:
            # Scheduled sampling, inverse sigmoid decay
            # Calculate number of ground-truth frames to pass in.
            num_ground_truth = np.int32(
                np.round(np.float32(batch_size) * (self.scheduled_sampling_k / (self.scheduled_sampling_k + np.exp(iter_num / self.scheduled_sampling_k))))
            )
            feedself = False

        for image, action in zip(images[:-1], actions[1:]):  # was: actions[:-1]
            # Reuse variables after the first timestep
            reuse = bool(gen_images)

            done_warm_start = len(gen_images) > self.num_frame_before_prediction - 1
            if feedself and done_warm_start:
                # Feed in generated image
                prev_image = gen_images[-1]
            elif done_warm_start:
                # Scheduled sampling
                prev_image = scheduled_sample(image, gen_images[-1], batch_size, num_ground_truth)
                prev_image = variable.Variable(prev_image)
            else:
                # Always feed in ground_truth
                prev_image = variable.Variable(image)

            # Predicted state is always fed back in
            state_action = F.concat((action, current_state), axis=1)

            """ Execute the ops array of transformations """
            # If an ops has a name of "ops" it means it's a custom ops
            encs = []
            maps = {}
            x = prev_image
            for i in xrange(len(self.ops)):
                for j in xrange(len(self.ops[i])):
                    op = self.ops[i][j]
                    if isinstance(op, types.FunctionType):
                        # Only these values are use now in the ops callback
                        x = op({
                            "x": x,
                            "encs": encs,
                            "map": maps,
                            "state_action": state_action,
                            "batch_size": batch_size
                        })
                    else:
                        x = op(x)
                # ReLU at the end of each transformation
                x = F.relu(x)
                # At the end of j iteration = completed a enc transformation
                encs.append(x)

            # Extract the variables
            hiddens = [
                maps.get("hidden1"), maps.get("hidden2"), maps.get("hidden3"), maps.get("hidden4"),
                maps.get("hidden5"), maps.get("hidden6"), maps.get("hidden7")
            ]
            enc0, enc1, enc2, enc3, enc4, enc5, enc6 = encs
            hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7 = hiddens

            """ Specific model transformations """
            transformed, enc7 = self.model(
                encs, hiddens,
                batch_size, prev_image, self.num_masks, int(color_channels)
            )  # StatelessCDNA
            encs.append(enc7)

            """ Compositing Masks """
            masks = self.masks(enc6)
            masks = F.relu(masks)
            masks = F.reshape(masks, (-1, self.num_masks + 1))
            masks = F.softmax(masks)
            masks = F.reshape(masks, (int(batch_size), self.num_masks+1, int(img_height), int(img_width)))  # Previously num_mask at the end, but our channels are on axis=1? ok!
            mask_list = F.split_axis(masks, indices_or_sections=self.num_masks+1, axis=1)  # Previously axis=3 but our channels are on axis=1 ?

            output = broadcast_scale(prev_image, mask_list[0])
            for layer, mask in zip(transformed, mask_list[1:]):
                output += broadcast_scale(layer, mask, axis=0)
            gen_images.append(output)

            current_state = self.current_state(state_action)
            gen_states.append(current_state)

        # End of transformations
        self.conv_res = encs

        # L2 loss, PSNR for eval
        loss, psnr_all = 0.0, 0.0
        summaries = []
        for i, x, gx in zip(range(len(gen_images)), images[self.num_frame_before_prediction:], gen_images[self.num_frame_before_prediction - 1:]):
            x = variable.Variable(x)
            recon_cost = F.mean_squared_error(x, gx)
            psnr_i = peak_signal_to_noise_ratio(x, gx)
            psnr_all += psnr_i
            summaries.append(self.prefix + '_recon_cost' + str(i) + ': ' + str(recon_cost.data))
            summaries.append(self.prefix + '_psnr' + str(i) + ': ' + str(psnr_i.data))
            loss += recon_cost
            # print(recon_cost.data)

        for i, state, gen_state in zip(range(len(gen_states)), states[self.num_frame_before_prediction:], gen_states[self.num_frame_before_prediction - 1:]):
            state = variable.Variable(state)
            state_cost = F.mean_squared_error(state, gen_state) * 1e-4
            summaries.append(self.prefix + '_state_cost' + str(i) + ': ' + str(state_cost.data))
            loss += state_cost

        summaries.append(self.prefix + '_psnr_all: ' + str(psnr_all.data if isinstance(psnr_all, variable.Variable) else psnr_all))
        self.psnr_all = psnr_all

        self.loss = loss = loss / np.float32(len(images) - self.num_frame_before_prediction)
        summaries.append(self.prefix + '_loss: ' + str(loss.data if isinstance(loss, variable.Variable) else loss))

        self.summaries = summaries
        self.gen_images = gen_images

        return self.loss


class ModelTrainer():
    def __init__(self, learning_rate, gpu, num_iterations, schedsamp_k, use_state, context_frames, num_masks, batch_size):
        self.gpu = gpu
        self.num_masks = num_masks
        self.use_state = use_state
        self.batch_size = batch_size
        self.schedsamp_k = schedsamp_k
        self.learning_rate = learning_rate
        self.context_frames = context_frames
        self.num_iterations = num_iterations

    def create_model(self):
        # create the model
        self.training_model = Model(num_masks=self.num_masks, is_cdna=True, use_state=self.use_state, scheduled_sampling_k=self.schedsamp_k, num_frame_before_prediction=self.context_frames, prefix='train')

    def gpu_support(self):
        # Enable GPU support if defined
        if self.gpu > -1:
            chainer.cuda.get_device_from_id(self.gpu).use()
            self.training_model.to_gpu()
            self.xp = cupy
            self.xp.cuda.alloc_pinned_memory(5*1024**3)  # 8gb
            print(self.xp.get_default_memory_pool().total_bytes())
        else:
            self.xp = np

    def create_opimiser(self):
        # Create the optimizers for the models
        self.optimizer = chainer.optimizers.Adam(alpha=self.learning_rate)
        self.optimizer.setup(self.training_model)

    def load_data_from_git(self):
        ''' Save the current GIT commit corresponding to the current training.
            When predicting or visualizing the model, change the working directory to the GIT snapshot.
            This way, instead of copying the files into the model folder, we use GIT functionality to preserve the training files. '''
        current_version = None
        try:
            subprocess.check_call(['git', 'status'])
            def git_exec(args):
                process = subprocess.Popen(['git'] + args, stdout=subprocess.PIPE)
                res = process.communicate()[0].rstrip().strip()
                return res

            current_version = git_exec(['rev-parse', '--abbrev-ref', 'HEAD']) + '\n' + git_exec(['rev-parse', 'HEAD'])
        except:
            pass

    def train(self, train_iter, valid_iter, logger, images_training, validation_interval, save_interval, output_dir, model_suffix_dir, training_suffix, validation_suffix, state_suffix, current_version, generator_type, data_dir):
        # Run training
        # As per Finn's implementation, one epoch is run on one batch size, randomly, but never more than once.
        # At the end of the queue, if the epochs len is not reach, the queue is generated again.
        local_losses = []
        local_psnr_all = []
        local_losses_valid = []
        local_psnr_all_valid = []

        global_losses = []
        global_psnr_all = []
        global_losses_valid = []
        global_psnr_all_valid = []

        summaries, summaries_valid = [], []
        training_queue = []
        validation_queue = []
        start_time = None
        stop_time = None

        save_dir = output_dir + '/' + model_suffix_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            # Save the version of the code
            f = open(save_dir + '/version', 'w')
            f.write(current_version + '\n')
            f.close()
        serializers.save_npz(output_dir + '/' + model_suffix_dir + '/' + training_suffix + '-initialisation', self.training_model)

        progress_bar = tqdm(range(0, self.num_iterations), total=(self.num_iterations/self.batch_size))
        for itr in progress_bar:
            epoch = train_iter.epoch
            batch = train_iter.next()

            if generator_type == 0:
                img_training_set, act_training_set, sta_training_set = concat_examples(batch)
            elif generator_type == 1:
                img_training_set, act_training_set, sta_training_set = load_and_concat_examples(batch, data_dir=data_dir)

            # Perform training
            # logger.info("Global iteration: {}".format(str(itr+1)))
            # loss = self.training_model(cupy.asarray(img_training_set, dtype=cupy.float32),
                                        # cupy.asarray(act_training_set, dtype=cupy.float32),
                                        # cupy.asarray(sta_training_set, dtype=cupy.float32), epoch)

            if start_time is None:
                start_time = time.time()

            self.optimizer.update(self.training_model,
                cupy.asarray(img_training_set, dtype=cupy.float32),
                cupy.asarray(act_training_set, dtype=cupy.float32),
                cupy.asarray(sta_training_set, dtype=cupy.float32),
                itr)

            loss = self.training_model.loss
            psnr_all = self.training_model.psnr_all
            summaries = self.training_model.summaries

            loss_data_cpu = chainer.cuda.to_cpu(loss.data)
            psnr_data_cpu = chainer.cuda.to_cpu(psnr_all.data)

            local_losses.append(loss_data_cpu)
            local_psnr_all.append(psnr_data_cpu)
            self.training_model.reset_state()

            # logger.info("Begining training for mini-batch {0}/{1} of epoch {2}, LOSS={3}".format(str(train_iter.current_position), str(len(images_training)), str(epoch+1), str(loss.data)))

            progress_bar.set_description("epoch: {}, ".format(epoch) + "loss: {:.4f}, ".format(float(loss.data)) + "mean loss: {:.4f}, ".format(float(np.mean(np.array(local_losses)))) + "mean psmr: {:.4f}".format(float(np.mean(np.array(local_psnr_all)))))
            progress_bar.update()

            loss, psnr_all, loss_data_cpu, psnr_data_cpu = None, None, None, None

            if train_iter.is_new_epoch:
                stop_time = time.time()
                logger.info("[TRAIN] Epoch #: {}".format(epoch+1))
                logger.info("[TRAIN] Epoch elapsed time: {}".format(stop_time-start_time))

                local_losses = np.array(local_losses)
                local_psnr_all = np.array(local_psnr_all)
                global_losses.append([local_losses.mean(), local_losses.std(), local_losses.min(), local_losses.max(), np.median(local_losses)])
                global_psnr_all.append([local_psnr_all.mean(), local_psnr_all.std(), local_psnr_all.min(), local_psnr_all.max(), np.median(local_psnr_all)])

                logger.info("[TRAIN] epoch loss: {}".format(local_losses.mean()))
                logger.info("[TRAIN] epoch psnr: {}".format(local_psnr_all.mean()))

                local_losses, local_psnr_all = [], []
                start_time, stop_time = None, None

            # if train_iter.is_new_epoch:  # and epoch+1 % validation_interval == 0:
            #     print("HERERERERERERERERERERER")

                start_time = time.time()
                for batch in valid_iter:
                    if generator_type == 0:
                        img_validation_set, act_validation_set, sta_validation_set = concat_examples(batch)
                    elif generator_type == 1:
                        img_validation_set, act_validation_set, sta_validation_set = load_and_concat_examples(batch, data_dir=data_dir)

                    # logger.info("Begining validation for mini-batch of epoch {0}".format(str(epoch+1)))

                    # Run through validation set
                    # loss_valid, psnr_all_valid, summaries_valid = validation_model(img_validation_set, act_validation_set, sta_validation_set, epoch, schedsamp_k, use_state, num_masks, context_frames)
                    with chainer.using_config('train', False):
                        loss_valid = self.training_model(cupy.asarray(img_validation_set, dtype=cupy.float32),
                                                        cupy.asarray(act_validation_set, dtype=cupy.float32),
                                                        cupy.asarray(sta_validation_set, dtype=cupy.float32), itr)

                    psnr_all_valid = self.training_model.psnr_all
                    summaries_valid = self.training_model.summaries

                    loss_valid_data_cpu = chainer.cuda.to_cpu(loss_valid.data)
                    psnr_all_valid_data_cpu = chainer.cuda.to_cpu(psnr_all_valid.data)

                    local_losses_valid.append(loss_valid_data_cpu)
                    local_psnr_all_valid.append(psnr_all_valid_data_cpu)
                    self.training_model.reset_state()

                    loss_valid, psnr_all_valid, loss_valid_data_cpu, psnr_all_valid_data_cpu = None, None, None, None
                stop_time = time.time()
                logger.info("[VALID] Epoch #: {}".format(epoch+1))
                logger.info("[VALID] epoch elapsed time: {}".format(stop_time-start_time))

                local_losses_valid = np.array(local_losses_valid)
                local_psnr_all_valid = np.array(local_psnr_all_valid)
                global_losses_valid.append([local_losses_valid.mean(), local_losses_valid.std(), local_losses_valid.min(), local_losses_valid.max(), np.median(local_losses_valid)])
                global_psnr_all_valid.append([local_psnr_all_valid.mean(), local_psnr_all_valid.std(), local_psnr_all_valid.min(), local_psnr_all_valid.max(), np.median(local_psnr_all_valid)])

                logger.info("[VALID] epoch loss: {}".format(local_losses_valid.mean()))
                logger.info("[VALID] epoch psnr: {}".format(local_psnr_all_valid.mean()))

                local_losses_valid, local_psnr_all_valid = [], []
                start_time, stop_time = None, None

                valid_iter.reset()
                self.training_model.reset_state()

            if train_iter.is_new_epoch and epoch % save_interval == 0:
                #if epoch % save_interval == 0:
                logger.info('Saving model')

                save_dir = output_dir + '/' + model_suffix_dir
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    # Save the version of the code
                    f = open(save_dir + '/version', 'w')
                    f.write(current_version + '\n')
                    f.close()

                serializers.save_npz(save_dir + '/' + training_suffix + '-' + str(epoch), self.training_model)
                # serializers.save_npz(save_dir + '/' + validation_suffix + '-' + str(epoch), validation_model)
                serializers.save_npz(save_dir + '/' + state_suffix + '-' + str(epoch), self.optimizer)
                np.save(save_dir + '/' + training_suffix + '-global_losses', np.array(global_losses))
                np.save(save_dir + '/' + training_suffix + '-global_psnr_all', np.array(global_psnr_all))
                np.save(save_dir + '/' + training_suffix + '-global_losses_valid', np.array(global_losses_valid))
                np.save(save_dir + '/' + training_suffix + '-global_psnr_all_valid', np.array(global_psnr_all_valid))

            #for summ in summaries:
                #logger.info(summ)
            summaries = []
            #for summ_valid in summaries_valid:
                #logger.info(summ_valid)
            summaries_valid = []

    def train_single_epoch(self):
        pass

    def train_single_batch(self):
        pass


class DataGenerator():
    def __init__(self, batch_size, logger, data_dir, train_val_split, generator_type):
        data_map = []
        with open(data_dir + '/map.csv', 'rb') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                data_map.append(row)

        if len(data_map) <= 1: # empty or only header
            logger.error("No file map found")
            exit()

        # Load the images, actions and states
        images = []
        actions = []
        states = []

        if generator_type == 0:
            for i in tqdm(xrange(1, len(data_map))):  # Exclude the header
                images.append(np.float32(np.load(data_dir + '/' + data_map[i][2])))
                actions.append(np.float32(np.load(data_dir + '/' + data_map[i][3])))
                states.append(np.float32(np.load(data_dir + '/' + data_map[i][4])))

                images = np.asarray(images, dtype=np.float32)
                actions = np.asarray(actions, dtype=np.float32)
                states = np.asarray(states, dtype=np.float32)
        elif generator_type == 1:
            for i in tqdm(xrange(1, len(data_map))):  # Exclude the header
                images.append(str(data_dir + '/' + data_map[i][2]))
                actions.append(str(data_dir + '/' + data_map[i][0]))
                states.append(str(data_dir + '/' + data_map[i][0]))

        train_val_split_index = int(np.floor(train_val_split * len(images)))
        self.images_training = np.asarray(images[:train_val_split_index])
        self.actions_training = np.asarray(actions[:train_val_split_index])
        self.states_training = np.asarray(states[:train_val_split_index])

        self.images_validation = np.asarray(images[train_val_split_index:])
        self.actions_validation = np.asarray(actions[train_val_split_index:])
        self.states_validation = np.asarray(states[train_val_split_index:])

        logger.info('Data set contain {0}, {1} will be use for training and {2} will be use for validation'.format(len(images)-1, train_val_split_index, len(images)-1-train_val_split_index))

        # Group the images, actions and states
        self.grouped_set_training = []
        self.grouped_set_validation = []
        for idx in xrange(len(self.images_training)):
            group = []
            group.append(self.images_training[idx])
            group.append(self.actions_training[idx])
            group.append(self.states_training[idx])
            self.grouped_set_training.append(group)
        for idx in xrange(len(self.images_validation)):
            group = []
            group.append(self.images_validation[idx])
            group.append(self.actions_validation[idx])
            group.append(self.states_validation[idx])
            self.grouped_set_validation.append(group)

    def return_Data(self):
        return [self.images_training, self.actions_training, self.states_training, self.images_validation, self.actions_validation, self.states_validation,
        self.grouped_set_training, self.grouped_set_validation]

    def train_iter(self, batch_size):
        #train_iter = chainer.iterators.SerialIterator(grouped_set_training, batch_size)
        train_iter = chainer.iterators.SerialIterator(self.grouped_set_training, batch_size, repeat=True, shuffle=True)
        valid_iter = chainer.iterators.SerialIterator(self.grouped_set_validation, batch_size, repeat=False, shuffle=True)
        return train_iter, valid_iter


# =================================================
# Main entry point of the training processes (main)
# =================================================
@click.command()
@click.option('--learning_rate', type=click.FLOAT, default=0.001, help='The base learning rate of the generator. was 0.001')
@click.option('--gpu', type=click.INT, default=0, help='ID of the gpu(s) to use')
@click.option('--batch_size', type=click.INT, default=8, help='Batch size for training.')
@click.option('--num_iterations', type=click.INT, default=int(50*39764), help='Number of training iterations. Number of epoch is: num_iterations/batch_size.')  # 50*5654 1/4 of 1000 dataset sample was 14204
@click.option('--data_dir', type=click.Path(exists=True), default='/home/user/Robotics/Data_sets/box_only_dataset/train_image_dataset_10c_10h_64_universal', help='Directory containing data.')
# @click.option('--data_dir', type=click.Path(exists=True), default='/home/user/Robotics/Data_sets/PRI/single_object_purple/train_formatted', help='Directory containing data.')
@click.option('--train_val_split', type=click.FLOAT, default=0.95, help='The percentage of data to use for the training set, vs. the validation set.')
@click.option('--schedsamp_k', type=click.FLOAT, default=900.0, help='The k parameter for schedules sampling. -1 for no scheduled sampling.')
@click.option('--use_state', type=click.INT, default=1, help='Whether or not to give the state+action to the model.')
@click.option('--context_frames', type=click.INT, default=10, help='Number of frames before predictions.')
@click.option('--num_masks', type=click.INT, default=10, help='Number of masks, usually 1 for DNA, 10 for CDNA, STP.')
@click.option('--validation_interval', type=click.INT, default=1, help='How often to run a batch through the validation model')
@click.option('--save_interval', type=click.INT, default=1, help='How often to save a model checkpoint (set to 50 originally)')
@click.option('--output_dir', type=click.Path(), default='/home/user/Robotics/tactile_prediction/tactile_prediction/uninversal_trainer/CDNA_chainer/saved_models', help='Directory for model checkpoints.')
# @click.option('--output_dir', type=click.Path(), default='/home/user/Robotics/CDNA/models/slip_detection/models', help='Directory for model checkpoints.')
@click.option('--current_version', type=click.STRING, default="001", help='Model Version for saving and logging')
@click.option('--generator_type', type=click.INT, default=1, help='0 = Load full data first (for small datasets), 1 = Load data on fly as training (for large datasets)')
def main(learning_rate, gpu, batch_size, num_iterations, data_dir, train_val_split, schedsamp_k, use_state, context_frames, num_masks, validation_interval, save_interval, output_dir,current_version, generator_type):
    logger = logging.getLogger(__name__)
    logger.info('Training the model')
    logger.info('Model: {}'.format("CDNA"))
    logger.info('GPU: {}'.format(gpu))
    logger.info('# Minibatch-size: {}'.format(batch_size))
    logger.info('# Num iterations: {}'.format(num_iterations))
    logger.info('# epoch: {}'.format(round(num_iterations/batch_size)))

    model_suffix_dir = "{0}-{1}-{2}".format(time.strftime("%Y%m%d-%H%M%S"), "CDNA", batch_size)
    training_suffix = "{0}".format('training')
    validation_suffix = "{0}".format('validation')
    state_suffix = "{0}".format('state')

    ### Initialise the model
    model_trainer = ModelTrainer(learning_rate, gpu, num_iterations, schedsamp_k, use_state, context_frames, num_masks, batch_size)    # Generate model trainer.
    model_trainer.create_model()                        # Create the model.
    model_trainer.create_opimiser()                     # Create the optimizers for the models.
    model_trainer.gpu_support()                         # Enable GPU if required.

    ### Generate the training data
    logger.info("Fetching the models and inputs")
    data_generator = DataGenerator(batch_size, logger, data_dir, train_val_split, generator_type)
    train_iter, valid_iter = data_generator.train_iter(batch_size)
    data = data_generator.return_Data()
    ### Train model:
    model_trainer.train(train_iter, valid_iter, logger, data[0], validation_interval, save_interval, output_dir, model_suffix_dir, training_suffix, validation_suffix, state_suffix, current_version, generator_type, data_dir)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt, stream=sys.stdout)
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
