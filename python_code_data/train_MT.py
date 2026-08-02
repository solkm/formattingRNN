#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the MT-like module (MT_broadInSharpOut_withR task).

@author: Sol
"""
from pathlib import Path
import os
root_dir = Path(__file__).parent
os.chdir(str(root_dir))

import numpy as np
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
import matplotlib.pyplot as plt
plt.ion()
import tensorflow as tf

from DotsTasks import MT_broadInSharpOut_withR
from psychrnn.backend.models.basicLinearOutput import Basic_linOut

# TRAIN
SAVE_WEIGHTS_PATH = './model_weights'
model_name = 'MTbroadsharp_m10'
N_rec = 150
rec_noise = 0.2
in_noise = 0.4
N_trainbatch = 200
coh = [0.6]

task = MT_broadInSharpOut_withR(T=1200, N_batch=N_trainbatch, N_rec=N_rec, in_noise=in_noise, coh=coh, k_in=0.3, k_out=0.8, catchP=0.03)

L2_in, L2_rec, L2_out, L2_FR = 0.01, 0, 0.01, 0.004
L1_in, L1_rec, L1_out = 0, 0, 0
dale = 0.8

network_params = task.get_task_params()
network_params['name'] = model_name
network_params['N_rec'] = N_rec
network_params['rec_noise'] = rec_noise
network_params['autapses'] = False
network_params['dale_ratio'] = dale
network_params['L2_in'] = L2_in
network_params['L2_rec'] = L2_rec
network_params['L2_out'] = L2_out
network_params['L2_firing_rate'] = L2_FR
network_params['L1_in'] = L1_in
network_params['L1_rec'] = L1_rec
network_params['L1_out'] = L1_out
network_params['transfer_function'] = tf.nn.relu
network_params['loss_function'] = 'custom_loss_function'

if network_params['loss_function'] == 'custom_loss_function':

    def loss_function_MSEandRewBias(predictions, y, output_mask):

        mse = tf.cast(tf.reduce_mean(input_tensor=tf.square(output_mask * (predictions - y))), tf.float32)

        rewRads = tf.math.atan2(y[:, -1, 1], y[:, -1, 0])
        rewEstRads = tf.math.atan2(predictions[:, -1, 1], predictions[:, -1, 0])
        distFromRew_rad = tf.abs(rewRads - rewEstRads)
        distFromRew_rad = tf.math.minimum(distFromRew_rad, 2*np.pi - distFromRew_rad)
        rewrad = 0.3 * tf.reduce_mean(input_tensor=tf.square(tf.cast(distFromRew_rad, tf.float32)))

        return mse + rewrad

    network_params['custom_loss_function'] = loss_function_MSEandRewBias

model = Basic_linOut(network_params)
temp_weights = model.get_weights()
for k,v in temp_weights.items():
    network_params[k] = v
model.destruct()

# fix microstim input weights at 0
N_out = task.N_out
N_in = task.N_in
W_in_fixed = np.zeros((N_rec, N_in))
W_rec_fixed = np.zeros((N_rec, N_rec))
W_out_fixed = np.zeros((N_out, N_rec))

network_params['W_in'][:, -1] = 0
W_in_fixed[:, -1] = 1
train_params = {}
train_params['fixed_weights'] = {
    'W_in': W_in_fixed,
    'W_rec': W_rec_fixed,
    'W_out': W_out_fixed
}

train_params['training_iters'] = 200000
train_params['learning_rate'] = 0.003

# Save weights during training
train_params['training_weights_path'] = f'{SAVE_WEIGHTS_PATH}/{model_name}_'
train_params['save_training_weights_epoch'] = 250

model = Basic_linOut(network_params)
losses, initialTime, trainTime = model.train(task, train_params)
np.save(f'./{model_name}_losses.npy', np.array(losses))

plt.figure(figsize=(5,4))
plt.plot(losses)
plt.title('Loss during training')
plt.ylabel('Minibatch loss')
plt.xlabel('Batch number')
plt.tight_layout()
plt.savefig('./' + model_name + '_trainingLoss', dpi=300)

model.save(f'{SAVE_WEIGHTS_PATH}/' + model_name)
model.destruct()
