#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the dlPFC-like module (DLPFC_combined task), using a frozen, already-trained
MT-like module (see train_MT.py) to generate its motion/reward inputs.

@author: Sol
"""
from pathlib import Path
import os
ROOT_DIR = Path('/Users/Sol/Desktop/CohenLab/DotsBehavior/formattingRNN/python_code_data')
os.chdir(ROOT_DIR)

from DotsTasks import MT_broadInSharpOut_withR, DLPFC_combined
from psychrnn.backend.models.basic import Basic
from psychrnn.backend.simulation import BasicSimulator_linOut
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# TRAIN
SAVE_WEIGHTS_PATH = './model_weights'
model_name = 'DLPFCcombined_m4'
N_trainbatch = 500

# MT params
MTmodel_name = 'MTbroadsharp_m10'
MT_weights = dict(np.load(f'./model_weights/{MTmodel_name}.npz', allow_pickle=True))
N_rec_MT = MT_weights['W_rec'].shape[0]
MT_NdirOuts = 72
in_noise_MT = 0.4
rec_noise_MT = 0.2
coh = [0.6]

# define MT simulator
MT_task = MT_broadInSharpOut_withR(N_batch=N_trainbatch, N_rec=N_rec_MT,
                                   in_noise=in_noise_MT, coh=coh, catchP=0)
MT_network_params = MT_task.get_task_params()
MT_network_params['name'] = MTmodel_name
MT_network_params['N_rec'] = N_rec_MT
MT_network_params['rec_noise'] = rec_noise_MT
MT_simulator = BasicSimulator_linOut(weights=MT_weights, params=MT_network_params)

# DLPFC params:
in_noise_DLPFC = 0.2
rec_noise_DLPFC = 0.2
initial_weights = None
N_rec_DLPFC = 150 if initial_weights is None else initial_weights['W_rec'].shape[0]
dale = 0.8 if initial_weights is None else float(initial_weights['dale_ratio'])

task = DLPFC_combined(N_batch=N_trainbatch, N_rec=N_rec_DLPFC, in_noise=in_noise_DLPFC, MT_task=MT_task, MT_simulator=MT_simulator, MT_NdirOuts=MT_NdirOuts)

L2_in, L2_rec, L2_out, L2_FR = 0.01, 0, 0.01, 0.004
L1_in, L1_rec, L1_out = 0, 0, 0
network_params = task.get_task_params()
network_params['name'] = model_name
network_params['rec_noise'] = rec_noise_DLPFC
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

model = Basic(network_params)
if initial_weights is not None:
    transfer_function = network_params['transfer_function']
    for k,v in initial_weights.items():
        network_params[k] = v
    network_params['transfer_function'] = transfer_function
else:
    temp_weights = model.get_weights()
    for k,v in temp_weights.items():
        network_params[k] = v
model.destruct()

# fix microstim input weights at 0
W_in_fixed = np.zeros(network_params['W_in'].shape)
W_rec_fixed = np.zeros(network_params['W_rec'].shape)
W_out_fixed = np.zeros(network_params['W_out'].shape)

network_params['W_in'][:, -1] = 0
W_in_fixed[:, -1] = 1

train_params = {}
train_params['fixed_weights'] = {
    'W_in': W_in_fixed,
    'W_rec': W_rec_fixed,
    'W_out': W_out_fixed
}

train_params['training_iters'] = 1000000
train_params['learning_rate'] = 0.003

# Save weights during training
train_params['training_weights_path'] = f'{SAVE_WEIGHTS_PATH}/{model_name}_'
train_params['save_training_weights_epoch'] = 500

model = Basic(network_params)
losses, initialTime, trainTime = model.train(task, train_params)

plt.figure(figsize=(5,4))
plt.plot(losses)
plt.title('Loss during training')
plt.ylabel('Minibatch loss')
plt.xlabel('Batch number')
plt.tight_layout()
plt.savefig('./' + model_name + '_trainingLoss', dpi=300)
np.save(f'./{model_name}_losses.npy', np.array(losses))

model.save(f'{SAVE_WEIGHTS_PATH}/' + model_name)
model.destruct()
