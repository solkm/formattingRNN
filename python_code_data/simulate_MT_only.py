#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate the MT-like module alone using BasicSimulator_linOut. 
Use this to test the MT module before the dlPFC-like module is trained (see train_DLPFC.py).

@author: Sol
"""
from pathlib import Path
import os
ROOT_DIR = Path('/Users/Sol/Desktop/CohenLab/DotsBehavior/formattingRNN/python_code_data')
os.chdir(ROOT_DIR)

from DotsTasks import MT_broadInSharpOut_withR
from psychrnn.backend.simulation import BasicSimulator_linOut
import numpy as np
import pickle

#%% SIMULATE
SAVED_WEIGHTS_PATH = './model_weights'
model_name = 'MTbroadsharp_m10'
loaded_weights = dict(np.load(f'{SAVED_WEIGHTS_PATH}/{model_name}.npz', allow_pickle=True))
N_rec = loaded_weights['W_rec'].shape[0]

# MT params
coh = [0.6]
in_noise = 0
rec_noise = 0

# MT microstimulation params
M_mstim_units = None
M_mstim_strength = 0
M_stimulated_angle = None

# Conditions to simulate
N_testbatch = 72*18
fix_shown = np.arange(0, 360, 5) # the shown motion directions to select from
fix_reward = np.arange(0, 360, 20) # the reward bias directions to select from
fix_onset = None # None or [t_ring, t_dots] to fix variable stimulus onset times
test1ofEach = True # if True, generates a batch of trials with one trial of each condition in fix_shown x fix_reward

# Saving config. The output and trial parameters are always saved, but the following allow saving additional data:
saveInput = False # if True, saves the model input for each trial
saveActivity = False # if True, saves the recurrent hidden layer activity (state_var)

# Define the MT simulator
task = MT_broadInSharpOut_withR(
    N_batch=N_testbatch, N_rec=N_rec, in_noise=in_noise, coh=coh, k_in=0.3, k_out=0.8,
    M_mstim_strength=M_mstim_strength, fix_shown=fix_shown, fix_reward=fix_reward,
    catchP=0.0, fix_onset=fix_onset, test1ofEach=test1ofEach
)

network_params = task.get_task_params()
network_params['name'] = model_name
network_params['N_rec'] = N_rec
network_params['rec_noise'] = rec_noise

if M_mstim_units is not None:
    M_microstim = np.zeros(N_rec)
    M_microstim[M_mstim_units] = 1
    loaded_weights['W_in'][:, -1] = M_microstim

simulator = BasicSimulator_linOut(weights=loaded_weights, params=network_params)

inputs, target, mask, trial_params = task.get_trial_batch()
output, state_var = simulator.run_trials(inputs)

# SAVE THE SIMULATION DATA
SAVE_DIR = './figure_code/testdata'
if M_mstim_units is not None:
    savename = f'{SAVE_DIR}/{model_name}_stimu{M_mstim_units}pref{M_stimulated_angle}s{M_mstim_strength}_coh{coh[0]}noisein{in_noise}rec{rec_noise}'
elif in_noise == rec_noise == 0 and test1ofEach:
    savename = f'{SAVE_DIR}/{model_name}_allCondsNoNoise_coh{coh[0]}'
else:
    print('Default savename used')
    savename = f'{SAVE_DIR}/{model_name}_coh{coh[0]}noisein{in_noise}rec{rec_noise}'

if saveInput == True:
    savefile = open(savename+'_input.pickle','wb')
    pickle.dump(inputs, savefile, protocol=4)
    savefile.close()

if saveActivity == True:
    savefile = open(savename+'_statevar.pickle','wb')
    pickle.dump(state_var, savefile, protocol=4)
    savefile.close()

savefile = open(savename+'_output.pickle','wb')
pickle.dump(output, savefile, protocol=4)
savefile.close()

savefile = open(savename+'_trialparams.pickle','wb')
pickle.dump(trial_params, savefile, protocol=4)
savefile.close()

# %%
