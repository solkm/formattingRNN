#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate the combined two-module model (frozen, trained MT-like module feeding a
frozen, trained dlPFC-like module) using BasicSimulator, to generate test data for
downstream analysis (see test_DLPFC.py).

@author: Sol
"""
from pathlib import Path
import os
ROOT_DIR = Path('/Users/Sol/Desktop/CohenLab/DotsBehavior/formattingRNN/python_code_data')
os.chdir(ROOT_DIR)

from DotsTasks import MT_broadInSharpOut_withR, DLPFC_combined
from psychrnn.backend.simulation import BasicSimulator_linOut
from psychrnn.backend.simulation import BasicSimulator
import numpy as np
import pickle

#%% SIMULATE
SAVED_WEIGHTS_PATH = './model_weights'
model_name = 'DLPFCcombined_m4'
DLPFC_weights = dict(np.load(f'{SAVED_WEIGHTS_PATH}/{model_name}.npz', allow_pickle=True))

# MT model to use, noise and coherence params
MTmodel_name = 'MTbroadsharp_m10'
MT_weights = dict(np.load(f'{SAVED_WEIGHTS_PATH}/{MTmodel_name}.npz', allow_pickle=True))
MT_NdirOuts = 72
in_noise_MT = 0
rec_noise_MT = 0
coh = [0.6]

# MT microstimulation params
M_mstim_units = None
M_mstim_strength = 0
M_stimulated_angle = None

# DLPFC noise params
in_noise_DLPFC = 0
rec_noise_DLPFC = 0

# DLPFC microstimulation params
D_mstim_units = None
D_mstim_strength = 0
D_stimulated_angle = None

# Conditions to simulate
N_testbatch = 72*18
fix_shown = np.arange(0, 360, 5) # the shown motion directions to select from
fix_reward = np.arange(0, 360, 20) # the reward bias directions to select from
fix_onset = None # None or [t_ring, t_dots] to fix variable stimulus onset times
test1ofEach = True # if True, generates a batch of trials with one trial of each condition in fix_shown x fix_reward
if test1ofEach:
    N_testbatch = len(fix_shown) * len(fix_reward)

# Saving config. The final output (of the DLPFC module) and trial parameters are always saved, 
#   but the following allow saving additional data:
saveDLPFCinput = False # if True, saves the DLPFC input (derived from MT module output) for each trial
saveDLPFCactivity = False # if True, saves the DLPFC hidden layer activity (state_var)
saveMTinput = False # if True, saves the MT input (input to the whole model) for each trial
saveMTactivity = False # if True, saves the MT hidden layer activity for each trial
saveMToutput = False # if True, saves the MT module's own output for each trial

# Define the MT simulator
MT_task = MT_broadInSharpOut_withR(
    N_batch=N_testbatch, N_rec=MT_weights['W_rec'].shape[0], in_noise=in_noise_MT, 
    coh=coh, k_in=0.3, k_out=0.8, catchP=0, fix_shown=fix_shown, fix_reward=fix_reward, 
    fix_onset=fix_onset, M_mstim_strength=M_mstim_strength, test1ofEach=test1ofEach
)
MT_network_params = MT_task.get_task_params()
MT_network_params['name'] = MTmodel_name
MT_network_params['N_rec'] = MT_weights['W_rec'].shape[0]
MT_network_params['rec_noise'] = rec_noise_MT

if M_mstim_units is not None:
    M_microstim = np.zeros(MT_weights['W_rec'].shape[0])
    M_microstim[M_mstim_units] = 1
    MT_weights['W_in'][:, -1] = M_microstim

MT_simulator = BasicSimulator_linOut(weights=MT_weights, params=MT_network_params)

# Define the combined model task and simulator
task = DLPFC_combined(
    N_batch=N_testbatch, N_rec=DLPFC_weights['W_rec'].shape[0], in_noise=in_noise_DLPFC,
    D_mstim_strength=D_mstim_strength, MT_task=MT_task, MT_simulator=MT_simulator,
    saveMTactivity=saveMTactivity, saveMTinput=saveMTinput, saveMToutput=saveMToutput, MT_NdirOuts=MT_NdirOuts
)
network_params = task.get_task_params()

network_params['N_rec'] = DLPFC_weights['W_rec'].shape[0]
network_params['rec_noise'] = rec_noise_DLPFC

if D_mstim_units is not None:
    D_microstim = np.zeros(network_params['N_rec'])
    D_microstim[D_mstim_units] = 1
    DLPFC_weights['W_in'][:, -1] = D_microstim

simulator = BasicSimulator(weights=DLPFC_weights, params=network_params)

inputs, target, mask, trial_params = task.get_trial_batch()
output, state_var = simulator.run_trials(inputs)

# SAVE THE SIMULATION DATA
folder = model_name
if D_mstim_units is not None:
    savename = f'./{folder}/{model_name}_stimDLPFCu{D_mstim_units}pref{D_stimulated_angle}s{D_mstim_strength}_rewRel0shownRel5n_coh{coh[0]}noisein{in_noise_MT}mt{in_noise_DLPFC}pfc_rec{rec_noise_MT}mt{rec_noise_DLPFC}pfc'
elif M_mstim_units is not None:
    savename = f'./{folder}/{model_name}_stimMTu{M_mstim_units}pref{M_stimulated_angle}s{M_mstim_strength}_rewRel0shownRel5n_coh{coh[0]}noisein{in_noise_MT}mt{in_noise_DLPFC}pfc_rec{rec_noise_MT}mt{rec_noise_DLPFC}pfc'
elif in_noise_MT == in_noise_DLPFC == rec_noise_MT == rec_noise_DLPFC == 0 and test1ofEach and N_testbatch == 18*72:
    savename = f'./{folder}/{model_name}_allCondsNoNoise_coh{coh[0]}'
else:
    print('Default savename used')
    savename = f'./{folder}/{model_name}_coh{coh[0]}noisein{in_noise_MT}mt{in_noise_DLPFC}pfc_rec{rec_noise_MT}mt{rec_noise_DLPFC}pfc'

if saveDLPFCinput == True:
    savefile = open(savename + '_DLPFCinput.pickle','wb')
    pickle.dump(inputs, savefile, protocol=4)
    savefile.close()

if saveDLPFCactivity == True:
    savefile = open(savename + '_DLPFCstatevar.pickle','wb')
    pickle.dump(state_var, savefile, protocol=4)
    savefile.close()

if saveMTinput == True:
    savefile = open(savename + '_MTinput.pickle','wb')
    pickle.dump(trial_params[0]['MTinput'], savefile, protocol=4)
    savefile.close()
    trial_params[0].pop('MTinput')

if saveMTactivity==True:
    savefile = open(savename + '_MTstatevar.pickle','wb')
    pickle.dump(trial_params[0]['MTactivity'], savefile, protocol=4)
    savefile.close()
    trial_params[0].pop('MTactivity')

if saveMToutput==True:
    savefile = open(savename + '_MToutput.pickle','wb')
    pickle.dump(trial_params[0]['MToutput'], savefile, protocol=4)
    savefile.close()
    trial_params[0].pop('MToutput')

savefile = open(savename + '_DLPFCoutput.pickle','wb')
pickle.dump(output, savefile, protocol=4)
savefile.close()

savefile = open(savename + '_trialparams.pickle','wb')
pickle.dump(trial_params, savefile, protocol=4)
savefile.close()
