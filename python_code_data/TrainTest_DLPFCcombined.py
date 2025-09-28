#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 14:01:08 2023

@author: Sol
"""
from pathlib import Path
import os
root_dir = Path(__file__).parent
os.chdir(str(root_dir))

from DotsTasks import MT_broadInSharpOut_withR, DLPFC_combined
from psychrnn.backend.models.basic import Basic
from psychrnn.backend.simulation import BasicSimulator_linOut
from psychrnn.backend.simulation import BasicSimulator
import matplotlib
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle
from cmcrameri import cm as cmc
import pandas as pd

#%% TRAIN
customLoss = True
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

if customLoss==True:
    network_params['loss_function'] = 'custom_loss_function'
    def loss_function_MSEandRewardDiff(predictions, y, output_mask, N_batch=N_trainbatch):
        loss1 = tf.cast(tf.reduce_mean(input_tensor=tf.square(output_mask * (predictions - y))), tf.float32)
        rewardDiff = 0
        for i in range(N_batch):
            maxReward = tf.math.reduce_max(y[i, -1, :])
            chosenUnit = tf.math.argmax(predictions[i, -1, :])
            chosenReward = tf.gather(y[i, -1, :], chosenUnit)
            rewardDiff += tf.square(maxReward - chosenReward)
        loss2 = 0.2 * tf.cast(rewardDiff/N_batch, tf.float32)
        return loss1 + loss2
    network_params['custom_loss_function'] = loss_function_MSEandRewardDiff

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
train_params['training_weights_path'] = f'./saved_weights/{model_name}_' 
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

model.save('./saved_weights/' + model_name)
model.destruct()
#%% TEST
model_name = 'DLPFCcombined_m4'
DLPFC_weights = dict(np.load(f'./saved_weights/{model_name}.npz', allow_pickle=True))

# MT params
MTmodel_name = 'MTbroadsharp_m10'
MT_weights = dict(np.load(f'./saved_weights/{MTmodel_name}.npz', allow_pickle=True))
MT_NdirOuts = 72
in_noise_MT = 0.4
rec_noise_MT = 0.2
coh = [0.6]
M_mstim_units = 34
M_mstim_strength = 0
M_stimulated_angle = 90

# DLPFC params
in_noise_DLPFC = 0.2
rec_noise_DLPFC = 0.2
D_mstim_units = None
D_mstim_strength = 0
D_stimulated_angle = None

N_testbatch = 72*250
fix_shown = (M_stimulated_angle+np.arange(0,360,5))%360
fix_reward = [M_stimulated_angle]
fix_onset = None
test1ofEach = False
saveMTactivity = False
if test1ofEach:
    N_testbatch = 18*72

# define MT simulator
MT_task = MT_broadInSharpOut_withR(N_batch=N_testbatch, N_rec=MT_weights['W_rec'].shape[0], in_noise=in_noise_MT, coh=coh, k_in=0.3, k_out=0.8, catchP=0, \
                                          fix_shown=fix_shown, fix_reward=fix_reward, fix_onset=fix_onset, M_mstim_strength=M_mstim_strength, test1ofEach=test1ofEach)
MT_network_params = MT_task.get_task_params()
MT_network_params['name'] = MTmodel_name
MT_network_params['N_rec'] = MT_weights['W_rec'].shape[0]
MT_network_params['rec_noise'] = rec_noise_MT

if M_mstim_units is not None:
    M_microstim = np.zeros(MT_weights['W_rec'].shape[0])
    M_microstim[M_mstim_units] = 1
    MT_weights['W_in'][:, -1] = M_microstim
    
MT_simulator = BasicSimulator_linOut(weights=MT_weights, params=MT_network_params)

# define the combined model task
task = DLPFC_combined(N_batch=N_testbatch, N_rec=DLPFC_weights['W_rec'].shape[0], in_noise=in_noise_DLPFC, D_mstim_strength=D_mstim_strength, \
                              MT_task=MT_task, MT_simulator=MT_simulator, saveMTactivity=saveMTactivity, MT_NdirOuts=MT_NdirOuts)

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

# calculate variables
choice = np.argmax(output[:, -1, :], axis=1)
choice_deg = choice*5
best_deg = np.array([np.argmax(trial_params[i]['reward'])*5 for i in range(trial_params.shape[0])])
shown_deg = np.array([trial_params[i]['shown_deg'] for i in range(trial_params.shape[0])])
bias_deg = np.array([trial_params[i]['good_deg'] for i in range(trial_params.shape[0])])
#%% save test data
folder = model_name
#savename = f'./{folder}/{model_name}_stimDLPFCu{D_mstim_units}pref{D_stimulated_angle}s{D_mstim_strength}_rewRel0shownRel5n_coh{coh[0]}noisein{in_noise_MT}mt{in_noise_DLPFC}pfc_rec{rec_noise_MT}mt{rec_noise_DLPFC}pfc' 
savename = f'./{folder}/{model_name}_stimMTu{M_mstim_units}pref{M_stimulated_angle}s{M_mstim_strength}_rewRel0shownRel5n_coh{coh[0]}noisein{in_noise_MT}mt{in_noise_DLPFC}pfc_rec{rec_noise_MT}mt{rec_noise_DLPFC}pfc' 
#savename = f'./{folder}/{model_name}_test{N_testbatch}_shown30nreward90n_on{fix_onset}_coh{coh[0]}noisein{in_noise_MT}mt{in_noise_DLPFC}pfc_rec{rec_noise_MT}mt{rec_noise_DLPFC}pfc' 
#savename = f'./{folder}/{model_name}_allCondsNoNoise_coh{coh[0]}'
'''
savefile = open(savename+'_DLPFCinput.pickle','wb')
pickle.dump(inputs, savefile, protocol=4)
savefile.close()
'''
savefile = open(savename+'_DLPFCoutput.pickle','wb')
pickle.dump(output, savefile, protocol=4)
savefile.close()
'''
savefile = open(savename+'_DLPFCstatevar.pickle','wb')
pickle.dump(state_var, savefile, protocol=4)
savefile.close()

if saveMTactivity==True:
    savefile = open(savename+'_MTstatevar.pickle','wb')
    pickle.dump(trial_params[0]['MTactivity'], savefile, protocol=4)
    savefile.close()
    trial_params[0].pop('MTactivity')
'''
savefile = open(savename+'_trialparams.pickle','wb')
pickle.dump(trial_params, savefile, protocol=4)
savefile.close()

#%% plot chosen vs shown
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.ion()
rcParams['font.size']=11
rcParams['font.sans-serif']='Helvetica'

plt.figure(figsize=(5,5))
plt.plot(np.arange(0,360,5), np.arange(0,360,5), c='k', lw=0.8)
size=10

b1_deg = 60
b1_inds = np.where(bias_deg == b1_deg)
choicecolor='dodgerblue'
bestcolor='navy'
plt.scatter(shown_deg[b1_inds], best_deg[b1_inds], facecolors=bestcolor, marker='s', s=size-2, label='most rewarded', alpha=0.5)
plt.scatter(shown_deg[b1_inds], choice_deg[b1_inds], facecolors=choicecolor, s=size, label=f'{b1_deg}'+r'$^{\degree}$ bias', alpha=0.5)

b2_deg = 240
b2_inds = np.where(bias_deg == b2_deg)
choicecolor='tomato'
bestcolor='darkred'
plt.scatter(shown_deg[b2_inds], best_deg[b2_inds], facecolors=bestcolor, marker='s', s=size-2, label='most rewarded', alpha=0.5)
plt.scatter(shown_deg[b2_inds], choice_deg[b2_inds], facecolors=choicecolor, s=size, label=f'{b2_deg}'+r'$^{\degree}$ bias', alpha=0.5)

plt.xlabel('Shown direction $^{\degree}$')
plt.ylabel('Chosen direction $^{\degree}$')
plt.title('Model behavior') 
plt.legend()
plt.tight_layout()

#plt.savefig(f'./{model_name}_chosenVshown_rew{b1_deg}and{b2_deg}_noisein{in_noise_MT}m{in_noise_DLPFC}d_rec{rec_noise_MT}mt{rec_noise_DLPFC}pfc_coh{coh[0]}_{N_testbatch}trials.png', dpi=300)
#%% plot example outputs
shownS = fix_shown
rew = fix_reward
for shown in shownS:
    color = cmc.romaO(shown/360)
    trials = np.random.choice(np.where(np.logical_and(shown_deg==shown, bias_deg==rew))[0], 50)
    plt.plot(np.arange(72), output[trials, -1, :].T, color=color, alpha=0.4)
    plt.plot(np.arange(72), target[trials[0], -1, :], color='k', ls='--')
    plt.xticks(np.arange(0,72,10), np.arange(0,360,50))
    
plt.ylabel('Reward function output')
plt.xlabel('Direction')
plt.title(f'Motion directions {shownS}, reward bias {rew}')
plt.tight_layout()

#plt.savefig(f'./{model_name}_exOutput_rew{rew}shown{shownS}_noisein{in_noise_MT}m{in_noise_DLPFC}d_rec{rec_noise_MT}mt{rec_noise_DLPFC}pfc_coh{coh[0]}.png', dpi=300)
#%% plot choice histogram for microstim

module = 'MT'

if module=='DLPFC':
    stimulated_angle = D_stimulated_angle
    strength = D_mstim_strength
    unit = D_mstim_units
elif module=='MT':
    stimulated_angle = M_stimulated_angle
    strength = M_mstim_strength
    unit = M_mstim_units

shown_angle = stimulated_angle-100
inds = np.where(shown_deg==shown_angle)[0]
best = best_deg[inds[0]]

plt.figure(figsize=(5, 4))
rcParams['font.size']=10
rcParams['font.sans-serif']='Helvetica'
rcParams['xtick.labelsize'] = 'medium'
rcParams['ytick.labelsize'] = 'medium'
plt.hist(choice_deg[inds], bins=np.arange(0, 360, 10), color='mediumblue')
plt.xlabel('Chosen angle (degrees)')
plt.ylabel('Number of trials')
ymax = 500
plt.vlines(shown_angle, 0, ymax, label=f'shown angle ({shown_angle})', color='k', ls='--')
if fix_reward[0] is not None:
    plt.vlines(fix_reward[0], 0, ymax, label=f'reward bias ({fix_reward[0]})', color='hotpink', ls='--')
    plt.vlines(best, 0, ymax, label=f'most rewarded ({best})', color='limegreen', ls='--')
if stimulated_angle is not None:
    plt.vlines(stimulated_angle, 0, 0.9*ymax, label=f'stimulated angle ({stimulated_angle})', color='r', ls='--')
    plt.title(f'Stimulate {module} unit {unit}, strength={strength}, coh={coh[0]}')

plt.legend(loc='upper left')
plt.tight_layout()

#plt.savefig(f'./{model_name}_mstimChoiceHist_stim{module}u{unit}pref{stimulated_angle}s{strength}_shown{shown_angle}rew{fix_reward[0]}_noisein{in_noise_MT}mt{in_noise_DLPFC}pfc_rec{rec_noise_MT}mt{rec_noise_DLPFC}pfc_coh{coh[0]}.png', dpi=300)
#%% plot microstim heatmap, one unit and mstim strength
from circledots_fxns import rel2center

module = 'MT'
unit = 34
stimulated_angle = 90
strength = 0

fix_shown = (stimulated_angle+np.arange(0,360,5))%360
savename = f'./DLPFCcombined_m4/DLPFCcombined_m4_stim{module}u{unit}pref{stimulated_angle}s{strength}_rewRel0shownRel5n_coh0.6noisein0.4mt0.2pfc_rec0.2mt0.2pfc'
output = pickle.load(open(savename + '_DLPFCoutput.pickle', 'rb'))
trial_params = pickle.load(open(savename + '_trialparams.pickle', 'rb'))
choice_deg = 5*np.argmax(output[:, -1, :], axis=1)
shown_deg = np.array([trial_params[i]['shown_deg'] for i in range(trial_params.shape[0])])

choiceFreqMat = np.zeros((fix_shown.shape[0], output.shape[2]))
for i in range(fix_shown.shape[0]):
    inds = np.where(shown_deg==fix_shown[i])[0]
    choices_i = np.argmax(output[inds, -1, :], axis=1)
    unique, counts = np.unique(choices_i, return_counts=True)
    choiceFreqMat[i, unique] += counts/np.sum(counts)

shown_rel2stim = rel2center(fix_shown, stimulated_angle)
shown_order_rel2stim = np.argsort(shown_rel2stim)
choice_rel2stim = rel2center(np.arange(output.shape[2])*5, stimulated_angle)
choice_order_rel2stim = np.argsort(choice_rel2stim)

choiceFreqMat_rel2stim = choiceFreqMat[shown_order_rel2stim, :][:, choice_order_rel2stim].T

plt.matshow(choiceFreqMat_rel2stim)
labels = np.arange(-150,160,50)
_, xticks, _ = np.intersect1d(shown_rel2stim[shown_order_rel2stim], labels, return_indices=True)
plt.xticks(xticks, labels)
_, yticks, _ = np.intersect1d(choice_rel2stim[choice_order_rel2stim], labels, return_indices=True)
plt.yticks(yticks, labels)

plt.colorbar()
plt.xlabel('motion direction relative to ustim site preference')
plt.ylabel('chosen angle relative to ustim site preference')
plt.gca().xaxis.tick_bottom()
plt.title('choice frequency')

#plt.savefig(savename + 'mstimHeatmap.png', dpi=300)

#np.save(savename + '_mstimHeatmap.npy', choiceFreqMat_rel2stim, allow_pickle=True)
#%% plot microstim difference heatmap, one unit

module = 'MT'
unit = 34
stimulated_angle = 90
strength2 = 12.0

mat1 = np.load(f'./DLPFCcombined_m4/DLPFCcombined_m4_stim{module}u{unit}pref{stimulated_angle}s0_rewRel0shownRel5n_coh0.6noisein0.4mt0.2pfc_rec0.2mt0.2pfc_mstimHeatmap.npy', allow_pickle=True)
mat2 = np.load(f'./DLPFCcombined_m4/DLPFCcombined_m4_stim{module}u{unit}pref{stimulated_angle}s{strength2}_rewRel0shownRel5n_coh0.6noisein0.4mt0.2pfc_rec0.2mt0.2pfc_mstimHeatmap.npy', allow_pickle=True)

plt.matshow(mat2-mat1)

labels = np.arange(-150,160,50)
ticks = np.array([ 5, 15, 25, 35, 45, 55, 65])
plt.xticks(ticks, labels)
plt.yticks(ticks, labels)

plt.colorbar()
plt.xlabel('motion direction relative to ustim site preference')
plt.ylabel('chosen angle relative to ustim site preference')
plt.gca().xaxis.tick_bottom()
plt.title(f'difference of choice frequency\nustim (strength {strength2}) - no ustim')

#plt.savefig(f'./DLPFCcombined_m4/DLPFCcombined_m4_stim{module}u{unit}pref{stimulated_angle}s0vs{strength2}_rewRel0shownRel5n_coh0.6noisein0.4mt0.2pfc_rec0.2mt0.2pfc_mstimDifferenceHeatmap.png', dpi=300)

#%% plot microstim difference heatmap, multiple units

module = 'MT'
units = [37, 4, 42, 25, 34]
strength2s = [11.0, 10.0, 10.0, 11.0, 12.0]
stim_angles = [110, 255, 0, 205, 90]

averageDifferenceMat = np.zeros((72, 72))

for i in range(len(units)):
    mat1 = np.load(f'./DLPFCcombined_m4/{module}layer_mstimPlots/DLPFCcombined_m4_stim{module}u{units[i]}pref{stim_angles[i]}s0_rewRel0shownRel5n_coh0.6noisein0.4mt0.2pfc_rec0.2mt0.2pfc_mstimHeatmap.npy', allow_pickle=True)
    mat2 = np.load(f'./DLPFCcombined_m4/{module}layer_mstimPlots/DLPFCcombined_m4_stim{module}u{units[i]}pref{stim_angles[i]}s{strength2s[i]}_rewRel0shownRel5n_coh0.6noisein0.4mt0.2pfc_rec0.2mt0.2pfc_mstimHeatmap.npy', allow_pickle=True)

    averageDifferenceMat += mat2-mat1

averageDifferenceMat /= len(units)

plt.matshow(averageDifferenceMat)

labels = np.arange(-150,160,50)
ticks = np.array([ 5, 15, 25, 35, 45, 55, 65])
plt.xticks(ticks, labels)
plt.yticks(ticks, labels)
plt.colorbar()
plt.xlabel('motion direction relative to ustim site preference')
plt.ylabel('chosen angle relative to ustim site preference')
plt.gca().xaxis.tick_bottom()
plt.title(f'average difference of choice frequency\nustim - no ustim, {len(units)} units')

#plt.savefig(f'./DLPFCcombined_m4/DLPFCcombined_m4_stim{module}u{units}_mstimDifferenceHeatmapAvg.png', dpi=300)
#%% binned microstim heatmap
og_mat = averageDifferenceMat
binsize_deg = 10
binned_mat = np.zeros((360//binsize_deg, 360//binsize_deg))

for i in range(360//binsize_deg):
    for j in range(360//binsize_deg):
        n = binsize_deg//5
        binned_mat[i, j] = np.mean(og_mat[i*n:(i+1)*n, j*n:(j+1)*n])

plt.matshow(binned_mat)

#%% tuning to CHOICE

model_name = 'DLPFCcombined_m4'
DLPFC_weights = dict(np.load(f'./saved_weights/{model_name}.npz', allow_pickle=True))
N_testbatch = 30000

# MT params
MTmodel_name = 'MTbroadsharp_m10'
MT_weights = dict(np.load(f'./saved_weights/{MTmodel_name}.npz', allow_pickle=True))
MT_NdirOuts = 72
in_noise_MT = 0.4
rec_noise_MT = 0.2
coh = [0.6]
fix_onset=[100, 500]

# DLPFC params
in_noise_DLPFC = 0.2
rec_noise_DLPFC = 0.2

# define MT simulator
MT_task = MT_broadInSharpOut_withR(N_batch=N_testbatch, N_rec=MT_weights['W_rec'].shape[0], in_noise=in_noise_MT, coh=coh, k_in=0.3, k_out=0.8, catchP=0, fix_onset=fix_onset)

MT_network_params = MT_task.get_task_params()
MT_network_params['name'] = MTmodel_name
MT_network_params['N_rec'] = MT_weights['W_rec'].shape[0]
MT_network_params['rec_noise'] = rec_noise_MT
    
MT_simulator = BasicSimulator_linOut(weights=MT_weights, params=MT_network_params)

# define the combined model task
task = DLPFC_combined(N_batch=N_testbatch, N_rec=DLPFC_weights['W_rec'].shape[0], in_noise=in_noise_DLPFC, MT_task=MT_task, MT_simulator=MT_simulator, MT_NdirOuts=MT_NdirOuts)

network_params = task.get_task_params()
network_params['N_rec'] = DLPFC_weights['W_rec'].shape[0]
network_params['rec_noise'] = rec_noise_DLPFC

simulator = BasicSimulator(weights=DLPFC_weights, params=network_params)   

inputs, target, mask, trial_params = task.get_trial_batch()
output, state_var = simulator.run_trials(inputs)

choice = np.argmax(output[:, -1, :], axis=1)
choice_deg = choice*5
avg_fr_stim = np.mean(np.maximum(state_var[:, fix_onset[1]//10:, :], 0), axis=1)

all_choices = np.arange(0,360,5)
tuningMat_choice = np.zeros((DLPFC_weights['W_rec'].shape[0], all_choices.shape[0]))
for c in range(all_choices.shape[0]):
    inds = np.where(choice_deg==all_choices[c])[0]
    tuningMat_choice[:, c] = np.mean(avg_fr_stim[inds], axis=0)

pref_choice = np.argmax(tuningMat_choice, axis=1)
order_c = np.argsort(pref_choice)
plt.matshow(tuningMat_choice[order_c,:])

savefile = open(f'./{model_name}_tuningMat_choice_{N_testbatch}trials_coh{coh[0]}noisein{in_noise_MT}mt{in_noise_DLPFC}pfc_rec{rec_noise_MT}mt{rec_noise_DLPFC}pfc.pickle','wb')
pickle.dump(tuningMat_choice, savefile, protocol=4)
savefile.close()
#%% tuning to SHOWN
model_name = 'DLPFCcombined_m4'
DLPFC_weights = dict(np.load(f'./saved_weights/{model_name}.npz', allow_pickle=True))

# MT params
MTmodel_name = 'MTbroadsharp_m10'
MT_weights = dict(np.load(f'./saved_weights/{MTmodel_name}.npz', allow_pickle=True))
MT_NdirOuts = 72
in_noise_MT = 0.4
rec_noise_MT = 0.2
coh = [0.6]
fix_onset=[100, 500]

# DLPFC params
in_noise_DLPFC = 0.2
rec_noise_DLPFC = 0.2

N_testbatch = 500

shownDirs_ = np.arange(0, 360, 5)
tuningMat_shown = np.zeros((DLPFC_weights['W_rec'].shape[0], shownDirs_.shape[0]))
for shownDir in shownDirs_:
    print(shownDir)
    # define MT simulator
    MT_task = MT_broadInSharpOut_withR(N_batch=N_testbatch, N_rec=MT_weights['W_rec'].shape[0], in_noise=in_noise_MT, coh=coh, k_in=0.3, k_out=0.8, catchP=0, \
                                       fix_onset=fix_onset, fix_shown=[shownDir])
    
    MT_network_params = MT_task.get_task_params()
    MT_network_params['name'] = MTmodel_name
    MT_network_params['N_rec'] = MT_weights['W_rec'].shape[0]
    MT_network_params['rec_noise'] = rec_noise_MT
        
    MT_simulator = BasicSimulator_linOut(weights=MT_weights, params=MT_network_params)
    
    # define the combined model task
    task = DLPFC_combined(N_batch=N_testbatch, N_rec=DLPFC_weights['W_rec'].shape[0], in_noise=in_noise_DLPFC, MT_task=MT_task, MT_simulator=MT_simulator, MT_NdirOuts=MT_NdirOuts)
    
    network_params = task.get_task_params()
    network_params['N_rec'] = DLPFC_weights['W_rec'].shape[0]
    network_params['rec_noise'] = rec_noise_DLPFC
    
    simulator = BasicSimulator(weights=DLPFC_weights, params=network_params)   
    
    inputs, target, mask, trial_params = task.get_trial_batch()
    output, state_var = simulator.run_trials(inputs)

    fr_stim = np.maximum(state_var[:, fix_onset[1]//10:, :], 0)
    avgFR = np.mean(np.mean(fr_stim, axis=1), axis=0)
    tuningMat_shown[:, shownDir//(360//shownDirs_.shape[0])] = avgFR

pref_dir = np.argmax(tuningMat_shown, axis=1)
order_s = np.argsort(pref_dir)
plt.matshow(tuningMat_shown[order_s,:])

savefile = open(f'./{model_name}_tuningMat_shown_{N_testbatch}testbatch_coh{coh[0]}noisein{in_noise_MT}mt{in_noise_DLPFC}pfc_rec{rec_noise_MT}mt{rec_noise_DLPFC}pfc.pickle','wb')
pickle.dump(tuningMat_shown, savefile, protocol=4)
savefile.close()

#%% plot pref dirs (E and I separated)

tuningType = 'shown'

model_name = 'DLPFCcombined_m4'
folder = 'DLPFCcombined_m4'
loaded_weights = dict(np.load(f'./saved_weights/{model_name}.npz', allow_pickle=True))
N_rec, dale = loaded_weights['W_rec'].shape[0], float(loaded_weights['dale_ratio'])
E_units = np.arange(int(N_rec*dale))
I_units = np.arange(int(N_rec*dale), N_rec)
all_units = np.arange(N_rec)
tuningMat_shown = pickle.load(open(f'./{folder}/{model_name}_tuningMat_{tuningType}_500testbatch_coh{coh[0]}noisein{in_noise_MT}mt{in_noise_DLPFC}pfc_rec{rec_noise_MT}mt{rec_noise_DLPFC}pfc.pickle','rb'))
ds = 360//tuningMat_shown.shape[1]
pref_dirsE = np.argmax(tuningMat_shown[E_units, :], axis=1) * ds
pref_dirsI = np.argmax(tuningMat_shown[I_units, :], axis=1) * ds
orderE = np.argsort(pref_dirsE)
orderI = np.argsort(pref_dirsI)
combined_order = np.concatenate((orderE, orderI + I_units[0]))
above_thresh = np.where(np.max(tuningMat_shown, axis=1)>0)[0]
combined_order_thresh = combined_order[np.isin(combined_order, above_thresh)]
plt.matshow(tuningMat_shown[combined_order_thresh, :])
plt.colorbar(shrink=0.5)

rcParams['font.size'] = 10
plt.title('avg FR during stimulus, sorted')
plt.xlabel(f'{tuningType} direction (degrees)')
plt.xticks(ticks=np.arange(0, tuningMat_shown.shape[1], 60//ds), labels=np.arange(0,360,60))
plt.ylabel('Recurrent unit')
plt.gca().xaxis.tick_bottom()
#plt.savefig(f'./{folder}/{model_name}_tuningMat_{tuningType}_sortedEandI_500testbatch_coh{coh[0]}.png', dpi=300)

#%% tuning/mstim dataframe
model_name = 'DLPFCcombined_m4'
DLPFC_weights = dict(np.load(f'./saved_weights/{model_name}.npz', allow_pickle=True))
W_out = DLPFC_weights['W_out']
N_rec = W_out.shape[1]
dale = float(DLPFC_weights['dale_ratio'])
prefDirs_out = np.argmax(W_out[:, :int(dale*N_rec)], axis=0)*5
maxWeights = np.max(W_out[:, :int(dale*N_rec)], axis=0)

tuningDf_out = pd.DataFrame(data={'pref_dir':prefDirs_out, 'max_weight':maxWeights})
#%% performance measure during training, from saved weights

def propRewardReceived(output, target):
    maxRewards = np.max(target[:, -1, :], axis=1)
    chosenUnits = np.argmax(output[:, -1, :], axis=1)
    props = np.zeros(maxRewards.shape[0])
    for i in range(maxRewards.shape[0]):
        receivedReward = target[i, -1, chosenUnits[i]]
        props[i] = receivedReward/maxRewards[i]
    return props

# MT params
MTmodel_name = 'MTbroadsharp_m10'
MT_weights = dict(np.load(f'./saved_weights/{MTmodel_name}.npz', allow_pickle=True))
MT_NdirOuts = 72
in_noise_MT = 0.4
rec_noise_MT = 0.2
coh = [0.6]

# DLPFC params
in_noise_DLPFC = 0.2
rec_noise_DLPFC = 0.2
N_testbatch = 2000

model_name = 'DLPFCcombined_m3'
DLPFC_weights = dict(np.load(f'./saved_weights/{model_name}.npz', allow_pickle=True))

# define MT simulator
MT_task = MT_broadInSharpOut_withR(N_batch=N_testbatch, N_rec=MT_weights['W_rec'].shape[0], in_noise=in_noise_MT, coh=coh, k_in=0.3, k_out=0.8, catchP=0)
MT_network_params = MT_task.get_task_params()
MT_network_params['name'] = MTmodel_name
MT_network_params['N_rec'] = MT_weights['W_rec'].shape[0]
MT_network_params['rec_noise'] = rec_noise_MT
    
MT_simulator = BasicSimulator_linOut(weights=MT_weights, params=MT_network_params)

# define the combined model task
task = DLPFC_combined(N_batch=N_testbatch, N_rec=DLPFC_weights['W_rec'].shape[0], in_noise=in_noise_DLPFC, MT_task=MT_task, MT_simulator=MT_simulator, MT_NdirOuts=MT_NdirOuts)

network_params = task.get_task_params()
network_params['N_rec'] = DLPFC_weights['W_rec'].shape[0]
network_params['rec_noise'] = rec_noise_DLPFC

# iterate over weights
performance = []
batches = []
save_weights_epoch = 500
i = save_weights_epoch
while i<=2000:
    train_name = model_name + f'_{i}'
    DLPFC_weights = dict(np.load(f'./saved_weights/{train_name}.npz', allow_pickle=True))
    print(i)
    batches.append(i)
    
    simulator = BasicSimulator(weights=DLPFC_weights, params=network_params)   
    
    inputs, target, mask, trial_params = task.get_trial_batch()
    output, state_var = simulator.run_trials(inputs)
    
    performance.append(np.mean(propRewardReceived(output, target)))
    
    i += save_weights_epoch

batches = np.array(batches)
performance = np.array(performance)
#%% plot meanPropRewardReceived
plt.figure(figsize=(6,5))
plt.plot(batches, performance)
plt.scatter(batches, performance)
plt.xticks(batches[::1])
plt.xlabel('Training batch')
plt.ylabel('Mean proportion of rewards')
plt.ylim(0.9, 0.98)
plt.tight_layout()

#plt.savefig(f'./{model_name}_meanPropRewardReceived_duringTraining.png', dpi=300)
