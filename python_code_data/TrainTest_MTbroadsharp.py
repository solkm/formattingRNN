#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 10:10:27 2023

@author: Sol
"""

import os
os. chdir('/Users/Sol/Desktop/CohenLab/DotsBehavior')
import sys
sys.path.insert(0, '/Users/Sol/PsychRNN/psychrnn/backend/models')
from basicLinearOutput import Basic_linOut
sys.path.insert(0, '/Users/Sol/PsychRNN/psychrnn/backend')
from simulation import BasicSimulator_linOut
from psychrnn.backend.models.basic import Basic
from psychrnn.backend.simulation import BasicSimulator
from DotsTasks import MT_broadInSharpOut_withR
import numpy as np
import pickle
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
import matplotlib.pyplot as plt
plt.ion()
from matplotlib import rcParams
from matplotlib import cm
import tensorflow as tf
rng = np.random.default_rng()
import pandas as pd
from cmcrameri import cm as cmc
#%% TRAIN
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
    
    def loss_function_MSEandArgmax_withRewBias(predictions, y, output_mask):
    
        loss1 = tf.cast(tf.reduce_mean(input_tensor=tf.square(output_mask * (predictions - y))), tf.float32)
        
        choiceDegs = 5 * tf.math.argmax(tf.math.reduce_mean(predictions[:, -10:, 2:], 1), 1)
        shownDegs = 5 * tf.math.argmax(y[:, -1, 2:], 1)
        distFromShown_deg = tf.abs(shownDegs - choiceDegs)
        distFromShown_deg = tf.math.minimum(distFromShown_deg, 360 - distFromShown_deg)
        loss2 = 1.5 * tf.reduce_mean(input_tensor=tf.square(tf.cast(distFromShown_deg, tf.float32) * np.pi/180))
        
        rewRads = tf.math.atan2(y[:, -1, 1], y[:, -1, 0])
        rewEstRads = tf.math.atan2(predictions[:, -1, 1], predictions[:, -1, 0])
        distFromRew_rad = tf.abs(rewRads - rewEstRads)
        distFromRew_rad = tf.math.minimum(distFromRew_rad, 2*np.pi - distFromRew_rad)
        loss3 = 0.3 * tf.reduce_mean(input_tensor=tf.square(tf.cast(distFromRew_rad, tf.float32)))
        
        return loss1 + loss2 + loss3
    
    network_params['custom_loss_function'] = loss_function_MSEandArgmax_withRewBias

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
train_params['training_weights_path'] = f'./saved_weights/{model_name}_' 
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

model.save('./saved_weights/' + model_name)
model.destruct()
#%% TEST
model_name = 'MTbroadsharp_m10'
loaded_weights = dict(np.load(f'./saved_weights/{model_name}.npz', allow_pickle=True))
N_rec = loaded_weights['W_rec'].shape[0]
M_mstim_units = 37
M_mstim_strength = 12.0
stimulated_angle = 110
fix_shown = [210]
fix_reward = [110]
fix_onset = None
N_testbatch = 500
test1ofEach = False
coh = [0.8]
in_noise = 0.6
rec_noise = 0.2

task = MT_broadInSharpOut_withR(N_batch=N_testbatch, N_rec=N_rec, in_noise=in_noise, coh=coh, k_in=0.3, k_out=0.8, M_mstim_strength=M_mstim_strength, \
                                fix_shown=fix_shown, fix_reward=fix_reward, catchP=0.0, fix_onset=fix_onset, test1ofEach=test1ofEach)

inputs, target, mask, trial_params = task.get_trial_batch()

network_params = task.get_task_params()
network_params['name'] = model_name
network_params['N_rec'] = N_rec
network_params['rec_noise'] = rec_noise

if M_mstim_units is not None:
    M_microstim = np.zeros(N_rec)
    M_microstim[M_mstim_units] = 1
    loaded_weights['W_in'][:, -1] = M_microstim

simulator = BasicSimulator_linOut(weights=loaded_weights, params=network_params)
output, state_var = simulator.run_trials(inputs)

# calculate task variables
N_dirOuts = 72
choice_period = np.arange(output.shape[1]-10, output.shape[1])
choice_output = np.mean(output[:, choice_period, :], axis=1)

choice_deg = np.argmax(choice_output[:, -N_dirOuts:], axis=1) * (360/N_dirOuts)
shown_deg = np.array([trial_params[i]['shown_deg'] for i in range(trial_params.shape[0])])
bias_deg = np.array([trial_params[i]['good_deg'] for i in range(trial_params.shape[0])]) 

bias_est = np.degrees(np.arctan2(choice_output[:, 1], choice_output[:, 0]))
bias_est[bias_est<0] += 360

folder = model_name
#%% save test data
folder = model_name
savename = f'./{folder}/{model_name}_allCondsNoNoise_coh{coh[0]}'
'''
savefile = open(savename+'_input.pickle','wb')
pickle.dump(inputs[:], savefile, protocol=4)
savefile.close()
'''
savefile = open(savename+'_output.pickle','wb')
pickle.dump(output[:], savefile, protocol=4)
savefile.close()

savefile = open(savename+'_trialparams.pickle','wb')
pickle.dump(trial_params[:], savefile, protocol=4)
savefile.close()

savefile = open(savename+'_statevar.pickle','wb')
pickle.dump(state_var[:], savefile, protocol=4)
savefile.close()


#%% chosen vs shown
rcParams['font.size']=11
plt.figure(figsize=(6,6))
plt.plot(np.linspace(0, 360, 10), np.linspace(0, 360, 10), c='k')
plt.scatter(shown_deg, choice_deg, facecolors='b', edgecolors='b', alpha=0.2, s=20)
plt.xlabel('Shown direction (degrees)')
plt.ylabel('Chosen direction (degrees)')
plt.tight_layout()

plt.savefig(f'./{folder}/{model_name}_chosenVshown_coh{coh[0]}noisein{in_noise}rec{rec_noise}.png', dpi=300)
#%% bias deg vs estimate
rcParams['font.size']=11
plt.figure(figsize=(6,6))
plt.plot(np.linspace(0, 360, 5), np.linspace(0, 360, 5), c='k')
plt.scatter(bias_deg, bias_est, facecolors='b', edgecolors='b', alpha=0.2, s=20)
plt.xlabel('Reward bias location (deg)')
plt.ylabel('Reward bias estimate (deg)')
plt.tight_layout()

plt.savefig(f'./{folder}/{model_name}_biasVestimate_coh{coh[0]}noisein{in_noise}rec{rec_noise}.png', dpi=300)
#%% plot output activity, mean and sd of condition
shown=fix_shown[0]
trials = np.where(shown_deg==shown)[0]
color = cmc.romaO(shown/360)

mean = np.mean(output[trials, -1, -N_dirOuts:], axis=0)
sd = np.std(output[trials, -1, -N_dirOuts:], axis=0)

#plt.plot(np.arange(N_dirOuts), target[trials[0], -1, -N_dirOuts:], label='target', color=color, ls='--')
plt.plot(np.arange(N_dirOuts), mean, label=f'coh={coh[0]}, shown={shown}', color=color)
plt.fill_between(np.arange(N_dirOuts), mean-sd, mean+sd, color=color, alpha=0.2)

plt.legend()
plt.ylabel('Output activity')
plt.xlabel('Direction (deg)')
plt.xticks(np.arange(0,N_dirOuts,10), 360/N_dirOuts*np.arange(0,N_dirOuts,10))
plt.tight_layout()
#plt.savefig(f'./{folder}/{model_name}_meanOutputActivity_shown[60,180,300]_coh{coh[0]}noisein{in_noise}rec{rec_noise}.png', dpi=300)

if stimulated_angle is not None: 
    shown_angle = fix_shown[0]
    plt.vlines(shown_angle/(360//N_dirOuts), np.min(mean-sd), np.max(mean+sd), label=f'shown ({shown_angle} deg)', color='k', ls='--')
    plt.vlines(stimulated_angle/(360//N_dirOuts), np.min(mean-sd), np.max(mean+sd), label=f'stimulated ({stimulated_angle} deg)', color='r', ls='--')
    plt.legend()
    plt.savefig(f'./{model_name}_meanOutputActivity_stimU{M_mstim_units}pref{stimulated_angle}s{M_mstim_strength}_shown{shown_angle}rew{fix_reward[0]}coh{coh[0]}_noisein{in_noise}rec{rec_noise}.png', dpi=300)

#%% plot output activity, many individual trials
sd = [170]
color = cm.viridis(coh[0])

t = 60

plt.figure(figsize=(6,4))
for s in sd:
    trials = np.where(shown_deg==s)[0]
    for i in range(trials.shape[0]):
        plt.plot(output[trials[i],t, -N_dirOuts:], alpha=0.5, color=color, zorder=2)
        plt.plot(target[trials[i],t, -N_dirOuts:], alpha=0.5, color='grey', zorder=1)

plt.ylabel('Output activity')
plt.xticks(np.arange(0, N_dirOuts, 10), 360//N_dirOuts * np.arange(0, N_dirOuts, 10))
plt.xlabel('Direction (deg)')
plt.tight_layout()
#plt.savefig(f'./{model_name}_exOutputActivities_shown{sd}_coh{coh[0]}noisein{in_noise}rec{rec_noise}.png', dpi=300)
#%% plot recurrent activity, single trial

trial = 0
plt.plot(state_var[trial,:,:], alpha=0.2)
if M_mstim_units is not None:
    plt.plot(state_var[trial,:,M_mstim_units], alpha=1, c='r')

#%% task visualization: plot inputs and target outputs over time

fix_shown = [180]
fix_onset = [100, 500]
N_testbatch = 1
coh = [0.4]
in_noise = 0.3

task = MT_broadInSharpOut_withR(N_batch=N_testbatch, in_noise=in_noise, coh=coh, fix_shown=fix_shown, catchP=0.0, fix_onset=fix_onset)

inputs, target, mask, trial_params = task.get_trial_batch()
fig, axs = plt.subplots(2, figsize=(3,6))
for t in range(fix_onset[1]//10, fix_onset[1]//10+15):
    axs[0].clear()
    axs[1].clear()
    axs[0].plot(np.arange(0,360,360//(inputs.shape[2]-3)), inputs[0, t, 2:-1], c='b', alpha=1)
    axs[1].plot(np.arange(0,360,360//(target.shape[2]-2)), target[0, t, 2:], c='orange', alpha=1)
    axs[0].set_title('Input_t')
    axs[1].set_title('TargetOutput_t')
    axs[0].set_ylim(0-in_noise, 1+in_noise)
    axs[1].set_ylim(0, 1.5)
    fig.tight_layout()
    plt.pause(0.1)
    
#%% after training: plot inputs and outputs over time
shown = 200
trial = np.where(shown_deg==shown)[0][1]

fig, axs = plt.subplots(2, figsize=(3,6))
for t in range(45, inputs.shape[1]):
    axs[0].clear()
    axs[1].clear()
    axs[0].plot(np.arange(0,360,360//(inputs.shape[2]-3)), inputs[trial, t, 2:-1], c='b', alpha=1)
    axs[1].plot(np.arange(0,360,360//(target.shape[2]-2)), target[trial, t, 2:], c='grey', alpha=1)
    axs[1].plot(np.arange(0,360,360//(output.shape[2]-2)), output[trial, t, 2:], c='orange', alpha=1)
    axs[0].set_title(f'Input t={t}')
    axs[1].set_title('Output')
    axs[0].set_ylim(np.min(inputs[:,:,2:-1]), np.max(inputs[:,:,2:-1]))
    axs[1].set_ylim(np.min(output[:,:,2:]), np.max(output[:,:,2:]))
    fig.tight_layout()
    plt.pause(0.1)
#%% microstim choice hist

shown_angle = fix_shown[0]
inds = np.where(shown_deg==shown_angle)[0]

plt.figure(figsize=(5,4))
rcParams['font.size']=11
rcParams['font.sans-serif']='Helvetica'
rcParams['xtick.labelsize'] = 'medium'
rcParams['ytick.labelsize'] = 'medium'
plt.hist(choice_deg[inds], bins=np.arange(0, 360, 15), color='mediumblue')
plt.xlabel('Chosen angle (degrees)')
plt.ylabel('Number of trials')
ymax=200
plt.vlines(shown_angle, 0, ymax, label=f'shown ({shown_angle} deg)', color='k', ls='--')
plt.vlines(stimulated_angle, 0, ymax, label=f'stimulated ({stimulated_angle} deg)', color='r', ls='--')
plt.title(f'Stimulate unit {M_mstim_units}\nstrength={M_mstim_strength}, coh={coh[0]}')
plt.legend(loc='best')
plt.tight_layout()

plt.savefig(f'./{model_name}_mstimChoiceHist_stimU{M_mstim_units}pref{stimulated_angle}s{M_mstim_strength}_shown{shown_angle}rew{fix_reward[0]}coh{coh[0]}_noisein{in_noise}rec{rec_noise}.png', dpi=300)

#%% find pref dirs

model_name = 'MTbroadsharp_m10'
loaded_weights = dict(np.load(f'./saved_weights/{model_name}.npz', allow_pickle=True))
N_rec = loaded_weights['W_rec'].shape[0]
in_noise = 0.4
rec_noise = 0.2
coh = [0.6]
fix_onset=[100, 500]
shownDirs_ = np.arange(0, 360, 5)
tuningMat_shown = np.zeros((N_rec, shownDirs_.shape[0]))
for shownDir in shownDirs_:
    print(shownDir)
    task = MT_broadInSharpOut_withR(N_batch=200, N_rec=N_rec, in_noise=in_noise, coh=coh, k_in=0.3, k_out=0.8, fix_shown=[shownDir], fix_onset=fix_onset)

    network_params = task.get_task_params()
    network_params['name'] = model_name
    network_params['N_rec'] = N_rec
    network_params['rec_noise'] = rec_noise
    
    test_inputs, _, _, _ = task.get_trial_batch()

    simulator = BasicSimulator_linOut(weights=loaded_weights, params=network_params)
    output, state_var = simulator.run_trials(test_inputs)

    fr_stim = np.maximum(state_var[:, fix_onset[1]//10:, :], 0)
    avgFR = np.mean(np.mean(fr_stim, axis=1), axis=0)
    tuningMat_shown[:, shownDir//(360//shownDirs_.shape[0])] = avgFR
    
savefile = open(f'./{model_name}_tuningMat_shown_coh{coh[0]}noisein{in_noise}rec{rec_noise}.pickle','wb')
pickle.dump(tuningMat_shown, savefile, protocol=4)
savefile.close()

#%% plot pref dirs

model_name = 'MTbroadsharp_m10'
folder = 'MTbroadsharp_m10'

loaded_weights = dict(np.load(f'./saved_weights/{model_name}.npz', allow_pickle=True))
N_rec, dale = loaded_weights['W_rec'].shape[0], float(loaded_weights['dale_ratio'])
E_units = np.arange(int(N_rec*dale))
I_units = np.arange(int(N_rec*dale), N_rec)
all_units = np.arange(N_rec)
tuningMat_shown = pickle.load(open(f'./{folder}/{model_name}_tuningMat_shown_coh{coh[0]}noisein{in_noise}rec{rec_noise}.pickle','rb'))
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
plt.xlabel('Shown direction (degrees)')
plt.xticks(ticks=np.arange(0, tuningMat_shown.shape[1], 60//ds), labels=np.arange(0,360,60))
plt.ylabel('Recurrent unit')
plt.gca().xaxis.tick_bottom()
#plt.savefig(f'./{folder}/{model_name}_tuningMat_shown_sortedEandI_coh{coh[0]}noisein{in_noise}rec{rec_noise}.png', dpi=300)
#%% tuning df 
model_name = 'MTbroadsharp_m10'
folder = 'MTbroadsharp_m10'
tuningMat_shown = pickle.load(open(f'./{folder}/{model_name}_tuningMat_shown_coh0.6noisein0.4rec0.2.pickle','rb'))

maxAvgFR = np.max(tuningMat_shown, axis=1)
pref_dirs = np.argmax(tuningMat_shown, axis=1) * 5
height = np.max(tuningMat_shown, axis=1) - np.min(tuningMat_shown, axis=1)

tuning_df_shown = pd.DataFrame()
tuning_df_shown['maxAvgFR'] = maxAvgFR
tuning_df_shown['pref_dirs'] = pref_dirs
tuning_df_shown['height'] = height
#%% plot single unit tuning

unit = 37
plt.figure(figsize=(5,4))
plt.plot(tuningMat_shown[unit, :])
plt.title(f"Unit {unit} tuning (from responses)\npref dir {tuning_df_shown.loc[unit, 'pref_dirs']}")
plt.xticks(np.arange(0,72,10), 5*np.arange(0,72,10))

plt.savefig(f'./{folder}/{model_name}_u{unit}tuningFromResponses.png', dpi=300)

#%% analyze losses

loss = np.load(open('./MTbroadsharp_m9/MTbroadsharp_m9_losses.npy', 'rb'))
plt.plot(loss)
