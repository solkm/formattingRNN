#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the trained MT-like module's own behavior, and the combined (MT + dlPFC)
model's behavior, from existing all-conditions, zero-noise test data (see
figure_code/testdata).

@author: Sol
"""
from pathlib import Path
import os
ROOT_DIR = Path('/Users/Sol/Desktop/CohenLab/DotsBehavior/formattingRNN/python_code_data')
os.chdir(ROOT_DIR)

import matplotlib
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import pickle

TESTDATA_PATH = './figure_code/testdata'
coh = 0.6

#%% DLPFC MODULE (combined model): load test data and calculate variables
model_name = 'DLPFCcombined_m4'
loadname = f'{model_name}_allCondsNoNoise_coh{coh}'

loadfile = open(f'{TESTDATA_PATH}/{loadname}_DLPFCoutput.pickle', 'rb')
output = pickle.load(loadfile)
loadfile.close()

loadfile = open(f'{TESTDATA_PATH}/{loadname}_trialparams.pickle', 'rb')
trial_params = pickle.load(loadfile)
loadfile.close()

choice = np.argmax(output[:, -1, :], axis=1)
choice_deg = choice*5 # the model's chosen direction
best_deg = np.array([np.argmax(trial_params[i]['reward'])*5 for i in range(trial_params.shape[0])]) # the most rewarded direction
shown_deg = np.array([trial_params[i]['shown_deg'] for i in range(trial_params.shape[0])]) # the shown motion direction
bias_deg = np.array([trial_params[i]['good_deg'] for i in range(trial_params.shape[0])]) # the reward bias direction

#%% Plot chosen vs shown directions from the loaded test data (combined model)
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

plt.xlabel(r'Shown direction $^{\degree}$')
plt.ylabel(r'Chosen direction $^{\degree}$')
plt.title('Combined model behavior')
plt.legend()
plt.tight_layout()

# plt.savefig(f'./{model_name}_chosenVshown_rew{b1_deg}and{b2_deg}_allCondsNoNoise_coh{coh}.png', dpi=300)

#%% MT MODULE: load test data and calculate variables
mt_model_name = 'MTbroadsharp_m10'
mt_loadname = f'{mt_model_name}_allCondsNoNoise_coh{coh}'

loadfile = open(f'{TESTDATA_PATH}/{mt_loadname}_output.pickle', 'rb')
mt_output = pickle.load(loadfile)
loadfile.close()

loadfile = open(f'{TESTDATA_PATH}/{mt_loadname}_trialparams.pickle', 'rb')
mt_trial_params = pickle.load(loadfile)
loadfile.close()

mt_choice_output = np.mean(mt_output[:, -10:, :], axis=1)

mt_choice_deg = np.argmax(mt_choice_output[:, -72:], axis=1) * 5
mt_shown_deg = np.array([mt_trial_params[i]['shown_deg'] for i in range(mt_trial_params.shape[0])])
mt_bias_deg = np.array([mt_trial_params[i]['good_deg'] for i in range(mt_trial_params.shape[0])])

mt_bias_est = np.degrees(np.arctan2(mt_choice_output[:, 1], mt_choice_output[:, 0]))
mt_bias_est[mt_bias_est<0] += 360

#%% MT module chosen direction (argmax on output activity) vs true shown direction
plt.figure(figsize=(6,6))
plt.plot(np.linspace(0, 360, 10), np.linspace(0, 360, 10), c='k')
plt.scatter(mt_shown_deg, mt_choice_deg, facecolors='b', edgecolors='b', alpha=0.2, s=20)
plt.xlabel('Shown direction (degrees)')
plt.ylabel('Chosen direction (degrees)')
plt.title('MT module behavior: motion outputs')
plt.tight_layout()

#%% MT module reward bias estimate (from output activity) vs true reward bias direction
plt.figure(figsize=(6,6))
plt.plot(np.linspace(0, 360, 5), np.linspace(0, 360, 5), c='k')
plt.scatter(mt_bias_deg, mt_bias_est, facecolors='b', edgecolors='b', alpha=0.2, s=20)
plt.xlabel('Reward bias location (deg)')
plt.ylabel('Reward bias estimate (deg)')
plt.title('MT module behavior: reward bias outputs')
plt.tight_layout()
