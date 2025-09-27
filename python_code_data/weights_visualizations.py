#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:06:41 2022

@author: Sol
"""
import sys
sys.path.append('/Users/Sol/Desktop/CohenLab/DynamicTaskPerceptionProject')
sys.path.append('/Users/Sol/Desktop/CohenLab/DotsBehavior')
import numpy as np
import visfunctions as vf
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
import pickle
import copy
from matplotlib.colors import Normalize
import pandas as pd

def L2_weight_reg(L2_in, L2_rec, L2_out, weights):
    reg_in = L2_in * np.mean(np.square(weights['W_in']))
    reg_rec = L2_rec * np.mean(np.square(weights['W_rec']))
    reg_out = L2_out * np.mean(np.square(weights['W_rec']))
    return reg_in, reg_rec, reg_out

def L1_weight_reg(L1_in, L1_rec, L1_out, weights):
    reg_in = L1_in * np.mean(np.abs(weights['W_in']))
    reg_rec = L1_rec * np.mean(np.abs(weights['W_rec']))
    reg_out = L1_out * np.mean(np.abs(weights['W_rec']))
    return reg_in, reg_rec, reg_out
#%%
model_name = 'MTbroadsharp_m10'
weights = dict(np.load(f'./saved_weights/{model_name}.npz', allow_pickle=True))

W_out = weights['W_out']
W_in = weights['W_in']
W_rec = weights['W_rec']

dale = float(weights['dale_ratio'])
N_rec = W_rec.shape[0]
E_units = np.arange(int(dale*N_rec))
I_units = np.arange(int(dale*N_rec), N_rec)
all_units = np.arange(N_rec)

#%% plot weights unsorted
vf.plot_weights(W_in, title='Input weights', vnorm=np.max(np.abs(W_in)))
#plt.savefig(f'./{model_name}_Win.png', dpi=300)

vf.plot_weights(W_out, title='Output weights', vnorm=np.max(np.abs(W_out)))
#plt.savefig(f'./{model_name}_Wout.png', dpi=300)

vf.plot_weights(W_rec, title='Recurrent weights', vnorm=np.max(np.abs(W_rec)))
plt.xlabel('From')
plt.ylabel('To')
#plt.savefig(f'./{model_name}_Wrec.png', dpi=300)

#%% sort output weights
N_dirOutUnits = 72
u_nnz_out = np.where(np.sum(W_out, axis=0)!=0)[0]
pref_dirs_out = np.argmax(W_out[-N_dirOutUnits:, u_nnz_out], axis=0)
order_out = u_nnz_out[np.argsort(pref_dirs_out)]

vf.plot_weights(W_out[-N_dirOutUnits:, order_out], vnorm=0.1+np.max(np.abs(W_out[-N_dirOutUnits:])))

plt.yticks(ticks=np.arange(0,N_dirOutUnits,10), labels=np.arange(0,N_dirOutUnits,10)*360//N_dirOutUnits)

plt.title('Output weights, sorted')
plt.ylabel('Output unit direction (deg)')
plt.xlabel('Recurrent unit (E)')
plt.gca().xaxis.tick_bottom()

#plt.savefig(f'./{model_name}_dirOutputWeights_sorted.png', dpi=300)
#%% sort input weights
N_dirInUnits = 72
pref_dirs_in = np.concatenate((np.argmax(W_in[E_units, -N_dirInUnits-1:-1], axis=1), np.argmax(W_in[I_units, -N_dirInUnits-1:-1], axis=1)))
order_in = np.concatenate((np.argsort(pref_dirs_in[:E_units.shape[0]]), E_units.shape[0] + np.argsort(pref_dirs_in[E_units.shape[0]:])))
vf.plot_weights(W_in[order_in,  -N_dirInUnits-1:-1], vnorm=0.6)
plt.ylabel('Recurrent unit sorted by input preference')
plt.xlabel('Direction (deg)')
plt.xticks(np.arange(0,N_dirInUnits,10), 360//N_dirInUnits * np.arange(0,N_dirInUnits,10))
plt.title('Input weights')

plt.savefig(f'./{model_name}_dirInputWeights_sorted.png', dpi=300)

#%% sort input weights to E units by output prefs
vf.plot_weights(W_in[order_out, -N_dirInUnits-1:-1], vnorm=np.max(np.abs(W_in[order_out, -N_dirInUnits-1:-1])))
plt.ylabel('Recurrent unit (E)\nsorted by output preference')
plt.xlabel('Direction (deg)')
plt.xticks(np.arange(0,N_dirInUnits,10), 360//N_dirInUnits * np.arange(0,N_dirInUnits,10))
plt.title('Input weights')

#plt.savefig(f'./{model_name}_dirInputWeights_sortedbyOut.png', dpi=300)

#%% plot recurrent weights by order from input weights
rcParams['font.size'] = 11
vf.plot_weights(W_rec[:, order_in][order_in, :], vnorm=np.max(np.abs(W_rec[:, order_in][order_in, :])))
plt.xlabel('From')
plt.ylabel('To')
plt.title('Recurrent weights\nsorted by preferred direction input')

plt.savefig(f'./{model_name}_RecWeights_sortedByInWeights.png', dpi=300)

#%% plot excitatory rec weights by order from output weights
rcParams['font.size'] = 11
vf.plot_weights(W_rec[:, order_out][order_out, :], vnorm=np.max(np.abs(W_rec[:, order_out][order_out, :])))
plt.xlabel('From')
plt.ylabel('To')
plt.title('Recurrent weights (E)\nsorted by preferred direction output')
plt.savefig(f'./{model_name}_RecWeightsE_sortedByOutWeights.png', dpi=300)

#%% tuning df from output weights
tuningDf = pd.DataFrame()
max_Wout = np.max(W_out[-N_dirOutUnits:], axis=0)
mean_Wout = np.mean(W_out[-N_dirOutUnits:], axis=0)
inds = np.where(max_Wout > 0.1)[0]

tuningDf['unit_index'] = inds
tuningDf['pref_dir'] = pref_dirs_out[inds] * 360/(N_dirOutUnits)
tuningDf['max_weight'] = max_Wout[inds]
#tuningDf.to_csv(f'./{model_name}_tunedUnits_outWeights.csv', index=False)
#%% single unit output weight tuning
N_dirOutUnits = 36
plt.figure(figsize=(5,4))
unit = 53
plt.plot(W_out[-N_dirOutUnits:, unit])
plt.ylabel('Output weight')
plt.xlabel('Direction (deg)')
plt.xticks(ticks=np.arange(0,N_dirOutUnits,10), labels=np.arange(0,N_dirOutUnits,10)*360//N_dirOutUnits)
plt.title(f'Unit {unit} tuning to output direction\npreferred={pref_dirs_out[unit]* 360/(N_dirOutUnits)} deg')
plt.tight_layout()
#plt.savefig(f'./{model_name}_unit{unit}_outputWeightsTuning.png', dpi=300)
#%% single unit input weight tuning (MT)
plt.figure(figsize=(6,5))
unit = 18
plt.plot(W_in[unit, 2:-1])
plt.ylabel('Input weight')
plt.xlabel('Direction (deg)')
plt.xticks(np.arange(0, 72, 10), np.arange(0, 360, 50))
plt.title(f'Unit {unit} tuning to input direction (from weights)')
#plt.savefig(f'./{model_name}_unit{unit}_inputWeightsTuning.png', dpi=300)

#%% get order from responses (FRs vs input)
folder = model_name
tuningDir = 'shown'

tuningMat = pickle.load(open(f'./{folder}/{model_name}_tuningMat_{tuningDir}_coh0.6con1.0noisein0.2rec0.1.pickle','rb'))
pref_dirs_responseE = np.argmax(tuningMat[E_units,:], axis=1)*360//tuningMat.shape[1]
pref_dirs_responseI = np.argmax(tuningMat[I_units,:], axis=1)*360//tuningMat.shape[1]
order_responseE = np.argsort(pref_dirs_responseE)
order_responseI = np.argsort(pref_dirs_responseI)
combined_order_response = np.concatenate((order_responseE, order_responseI + I_units[0]))

#%% plot output weights by order from responses 
vf.plot_weights(W_out[-N_dirOutUnits:, combined_order_response])
plt.yticks(ticks=np.arange(0,N_dirOutUnits,10), labels=np.arange(0,N_dirOutUnits,10)*360//N_dirOutUnits)

plt.title('Output weights, sorted by recurrent layer response to stimulus')
plt.ylabel('Output unit direction (degrees)')
plt.xlabel('Recurrent unit')
#plt.savefig(f'./{model_name}_dirOutputWeights_sortedByTuningMat_shown.png', dpi=300)

#%% single unit shown dir tuning
plt.figure(figsize=(6,5))
unit = 108
plt.plot(tuningMat[unit, :])
plt.ylabel('Output weight')
plt.xlabel('Direction (deg)')
plt.xticks(np.arange(0, 72, 10), np.arange(0, 360, 50))
plt.title(f'Unit {unit} tuning to stimulus input (from average response)')
#plt.savefig(f'./{model_name}_unit{unit}_shownDirTuning.png', dpi=300)

#%% plot recurrent weights by order from responses 
rcParams['font.size'] = 12
vnorm = 0.8
vf.plot_weights(W_rec[:, combined_order_response][combined_order_response, :], vnorm=vnorm)
plt.xlabel('From')
plt.ylabel('To')
plt.title('Recurrent weights\nsorted by preferred direction')
#plt.savefig(f'./{model_name}_RecWeights_sortedByTuningMat_shown.png', dpi=300)

#%% plot input weights by order from responses(DLPFC)
rcParams['font.size'] = 11
plt.matshow(W_in[combined_order_response, :-1], norm=Normalize(vmin=-.5, vmax=.5), cmap='RdBu', aspect=0.15)
plt.xlabel('Input units: 0=cos(reward), 1=sin(reward)\n2=cos(motion), 3=sin(motion)', fontsize=8)
plt.ylabel('Recurrent unit\nsorted by preferred direction', fontsize=8)
plt.title('Input weights')
#plt.savefig(f'./{model_name}_InputWeights_sortedByTuningMat_shown.png', dpi=300)

#%% plot input weights by order from responses(MT)
rcParams['font.size'] = 10
plt.matshow(W_in[combined_order_response, -N_dirInUnits-1:-1], norm=Normalize(vmin=-.5, vmax=.5), cmap='RdBu', aspect=1)
plt.xlabel('Input unit direction')
plt.ylabel('Recurrent unit\nsorted by preferred direction')
plt.xticks(np.arange(0,36,6), np.arange(0,360,60))
plt.title('Input weights')
plt.gca().xaxis.tick_bottom()
#plt.savefig(f'./{model_name}_InputWeights_sortedByTuningMat_shown.png', dpi=300)

#%% histograms of connections

EtoE = W_rec[E_units[0]:E_units[-1]+1, E_units[0]:E_units[-1]+1]
EtoI = W_rec[I_units[0]:I_units[-1]+1, E_units[0]:E_units[-1]+1]
ItoE = W_rec[E_units[0]:E_units[-1]+1, I_units[0]:I_units[-1]+1]
ItoI = W_rec[I_units[0]:I_units[-1]+1, I_units[0]:I_units[-1]+1]


maxE = max(np.max(EtoE), np.max(EtoI))
binsE = np.linspace(0, np.round(maxE+.01, 2), 50)
plt.figure(figsize=(4,4))
plt.hist(EtoE.flatten(), binsE, label='EtoE', color='b', alpha=0.5, edgecolor='k')
plt.hist(EtoI.flatten(), binsE, label='EtoI', color='c', alpha=0.5, edgecolor='k')
plt.legend()
plt.ylabel('Number of connections')
plt.xlabel('Connection strength')
plt.title('MT model excitatory connections')
plt.tight_layout()
#plt.savefig(f'{model_name}_RecExcitatoryHist.png', dpi=300)

minI = min(np.min(ItoE), np.min(ItoI))
binsI = np.linspace(np.round(minI-.01, 2), 0, 50)
plt.figure(figsize=(4,4))
plt.hist(ItoE.flatten(), binsI, label='ItoE', color='orange', alpha=0.5, edgecolor='k')
plt.hist(ItoI.flatten(), binsI, label='ItoI', color='r', alpha=0.5, edgecolor='k')
plt.legend()
plt.ylabel('Number of connections')
plt.xlabel('Connection strength')
plt.title('MT model inhibitory connections')
plt.tight_layout()
#plt.savefig(f'{model_name}_RecInhibitoryHist.png', dpi=300)

#%% projections - two-area model
folder = ''
model_name = ''

sensE = slice(0, int(dale*N_rec/2))
decE = slice(int(dale*N_rec/2), int(dale*N_rec))
sensI = slice(int(dale*N_rec), int(0.5*(1+dale)*N_rec))
decI = slice(int(0.5*(1+dale)*N_rec), N_rec)

SetoDe = W_rec[sensE, decE]
sumProj_SetoDe = np.sum(SetoDe, axis=1)

SetoDi = W_rec[sensE, decI]
sumProj_SetoDi = np.sum(SetoDi, axis=1)

tuningMat_shown = pickle.load(open(f'./{folder}/{model_name}_tuningMat_shown.pickle','rb'))
maxFR_Se = np.max(tuningMat_shown[:120], axis=1)

plt.figure(figsize=(6,5))
plt.scatter(sumProj_SetoDe, maxFR_Se[:120], s=15, alpha=0.8, label='E')
plt.scatter(sumProj_SetoDi, maxFR_Se[:120], s=15, alpha=0.8, label='I')
plt.xlabel('Summed projections from decision module')
plt.ylabel('Maximum average firing rate')
plt.title('Sensory units tuning vs outgoing projections')
plt.legend()
plt.tight_layout()

#plt.savefig(f'./{model_name}_sensoryTuningvOutProj_DtoSe.png', dpi=300)

#%% perturb weights - two-area model
model_name = 'dots_m9'
weights = dict(np.load(f'./saved_weights/{model_name}.npz'))
DetoDe80 = copy.deepcopy(weights)
DetoDe80['W_rec'][120:240, 120:240] *= 0.8

np.savez(f'./{model_name}_DetoDe80.npz', **DetoDe80)

#%% perturb weights - DLPFC model
model_name = 'dotsDLPFC_m2'
weights = dict(np.load(f'./saved_weights/{model_name}.npz'))
perturbed = copy.deepcopy(weights)

EtoE_coeff = 1
ItoI_coeff = 1
ItoE_coeff = 1
EtoI_coeff = 1.2

perturbed['W_rec'][:120, :120] *= EtoE_coeff
perturbed['W_rec'][120:, 120:] *= ItoI_coeff
perturbed['W_rec'][:120, 120:] *= ItoE_coeff
perturbed['W_rec'][120:, :120] *= EtoI_coeff

if EtoE_coeff != 1 and ItoI_coeff == 1 and ItoE_coeff==1 and EtoI_coeff==1:
    np.savez(f'./saved_weights/{model_name}_EtoE{int(EtoE_coeff*100)}.npz', **perturbed)
elif ItoI_coeff != 1 and EtoE_coeff == 1 and ItoE_coeff==1 and EtoI_coeff==1:
    np.savez(f'./saved_weights/{model_name}_ItoI{int(ItoI_coeff*100)}.npz', **perturbed)
elif  ItoE_coeff!=1 and ItoI_coeff == 1 and EtoE_coeff == 1 and EtoI_coeff==1:
    np.savez(f'./saved_weights/{model_name}_ItoE{int(ItoE_coeff*100)}.npz', **perturbed)   
elif  EtoI_coeff!=1 and ItoI_coeff == 1 and EtoE_coeff == 1 and ItoE_coeff==1:
    np.savez(f'./saved_weights/{model_name}_EtoI{int(EtoI_coeff*100)}.npz', **perturbed)   
#%% output tuning vs shown tuning
folder = 'MThighD_m1'
inds = np.where(np.max(W_out[-72:], axis=0)>0.2)[0]
tuningMat_shown = pickle.load(open(f'./{folder}/{model_name}_tuningMat_shown.pickle','rb'))
pref_dirs_shown = np.argmax(tuningMat_shown, axis=1)

plt.plot(np.arange(360), np.arange(360))
for i in range(inds.shape[0]):
    plt.scatter(pref_dirs_out[inds[i]]*5, pref_dirs_shown[E_units][inds[i]]*5)
    plt.text(pref_dirs_out[inds[i]]*5 - 3, pref_dirs_shown[E_units][inds[i]]*5 - 2, f'{inds[i]}', fontsize=6)

plt.xlabel('preferred direction, from output weights')
plt.ylabel('preferred direction, from response by shown direction')
plt.tight_layout()
#plt.savefig(f'./{folder}/{model_name}_tuningOutWeightsVsShownnDir.png', dpi=300)
