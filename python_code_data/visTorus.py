#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 12:39:17 2023

@author: Sol
"""

import numpy as np
import pickle
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
from matplotlib import interactive
interactive(True)
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from cmcrameri import cm as cmc
import circledots_fxns as cf
from matplotlib.lines import Line2D
    
#%% load testdata all conditions

N_dirOuts = 72
model_name = 'DLPFCcombined_m4'
folder = model_name

test_name = f'{model_name}_allCondsNoNoise_coh0.6'
state_var_ac = pickle.load(open(f'./{folder}/' + test_name + '_DLPFCstatevar.pickle', 'rb'))
trial_params_ac = pickle.load(open(f'./{folder}/' + test_name + '_trialparams.pickle', 'rb'))
output_ac = pickle.load(open(f'./{folder}/' + test_name + '_DLPFCoutput.pickle', 'rb'))

choice_ac = np.argmax(output_ac[:, -1, -N_dirOuts:], axis=1)
choice_deg_ac = choice_ac * 360/N_dirOuts

shown_degs_ac = np.array([trial_params_ac[i]['shown_deg'] for i in range(trial_params_ac.shape[0])])
good_degs_ac = np.array([trial_params_ac[i]['good_deg'] for i in range(trial_params_ac.shape[0])])
        
weights = dict(np.load(f'./saved_weights/{model_name}.npz', allow_pickle=True))

savename = f'{model_name}_DLPFClayer_allCondsNoNoise_coh0.6'

#%% all conditions PCA (no averaging)
fr_ac =  np.maximum(state_var_ac[:,:,:],0)
t_pca = fr_ac.shape[1]-1
fr_pca = fr_ac[:, t_pca, :]

X = fr_pca

pca_ac = PCA(n_components=0.9)
pca_ac.fit_transform(X)
print(pca_ac.explained_variance_ratio_)
print(pca_ac.explained_variance_ratio_.shape)

#%% animation 3D, plot several reward conditions, last timepoint

rewConds = np.arange(0, 360, 20)
pca_components = pca_ac.components_[:3]

fig = plt.figure(figsize=(6,6))
rcParams['font.size']=12

ax = fig.add_subplot(projection ='3d')
    
for r in range(rewConds.shape[0]):
    trials = np.where(good_degs_ac==rewConds[r])[0]
    trajectories = np.dot(fr_ac[trials][:,-1, :], pca_components.T)
    color = cmc.romaO(rewConds[r]/360)
    ring = ax.plot(trajectories[:, 0], trajectories[:,1], trajectories[:,2], zdir='z', color=color, label=rewConds[r])
    ax.legend(handles=ring)
    plt.pause(0.5)
    ax.get_legend().remove()
#%% matlab is better for 3D

from scipy.io import savemat

savemat(f'./matlab_code_data/{test_name}.mat', {'DLPFCstatevar':state_var_ac, 'trialparams':trial_params_ac, 'DLPFCoutput':output_ac})









