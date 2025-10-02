#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 12:47:45 2023

@author: Sol
"""
#%% Imports
import numpy as np
import pickle
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
from matplotlib import interactive
interactive(True)
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.decomposition import PCA
from cmcrameri import cm as cmc
from matplotlib.lines import Line2D
from pathlib import Path
import sys

root_dir = Path(__file__).parent
sys.path.append(str(root_dir))

#%% Load test data
model_name = 'DLPFCcombined_m4'
folder = f'figure_code/testdata' # edit to your path
test_name = f'{model_name}_allCondsNoNoise_coh0.6'

weights_DLPFC = dict(np.load(f'{root_dir}/model_weights/{model_name}.npz', allow_pickle=True))
N_dirOuts = weights_DLPFC['W_out'].shape[0]

state_var_DLPFC = pickle.load(open(f'{root_dir}/{folder}/' + test_name + '_DLPFCstatevar.pickle', 'rb'))
trial_params = pickle.load(open(f'{root_dir}/{folder}/' + test_name + '_trialparams.pickle', 'rb'))
output_DLPFC = pickle.load(open(f'{root_dir}/{folder}/' + test_name + '_DLPFCoutput.pickle', 'rb'))
state_var_MT= pickle.load(open(f'{root_dir}/{folder}/' + test_name + '_MTstatevar.pickle', 'rb'))

choice_degs = np.argmax(output_DLPFC[:, -1, :], axis=1) * 360 / N_dirOuts
shown_degs = np.array([trial_params[i]['shown_deg'] for i in range(trial_params.shape[0])])
good_degs = np.array([trial_params[i]['good_deg'] for i in range(trial_params.shape[0])])

fr_MT = np.maximum(state_var_MT[:, :, :],0) # apply ReLu
fr_DLPFC = np.maximum(state_var_DLPFC[:, :, :],0) # apply ReLu

unique_shown = np.unique(shown_degs)
unique_bias = np.unique(good_degs)

#%% Averaging over motion/reward conditions

# Average firing rates over trials with same shown direction (marginalizing over reward bias)
fr_MT_shownavg = np.zeros((unique_shown.shape[0], fr_MT.shape[1], fr_MT.shape[2]))
fr_DLPFC_shownavg = np.zeros((unique_shown.shape[0], fr_DLPFC.shape[1], fr_DLPFC.shape[2]))
for i in range(unique_shown.shape[0]):
    inds = np.where(shown_degs == unique_shown[i])[0]
    fr_MT_shownavg[i,:,:] = np.mean(fr_MT[inds,:,:], axis=0)
    fr_DLPFC_shownavg[i, :, :] = np.mean(fr_DLPFC[inds,:,:], axis=0)

# Average firing rates over trials with same reward bias (marginalizing over shown motion direction)
fr_MT_rewardavg = np.zeros((unique_bias.shape[0], fr_MT.shape[1], fr_MT.shape[2]))
fr_DLPFC_rewardavg = np.zeros((unique_bias.shape[0], fr_DLPFC.shape[1], fr_DLPFC.shape[2]))
for i in range(unique_bias.shape[0]):
    inds = np.where(good_degs == unique_bias[i])[0]
    fr_MT_rewardavg[i] = np.mean(fr_MT[inds,:,:], axis=0)
    fr_DLPFC_rewardavg[i] = np.mean(fr_DLPFC[inds,:,:], axis=0)

#%% PCA on averaged activity to get motion and reward dimensions
t_pca = fr_MT.shape[1] - 1 # timepoint for PCA (last timepoint)
n_m_dims = 2 # number of motion dimensions
n_r_dims = 1 # number of reward dimensions

# MT-like module
pca_MT_shownavg = PCA(n_components=0.9)
pca_MT_shownavg.fit_transform(fr_MT_shownavg[:, t_pca, :])
MT_shownavg_var_exp = pca_MT_shownavg.explained_variance_ratio_
print(MT_shownavg_var_exp)

pca_MT_rewardavg = PCA(n_components=0.9)
pca_MT_rewardavg.fit_transform(fr_MT_rewardavg[:, t_pca, :])
MT_rewardavg_var_exp = pca_MT_rewardavg.explained_variance_ratio_
print(MT_rewardavg_var_exp)

MT_MRdims = np.concatenate((pca_MT_shownavg.components_[:n_m_dims, :], # motion dimensions
                            pca_MT_rewardavg.components_[:n_r_dims, :]), axis=0) # reward dimensions

# DLPFC-like module
pca_DLPFC_shownavg = PCA(n_components=0.9)
pca_DLPFC_shownavg.fit_transform(fr_DLPFC_shownavg[:, t_pca, :])
DLPFC_shownavg_var_exp = pca_DLPFC_shownavg.explained_variance_ratio_
print(DLPFC_shownavg_var_exp)

pca_DLPFC_rewardavg = PCA(n_components=0.9)
pca_DLPFC_rewardavg.fit_transform(fr_DLPFC_rewardavg[:, t_pca, :])
DLPFC_rewardavg_var_exp = pca_DLPFC_rewardavg.explained_variance_ratio_
print(DLPFC_rewardavg_var_exp)

DLPFC_MRdims = np.concatenate((pca_DLPFC_shownavg.components_[:n_m_dims, :], # motion dimensions
                               pca_DLPFC_rewardavg.components_[:n_r_dims, :]), axis=0) # reward dimensions

#%% Save activity in M & R dims to mat file
# import scipy.io as sio
# save_folder = './matlab_code_data/modeltestdata'
# sio.savemat(f'{save_folder}/{test_name}_MRdims.mat', 
#             {'MT_fr_MRdims': fr_MT[:, -1, :] @ MT_MRdims.T,
#              'DLPFC_fr_MRdims': fr_DLPFC[:, -1, :] @ DLPFC_MRdims.T,
#              'shown_degs': shown_degs,
#              'good_degs': good_degs,
#              'choice_degs': choice_degs,
#              'MT_MRdims': MT_MRdims,
#              'DLPFC_MRdims': DLPFC_MRdims,
#             })

#%% Motion and reward dimensions plot, 2 reward conditions: MT
dim = 3 # 2D or 3D
rew_conds = [60, 240]
trials = np.stack((np.where(good_degs==rew_conds[0])[0], np.where(good_degs==rew_conds[1])[0]))

color_shown = True # colored by shown directions if True, by chosen directions if False
marker = 'o'
size = 20
alpha = 1
t = t_pca

# Plot condition 1
fig = plt.figure(figsize=(4,4))
edgecolors1 = None
colors1 = cmc.romaO(shown_degs[trials[0]]/360) if color_shown==True else cmc.romaO(choice_degs[trials[0]]/360)

traj1 = np.dot(fr_MT[trials[0]][:, t, :], MT_MRdims[:dim].T)
if dim==3:
    rcParams['grid.color'] = 'gainsboro'
    ax = fig.add_subplot(projection='3d')
    ax.scatter(traj1[:, 0], traj1[:, 1], traj1[:, 2], marker=marker, zdir='z', color=colors1, s=size, edgecolor=edgecolors1, alpha=alpha)
elif dim==2:
    ax = fig.gca()
    ax.scatter(traj1[:, 0], traj1[:, 1], marker=marker, color=colors1, s=size, edgecolor=edgecolors1, alpha=alpha)

# Plot condition 2
edgecolors2 = cmc.romaO(shown_degs[trials[1]]/360) if color_shown==True else cmc.romaO(choice_degs[trials[1]]/360)
colors2 = 'none'
lw = 1.2

traj2 = np.dot(fr_MT[trials[1]][:, t, :], MT_MRdims[:dim].T)

if dim==3:
    ax.scatter(traj2[:, 0], traj2[:, 1], traj2[:, 2], marker=marker, zdir='z', color=colors2, s=size, edgecolor=edgecolors2, linewidths=lw, alpha=alpha)
elif dim==2:
    ax = fig.gca()
    ax.scatter(traj2[:, 0], traj2[:, 1], marker=marker, color=colors2, s=size, edgecolor=edgecolors2, linewidths=lw, alpha=alpha)

# Labels and legend
rcParams['font.size']=12
plt.title('MT')
plt.xlabel('motion dim 1')
plt.ylabel('motion dim 2')
if dim==3:
    ax.set_zlabel('reward dim 1')
    ax.set_zticks([0, 1, 2, 3])
ax.set_aspect('equal')
plt.tick_params(labelsize=10)

plt.tight_layout()

label1 = f'{rew_conds[0]} deg bias'
label2 = f'{rew_conds[1]} deg bias'
legend_elements = [Line2D([0], [0], marker='o', color='w', label=label1, markerfacecolor=[0.2,0.2,0.2], markeredgecolor='none', markersize=8), 
                   Line2D([0], [0], marker='o', color='w', label=label2, markerfacecolor='none', markeredgecolor=[0.2,0.2,0.2], markeredgewidth=2, markersize=8)]
plt.legend(handles=legend_elements, loc='best')

# plt.savefig(f'./{model_name}/eps_figs/MT_MRdims_{dim}d_bias{rew_conds[0]}and{rew_conds[1]}.eps', format='eps')

#%% Motion and reward dimensions plot, 2 reward conditions: DLPFC

dim = 3 # 2D or 3D
rew_conds = [60, 240]
trials = np.stack((np.where(good_degs==rew_conds[0])[0], np.where(good_degs==rew_conds[1])[0]))

color_shown = True # colored by shown directions if True, by chosen directions if False
marker = 'o'
size = 20
alpha = 1
t = t_pca

# plot condition 1
fig = plt.figure(figsize=(4,4))
edgecolors1 = None
colors1 = cmc.romaO(shown_degs[trials[0]]/360) if color_shown==True else cmc.romaO(choice_degs[trials[0]]/360)

traj1 = np.dot(fr_DLPFC[trials[0]][:, t, :], DLPFC_MRdims[:dim].T)
if dim==3:
    rcParams['grid.color'] = 'gainsboro'
    ax = fig.add_subplot(projection='3d')
    ax.scatter(traj1[:, 0], traj1[:, 1], traj1[:, 2], marker=marker, zdir='z', color=colors1, s=size, edgecolor=edgecolors1, alpha=alpha)
elif dim==2:
    ax = fig.gca()
    ax.scatter(traj1[:, 0], traj1[:, 1], marker=marker, color=colors1, s=size, edgecolor=edgecolors1, alpha=alpha)

# plot condition 2
edgecolors2 = cmc.romaO(shown_degs[trials[1]]/360) if color_shown==True else cmc.romaO(choice_degs[trials[1]]/360)
colors2 = 'none'
lw = 1.2

traj2 = np.dot(fr_DLPFC[trials[1]][:, t, :], DLPFC_MRdims[:dim].T)

if dim==3:
    ax.scatter(traj2[:, 0], traj2[:, 1], traj2[:, 2], marker=marker, zdir='z', color=colors2, s=size, edgecolor=edgecolors2, linewidths=lw, alpha=alpha)
elif dim==2:
    ax = fig.gca()
    ax.scatter(traj2[:, 0], traj2[:, 1], marker=marker, color=colors2, s=size, edgecolor=edgecolors2, linewidths=lw, alpha=alpha)

# labels and legend
rcParams['font.size']=12
plt.title('DLPFC')
plt.xlabel('motion dim 1')
plt.ylabel('motion dim 2')
if dim==3:
    ax.set_zlabel('reward dim 1')
    ax.set_zticks(np.arange(-10, 21, 10))
ax.set_aspect('equal')
plt.tick_params(labelsize=10)

plt.tight_layout()

label1 = f'{rew_conds[0]} deg bias'
label2 = f'{rew_conds[1]} deg bias'
legend_elements = [Line2D([0], [0], marker='o', color='w', label=label1, markerfacecolor=[0.2,0.2,0.2], markeredgecolor='none', markersize=8), 
                   Line2D([0], [0], marker='o', color='w', label=label2, markerfacecolor='none', markeredgecolor=[0.2,0.2,0.2], markeredgewidth=2, markersize=8)]
plt.legend(handles=legend_elements, loc='best')

#plt.savefig(f'./{model_name}/eps_figs/DLPFC_MRdims_{dim}d_bias{rew_conds[0]}and{rew_conds[1]}.eps', format='eps')

#%% Normal PCA, no averaging

t_pca = fr_MT.shape[1] - 1 # timepoint for PCA (last timepoint)

pca_MT = PCA(n_components=0.9)
pca_MT.fit_transform(fr_MT[:, t_pca, :])
print(pca_MT.explained_variance_ratio_)

pca_DLPFC = PCA(n_components=0.9)
pca_DLPFC.fit_transform(fr_DLPFC[:, t_pca, :])
print(pca_DLPFC.explained_variance_ratio_)

#%% PC plot, all conditions: MT

dim = 3 # 2D or 3D

color_shown = True # colored by shown directions if True, by chosen directions if False
marker = 'o'
size = 15
alpha = 0.8
t = t_pca

# plot condition 1
fig = plt.figure(figsize=(4,4))
colors = cmc.romaO(shown_degs/360) if color_shown==True else cmc.romaO(choice_degs/360)

traj1 = np.dot(fr_MT[:, t, :], pca_MT.components_[:dim].T)
if dim==3:
    rcParams['grid.color'] = 'gainsboro'
    ax = fig.add_subplot(projection='3d')
    ax.scatter(traj1[:, 0], traj1[:, 1], traj1[:, 2], marker=marker, zdir='z', color=colors, s=size, alpha=alpha)
elif dim==2:
    ax = fig.gca()
    ax.scatter(traj1[:, 0], traj1[:, 1], marker=marker, color=colors, s=size, alpha=alpha)

# labels and legend
rcParams['font.size']=12
plt.title('MT')
plt.xlabel('PC1')
plt.ylabel('PC2')
if dim==3:
    ax.set_zlabel('PC3')
ax.set_aspect('equal')
plt.tick_params(labelsize=10)

plt.tight_layout()

#plt.savefig(f'./{model_name}/eps_figs/MT_{dim}d_allconditions.eps', format='eps')

#%% PC plot, all conditions: DLPFC

dim = 3 # 2D or 3D

color_shown = True # colored by shown directions if True, by chosen directions if False
marker = 'o'
size = 15
alpha = 0.8
t = t_pca

# plot condition 1
fig = plt.figure(figsize=(4,4))
colors = cmc.romaO(shown_degs/360) if color_shown==True else cmc.romaO(choice_degs/360)

traj1 = np.dot(fr_DLPFC[:, t, :], pca_DLPFC.components_[:dim].T)
if dim==3:
    rcParams['grid.color'] = 'gainsboro'
    ax = fig.add_subplot(projection='3d')
    ax.scatter(traj1[:, 0], traj1[:, 1], traj1[:, 2], marker=marker, zdir='z', color=colors, s=size, alpha=alpha)
elif dim==2:
    ax = fig.gca()
    ax.scatter(traj1[:, 0], traj1[:, 1], marker=marker, color=colors, s=size, alpha=alpha)

# labels and legend
rcParams['font.size']=12
plt.title('DLPFC')
plt.xlabel('PC1')
plt.ylabel('PC2')
if dim==3:
    ax.set_zlabel('PC3')
ax.set_aspect('equal')
plt.tick_params(labelsize=10)
plt.tight_layout()

# plt.savefig(f'./{model_name}/eps_figs/DLPFC_{dim}d_allconditions.eps', format='eps')