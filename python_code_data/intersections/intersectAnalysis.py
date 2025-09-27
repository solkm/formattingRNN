#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 15:22:43 2023

@author: Sol
"""

from sklearn.decomposition import PCA
import pickle
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm as cmc
#%% load test data
model_name = 'DLPFCcombined_m4'
folder = 'DLPFCcombined_m4/testdata'

test_name = f'{model_name}_shown1nrew10n_noNoise_coh0.6'
state_var = pickle.load(open(f'./{folder}/' + test_name + '_DLPFCstatevar.pickle', 'rb'))
trial_params = pickle.load(open(f'./{folder}/' + test_name + '_trialparams.pickle', 'rb'))
output = pickle.load(open(f'./{folder}/' + test_name + '_DLPFCoutput.pickle', 'rb'))

choice_degs = 5*np.argmax(output[:, -1, -72:], axis=1)
shown_degs = np.array([trial_params[i]['shown_deg'] for i in range(trial_params.shape[0])])
bias_degs = np.array([trial_params[i]['good_deg'] for i in range(trial_params.shape[0])])
#%% PCA
fr_end =  np.maximum(state_var[:,-1,:], 0)

pca = PCA(n_components=10)
pca.fit_transform(fr_end)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.shape[0])

# reduce dimensionality
dimred = pca.explained_variance_ratio_.shape[0]
fr_end_red = np.dot(fr_end, pca.components_[:dimred, :].T)
#%% distance analysis
unique_shown = np.unique(shown_degs)
unique_bias = np.unique(bias_degs)
intersection = []
distDiff = []

for i in range(unique_bias.shape[0]):
    for j in range(unique_shown.shape[0]):
        same_shown_inds = np.where(shown_degs==unique_shown[j])[0]
        same_bias_inds = np.where(bias_degs==unique_bias[i])[0]
        point_ind = np.intersect1d(same_shown_inds, same_bias_inds)

        same_bias_inds = same_bias_inds[same_bias_inds!=point_ind]
        dists_sameBias = np.sqrt(np.sum(np.square((fr_end_red[same_bias_inds] - fr_end_red[point_ind])), axis=1))
        minDist_sameBias = np.min(dists_sameBias)
        
        diff_bias_inds = np.where(bias_degs!=unique_bias[i])[0]
        dists_diffBias = np.sqrt(np.sum(np.square((fr_end_red[diff_bias_inds] - fr_end_red[point_ind])), axis=1))
        minDist_diffBias = np.min(dists_diffBias)
        
        distDiff.append(minDist_diffBias - minDist_sameBias)
        
        if minDist_diffBias < minDist_sameBias:
            closest_ind = diff_bias_inds[np.argmin(dists_diffBias)]
            intersection.append([(unique_bias[i], unique_shown[j]), (bias_degs[closest_ind], shown_degs[closest_ind]), minDist_diffBias])
#%% 3d plot

fig = plt.figure(figsize=(6,6))

ax = fig.add_subplot(projection ='3d')
    
colors = cmc.romaO(bias_degs/360)
ax.scatter(fr_end_red[:,0], fr_end_red[:,1], fr_end_red[:,2], zdir='z', color=colors, s=1)
