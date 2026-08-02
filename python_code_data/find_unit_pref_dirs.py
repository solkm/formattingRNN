#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find each recurrent unit's preferred direction in the MT-like and dlPFC-like
modules, by multiple methods, for selecting/describing units for
microstimulation experiments (see generate_microstim_testdata.py).

Uses the existing all-conditions, zero-noise test data in figure_code/testdata.
Note: the units' preferred directions as hard-coded in generate_microstim_testdata.py 
match exactly for the DLPFC weight-based tuning, and within 5 degrees for the 
MT shown-direction tuning (since the latter was determined previously from a simulation with noise).

For each module, computes:
- weight_pref: which of the 72 direction-tuned output channels an excitatory
  unit's (Dale's-law-effective) output weights favor most. NaN for inhibitory
  units, whose output weights are zero.
- shown_pref: which shown motion direction elicits a unit's highest average
  firing rate (at the end of the trial).

For the dlPFC-like module only, additionally computes:
- choice_pref: which of the model's own choices elicits a unit's highest
  average firing rate.

Saves MT_tuning_df.csv and DLPFC_tuning_df.csv to figure_code/testdata, and
plots a sorted tuning matrix for each metric (units ordered
excitatory-then-inhibitory, each sorted by preferred direction).

@author: Sol
"""
from pathlib import Path
import os
ROOT_DIR = Path('/Users/Sol/Desktop/CohenLab/DotsBehavior/formattingRNN/python_code_data')
os.chdir(ROOT_DIR)

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

#%% LOAD DATA
SAVED_WEIGHTS_PATH = './model_weights'
TESTDATA_PATH = './figure_code/testdata'
test_name = 'DLPFCcombined_m4_allCondsNoNoise_coh0.6'

MT_weights = dict(np.load(f'{SAVED_WEIGHTS_PATH}/MTbroadsharp_m10.npz', allow_pickle=True))
DLPFC_weights = dict(np.load(f'{SAVED_WEIGHTS_PATH}/DLPFCcombined_m4.npz', allow_pickle=True))

state_var_MT = pickle.load(open(f'{TESTDATA_PATH}/{test_name}_MTstatevar.pickle', 'rb'))
state_var_DLPFC = pickle.load(open(f'{TESTDATA_PATH}/{test_name}_DLPFCstatevar.pickle', 'rb'))
output_DLPFC = pickle.load(open(f'{TESTDATA_PATH}/{test_name}_DLPFCoutput.pickle', 'rb'))
trial_params = pickle.load(open(f'{TESTDATA_PATH}/{test_name}_trialparams.pickle', 'rb'))

fr_MT = np.maximum(state_var_MT[:, -1, :], 0)      # firing rate (ReLU), end of trial
fr_DLPFC = np.maximum(state_var_DLPFC[:, -1, :], 0)

shown_degs = np.array([trial_params[i]['shown_deg'] for i in range(trial_params.shape[0])])
N_dirOuts_DLPFC = output_DLPFC.shape[2]
choice_degs = np.argmax(output_DLPFC[:, -1, :], axis=1) * 360 / N_dirOuts_DLPFC

unique_shown = np.unique(shown_degs)
unique_choice = np.unique(choice_degs)

#%% HELPER FUNCTIONS

def tuning_matrix_from_activity(fr, condition_vals, unique_conditions):
    """(N_rec, N_conditions) matrix of each unit's mean firing rate per condition."""
    tuning_mat = np.zeros((fr.shape[1], unique_conditions.shape[0]))
    for i, cond in enumerate(unique_conditions):
        inds = np.where(condition_vals == cond)[0]
        tuning_mat[:, i] = np.mean(fr[inds, :], axis=0)
    return tuning_mat

def plot_sorted_tuning(tuning_mat, angle_labels, e_units, i_units, title):
    """Plot units x direction-bin tuning matrix, sorted E-then-I by preferred direction.

    Returns each unit's preferred direction (angle_labels[argmax]). For units
    excluded from both e_units/i_units (e.g. inhibitory units in a weight-based
    tuning matrix, where the preference is not meaningful), pass an empty
    i_units/e_units array as appropriate -- the returned value for those units
    should be overwritten by the caller.
    """
    pref = angle_labels[np.argmax(tuning_mat, axis=1)]
    order = np.concatenate([e_units[np.argsort(pref[e_units])], i_units[np.argsort(pref[i_units])]])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.matshow(tuning_mat[order, :])
    fig.colorbar(im, shrink=0.5)
    ax.set_title(title)
    ax.set_xlabel('Direction (degrees)')
    ticks = np.arange(0, angle_labels.shape[0], max(1, angle_labels.shape[0]//6))
    ax.set_xticks(ticks)
    ax.set_xticklabels(angle_labels[ticks].astype(int))
    ax.set_ylabel('Recurrent unit (sorted)')
    ax.xaxis.set_ticks_position('bottom')
    fig.tight_layout()
    return pref

#%% MT MODULE
N_rec_MT = MT_weights['W_rec'].shape[0]
dale_MT = float(MT_weights['dale_ratio'])
E_units_MT = np.arange(int(N_rec_MT*dale_MT))
I_units_MT = np.arange(int(N_rec_MT*dale_MT), N_rec_MT)
empty_units = np.array([], dtype=int)

# Shown-direction response tuning (both E and I units)
tuningMat_shown_MT = tuning_matrix_from_activity(fr_MT, shown_degs, unique_shown)
shown_pref_MT = plot_sorted_tuning(tuningMat_shown_MT, unique_shown, E_units_MT, I_units_MT,
                                   'MT: tuning to shown direction')

# Output-weight tuning (E units only; excludes the 2 reward-estimate output channels)
motion_W_out_MT = MT_weights['W_out'][2:, :].T  # (N_rec, 72)
weight_pref_MT = plot_sorted_tuning(motion_W_out_MT, unique_shown, E_units_MT, empty_units,
                                     'MT: output-weight tuning (E units)')
weight_pref_MT[I_units_MT] = np.nan

MT_tuning_df = pd.DataFrame({'unit': np.arange(N_rec_MT), 'shown_pref': shown_pref_MT, 'weight_pref': weight_pref_MT})
MT_tuning_df.to_csv(f'{TESTDATA_PATH}/MT_tuning_df.csv', index=False)

#%% DLPFC MODULE
N_rec_D = DLPFC_weights['W_rec'].shape[0]
dale_D = float(DLPFC_weights['dale_ratio'])
E_units_D = np.arange(int(N_rec_D*dale_D))
I_units_D = np.arange(int(N_rec_D*dale_D), N_rec_D)

# Shown-direction response tuning
tuningMat_shown_D = tuning_matrix_from_activity(fr_DLPFC, shown_degs, unique_shown)
shown_pref_D = plot_sorted_tuning(tuningMat_shown_D, unique_shown, E_units_D, I_units_D,
                                  'dlPFC: tuning to shown direction')

# Choice response tuning
tuningMat_choice_D = tuning_matrix_from_activity(fr_DLPFC, choice_degs, unique_choice)
choice_pref_D = plot_sorted_tuning(tuningMat_choice_D, unique_choice, E_units_D, I_units_D,
                                   'dlPFC: tuning to choice')

# Output-weight tuning (E units only)
W_out_D = DLPFC_weights['W_out'].T  # (N_rec, 72)
weight_pref_D = plot_sorted_tuning(W_out_D, unique_shown, E_units_D, empty_units,
                                    'dlPFC: output-weight tuning (E units)')
weight_pref_D[I_units_D] = np.nan

DLPFC_tuning_df = pd.DataFrame({'unit': np.arange(N_rec_D), 'shown_pref': shown_pref_D,
                                 'weight_pref': weight_pref_D, 'choice_pref': choice_pref_D})
DLPFC_tuning_df.to_csv(f'{TESTDATA_PATH}/DLPFC_tuning_df.csv', index=False)

# %%
