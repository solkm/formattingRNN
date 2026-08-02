#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate microstimulation test data for multiple units/conditions in one run,
then aggregate the results into the mstimMT/mstimDLPFC .mat struct format
consumed by the MATLAB microstimulation plotting scripts (modelMicrostimHeatmap.m,
modelMicrostimHeatmap_binned.m, polarPlotsUstimChoiceEx_model.m,
polarPlotsUstimChoiceSummary_model.m).

@author: Sol
"""
from pathlib import Path
import os
ROOT_DIR = Path('/Users/Sol/Desktop/CohenLab/DotsBehavior/formattingRNN/python_code_data')
os.chdir(ROOT_DIR)

from DotsTasks import MT_broadInSharpOut_withR, DLPFC_combined
from psychrnn.backend.simulation import BasicSimulator_linOut, BasicSimulator
import numpy as np
from scipy.io import savemat
import pickle

#%% GENERATE PER-UNIT TEST DATA
SAVED_WEIGHTS_PATH = './model_weights'
MATLAB_SAVE_PATH = '../matlab_code_data/modeltestdata'

model_name = 'DLPFCcombined_m4'
DLPFC_weights = dict(np.load(f'{SAVED_WEIGHTS_PATH}/{model_name}.npz', allow_pickle=True))
DLPFC_Nrec = DLPFC_weights['W_rec'].shape[0]

MTmodel_name = 'MTbroadsharp_m10'
MT_weights = dict(np.load(f'{SAVED_WEIGHTS_PATH}/{MTmodel_name}.npz', allow_pickle=True))
MT_NdirOuts = MT_weights['W_out'].shape[0] - 2
MT_Nrec = MT_weights['W_rec'].shape[0]

# Noise and coherence
coh = [0.6]
in_noise_DLPFC = 0.2
rec_noise_DLPFC = 0.2
in_noise_MT = 0.4
rec_noise_MT = 0.2

# Which module to stimulate, and the units/preferred directions/strengths to test.
# Each unit appears twice: once at strength 0 (no stim) and once at its test strength.
module_to_stim = 'DLPFC' # 'MT' or 'DLPFC'

if module_to_stim == 'MT':
    MT_mstim_units_ = [37, 37, 4, 4, 42, 42, 25, 25, 34, 34]
    MT_mstim_angles_ = [110, 110, 255, 255, 0, 0, 205, 205, 90, 90]
    MT_mstim_strengths_ = [0, 11, 0, 10, 0, 10, 0, 11, 0, 12]
    assert len(MT_mstim_units_) == len(MT_mstim_angles_) == len(MT_mstim_strengths_)
    n_conditions = len(MT_mstim_units_)

elif module_to_stim == 'DLPFC':
    DLPFC_mstim_units_ = [53, 53, 25, 25, 54, 54, 8, 8, 9, 9]
    DLPFC_mstim_angles_ = [295, 295, 160, 160, 15, 15, 140, 140, 75, 75]
    DLPFC_mstim_strengths_ = [0, 45, 0, 45, 0, 45, 0, 45, 0, 48]
    assert len(DLPFC_mstim_units_) == len(DLPFC_mstim_angles_) == len(DLPFC_mstim_strengths_)
    n_conditions = len(DLPFC_mstim_units_)

# Trials per condition, and the reward bias / shown directions to test, both defined
# relative to each stimulated unit's preferred direction.
N_testbatch = 72*250
reward_rel = 180
shown_rel = np.arange(0, 360, 5)

folder = 'mstim_testdata'
os.makedirs(folder, exist_ok=True)
gen_params_label = f'_rewRel{reward_rel}shownRel5n_coh{coh[0]}noisein{in_noise_MT}mt{in_noise_DLPFC}pfc_rec{rec_noise_MT}mt{rec_noise_DLPFC}pfc'

for i in range(n_conditions):
    print(i)
    if module_to_stim == 'MT':
        M_mstim_units = [MT_mstim_units_[i]]
        M_stimulated_angle = MT_mstim_angles_[i]
        M_mstim_strength = MT_mstim_strengths_[i]

        D_mstim_units, D_mstim_strength = None, 0

        fix_shown = (M_stimulated_angle + shown_rel) % 360
        fix_reward = [(M_stimulated_angle + reward_rel) % 360]

        savename = f'./{folder}/{model_name}_stimMTu{M_mstim_units[0]}pref{M_stimulated_angle}s{M_mstim_strength}' + gen_params_label

    elif module_to_stim == 'DLPFC':
        D_mstim_units = [DLPFC_mstim_units_[i]]
        D_stimulated_angle = DLPFC_mstim_angles_[i]
        D_mstim_strength = DLPFC_mstim_strengths_[i]

        M_mstim_units, M_mstim_strength = None, 0

        fix_shown = (D_stimulated_angle + shown_rel) % 360
        fix_reward = [(D_stimulated_angle + reward_rel) % 360]

        savename = f'./{folder}/{model_name}_stimDLPFCu{D_mstim_units[0]}pref{D_stimulated_angle}s{D_mstim_strength}' + gen_params_label

    if os.path.exists(savename+'_DLPFCoutput.pickle'):
        continue

    # define MT simulator
    MT_task = MT_broadInSharpOut_withR(N_batch=N_testbatch, N_rec=MT_Nrec,
                                    in_noise=in_noise_MT, coh=coh,
                                    fix_shown=fix_shown, fix_reward=fix_reward,
                                    M_mstim_strength=M_mstim_strength)
    MT_network_params = MT_task.get_task_params()
    MT_network_params['name'] = MTmodel_name
    MT_network_params['N_rec'] = MT_Nrec
    MT_network_params['rec_noise'] = rec_noise_MT

    MT_weights_i = MT_weights.copy()
    if M_mstim_units is not None:
        M_microstim = np.zeros(MT_Nrec)
        M_microstim[M_mstim_units] = 1
        MT_weights_i['W_in'][:, -1] = M_microstim

    MT_simulator = BasicSimulator_linOut(weights=MT_weights_i, params=MT_network_params)

    # define the combined model task
    task = DLPFC_combined(N_batch=N_testbatch, N_rec=DLPFC_Nrec,
                          in_noise=in_noise_DLPFC, D_mstim_strength=D_mstim_strength,
                          MT_task=MT_task, MT_simulator=MT_simulator, MT_NdirOuts=MT_NdirOuts)
    network_params = task.get_task_params()
    network_params['N_rec'] = DLPFC_Nrec
    network_params['rec_noise'] = rec_noise_DLPFC

    DLPFC_weights_i = DLPFC_weights.copy()
    if D_mstim_units is not None:
        D_microstim = np.zeros(DLPFC_Nrec)
        D_microstim[D_mstim_units] = 1
        DLPFC_weights_i['W_in'][:, -1] = D_microstim

    simulator = BasicSimulator(weights=DLPFC_weights_i, params=network_params)

    inputs, _, _, trial_params = task.get_trial_batch()
    output, _ = simulator.run_trials(inputs)

    # save as .pickle
    savefile = open(savename+'_DLPFCoutput.pickle','wb')
    pickle.dump(output, savefile, protocol=4)
    savefile.close()

    savefile = open(savename+'_trialparams.pickle','wb')
    pickle.dump(trial_params, savefile, protocol=4)
    savefile.close()

    del inputs, trial_params, output, MT_weights_i, DLPFC_weights_i, \
        MT_simulator, simulator, task, MT_task, MT_network_params, network_params

#%% AGGREGATE AND SAVE AS .MAT FILE

units = MT_mstim_units_[::2] if module_to_stim == 'MT' else DLPFC_mstim_units_[::2]

mstim_dict = {'unit':[], 'pref_dir':[], 'strength':[], 'reward_bias':[],
              'motion_dirs':[], 'choices':[], 'output_end':[]}

for i in range(n_conditions):
    if module_to_stim == 'MT':
        mstim_dict['unit'].append(MT_mstim_units_[i])
        mstim_dict['pref_dir'].append(MT_mstim_angles_[i])
        mstim_dict['strength'].append(MT_mstim_strengths_[i])
        mstim_dict['reward_bias'].append((MT_mstim_angles_[i] + reward_rel) % 360)

    elif module_to_stim == 'DLPFC':
        mstim_dict['unit'].append(DLPFC_mstim_units_[i])
        mstim_dict['pref_dir'].append(DLPFC_mstim_angles_[i])
        mstim_dict['strength'].append(DLPFC_mstim_strengths_[i])
        mstim_dict['reward_bias'].append((DLPFC_mstim_angles_[i] + reward_rel) % 360)

    # open pickle files
    savename = f"./{folder}/{model_name}_stim{module_to_stim}u{mstim_dict['unit'][-1]}pref{mstim_dict['pref_dir'][-1]}s{mstim_dict['strength'][-1]}" + gen_params_label
    output = pickle.load(open(savename+'_DLPFCoutput.pickle','rb'))
    trial_params = pickle.load(open(savename+'_trialparams.pickle','rb'))

    # calculate variables
    choice_deg = 5*np.argmax(output[:, -1, :], axis=1)
    shown_deg = np.array([trial_params[i]['shown_deg'] for i in range(trial_params.shape[0])])

    # add to dict
    mstim_dict['motion_dirs'].append(shown_deg)
    mstim_dict['choices'].append(choice_deg)
    mstim_dict['output_end'].append(output[:, -1, :])

savemat(f'{MATLAB_SAVE_PATH}/{model_name}_stim{module_to_stim}u{units}bias{reward_rel}_paramsChoicesOutputs.mat',
        {f'mstim{module_to_stim}':mstim_dict})
