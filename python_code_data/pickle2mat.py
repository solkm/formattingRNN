import numpy as np
import pickle
import scipy.io as sio

data_folder = './python_code_data/figure_code/testdata'
testname = 'DLPFCcombined_m4_allCondsNoNoise_coh0.6'

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def dict_array_to_dict(dict_array, ordered_keys=None):
    """Convert array/list of dicts with the same keys to a single dict."""
    keys = dict_array[0].keys() if ordered_keys is None else ordered_keys
    return {k: [d[k] for d in dict_array] for k in keys}

trial_params = load_pickle(f'{data_folder}/{testname}_trialparams.pickle')
tp_keys = ['trial', 'shown_deg', 'good_deg', 't_ring', 't_dots']
trial_params_dict = dict_array_to_dict(trial_params, tp_keys)
mt_statevar = load_pickle(f'{data_folder}/{testname}_MTstatevar.pickle')
dlpfc_statevar = load_pickle(f'{data_folder}/{testname}_DLPFCstatevar.pickle')
dlpfc_input = load_pickle(f'{data_folder}/{testname}_DLPFCinput.pickle')
dlpfc_output = load_pickle(f'{data_folder}/{testname}_DLPFCoutput.pickle')

#%% Save as .mat file
save_folder = './matlab_code_data/modeltestdata'
sio.savemat(f'{save_folder}/{testname}.mat', 
            {'trial_params': trial_params_dict, 
             'mt_statevar': mt_statevar, 
             'dlpfc_statevar': dlpfc_statevar, 
             'dlpfc_input': dlpfc_input, 
             'dlpfc_output': dlpfc_output})
#%%