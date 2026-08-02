# formattingRNN

# Overview

This repo contains code for the recurrent neural network (RNN) modeling component of the preprint:

> Linking neural population formatting to function. Douglas A. Ruff, Sol K. Markman, Jason Z. Kim, Marlene R. Cohen. bioRxiv 2025.01.03.631242; doi: https://doi.org/10.1101/2025.01.03.631242

Published in final form as:

> (Currently in review)
> See the `manuscript` folder for the most recent version compatible with the descriptions below (`manuscript_c.pdf`, supplementary methods in `manuscript_c_supp.pdf`).

The model is a two-module RNN: an MT-like module trained to output a sharpened motion-direction estimate and the reward-center location from noisy motion and reward-condition inputs, and a dlPFC-like module, driven by the (frozen) MT-like module's output, trained to output the expected reward associated with each possible choice. See Supplementary Methods section 6 ("Recurrent neural network model") for the full mathematical description.

# Data

Simulated model test data (used by the figure-generating scripts below) is too large for this repository and is hosted separately on Zenodo:

> https://doi.org/10.5281/zenodo.21764173

Download the two files there, unzip and add their contents into the following folders in the repo:
- `python_model_data.zip` → `python_code_data/figure_code/testdata/`
- `matlab_model_data.zip` → `matlab_code_data/modeltestdata/`

Trained model weights (`model_weights/*.npz`) are small enough to be included directly in this repo.

# Resources

This project was developed using the PsychRNN package:
> Documentation: https://psychrnn.readthedocs.io/en/latest/
> GitHub: https://github.com/murraylab/PsychRNN
> Paper: https://www.eneuro.org/content/8/1/ENEURO.0427-20.2020.

All code needed for this project is included here in `python_code_data/psychrnn`. The dlPFC-like module uses the standard, unmodified upstream `Basic` model class (nonlinear/ReLU output). The MT-like module uses `Basic_linOut` (and its counterpart `BasicSimulator_linOut`), a linear-output variant — no output nonlinearity — added on top of upstream PsychRNN for this project, matching the linear vs. nonlinear output distinction described in supp Methods §6.1.

Plotting code uses the [cmcrameri](https://github.com/callumrollo/cmcrameri) package (Python bindings for Fabio Crameri's perceptually uniform "Scientific colour maps") for the circular/directional colormaps used throughout (e.g. `cmc.romaO`).

MATLAB analysis code (`matlab_code_data`) uses the [Circular Statistics Toolbox (CircStat)](https://www.mathworks.com/matlabcentral/fileexchange/10676-circular-statistics-toolbox-directional-statistics) for circular statistics (e.g. `circ_mean`, `circ_dist`), included here in `matlab_code_data/CircStat2012a`.

# Training the RNN model

Located in `python_code_data`:

### `train_MT.py`
Trains the MT-like module (`MT_broadInSharpOut_withR` task, defined in `DotsTasks.py`) on a noisy 72-direction motion input plus a 2D reward-condition cue, with 150 recurrent units under a Dale's law constraint (80% excitatory / 20% inhibitory), matching supp Methods §6.1. Uses TensorFlow's Adam optimizer via psychrnn's `Basic_linOut.train()`. The loss is a sum of two differentiable terms:
- `mse`: masked mean squared error between network output and target.
- `rewrad`: a circular (`atan2`-based) squared error between the network's estimated reward-center angle and the true one at the final timestep.

Trained weights are saved to `model_weights/` (as `MTbroadsharp_m10.npz` for the model used in the paper).

### `train_DLPFC.py`
Trains the dlPFC-like module (`DLPFC_combined` task, defined in `DotsTasks.py`), using a frozen, already-trained MT-like module (loaded from `model_weights/`) to generate its motion/reward-center inputs. Also 150 recurrent units, Dale's law 80/20, trained with psychrnn's `Basic.train()`. The loss is the default masked mean squared error (`mean_squared_error`, from `psychrnn/backend/loss_functions.py`) between the network's predicted per-choice reward function and the target reward function described in supp Methods §6.1 (Eq. for `R(θᵢ, θ_M, θ_R)`).

Trained weights are saved to `model_weights/` (as `DLPFCcombined_m4.npz` for the model used in the paper).

# Simulating trained models

`simulate_MT_only.py` and `simulate_combined_model.py` load trained weights and run trial batches through `BasicSimulator`/`BasicSimulator_linOut` (psychrnn's NumPy forward-pass implementation, distinct from the TensorFlow training graph), saving the results as pickled test data for downstream analysis. Both optionally implement the unit-perturbation method from supp Methods §6.2: an additional, untrained input channel is added to the model, with its `W_in` column zeroed and frozen during training (see `train_MT.py`/`train_DLPFC.py`) and then set post-hoc to project onto a single chosen unit, scaled by a configurable stimulation strength.

- `simulate_MT_only.py` — simulates the MT-like module by itself (not feeding into dlPFC), for testing/perturbing MT independent of the dlPFC module.
- `simulate_combined_model.py` — simulates the full two-module pipeline: a frozen, trained MT-like module feeding a frozen, trained dlPFC-like module. 
- `pickle2mat.py`
Converts a single `simulate_combined_model.py` test run's pickled output (trial parameters, MT/dlPFC recurrent activity, dlPFC input/output) into a single `.mat` file for use in MATLAB.

### `generate_microstim_testdata.py`
Generates microstimulation test data for multiple units/conditions in one run (looping over a set of stimulated units, each with and without stimulation), then aggregates the results into the `mstimMT`/`mstimDLPFC` `.mat` struct format consumed by the MATLAB microstimulation plotting scripts (e.g. `modelMicrostimHeatmap_binned.m`, `polarPlotsUstimChoiceSummary_model.m`). The reward bias relative to each stimulated unit's preferred direction (aligned vs. opposite) is set by `reward_rel`, reflected in the output filename (`bias180` = opposite; `bias0` = aligned) — matching **Figure 4B** (aligned) vs. **4C** (opposite).

### `find_unit_pref_dirs.py`
Finds each recurrent unit's preferred direction in both modules, relevant for determining the preferred directions of the units used in `generate_microstim_testdata.py`: output-weight tuning (`weight_pref`, both modules) and response tuning to the shown motion direction (`shown_pref`, both modules) and to the model's own choice (`choice_pref`, dlPFC only), computed from the existing all-conditions, zero-noise test data in `figure_code/testdata`. Saves `MT_tuning_df.csv`/`DLPFC_tuning_df.csv` and plots a sorted tuning matrix for each metric.

# Generating figures

*Note: this code runs the same analyses, but may not generate the same exact plots, as those shown in the paper.*

### Python figure scripts

Located in `python_code_data/figure_code/`

- `model_behavior_plots.py` — Plots the MT-like module's own behavior (chosen direction and decoded reward-bias estimate vs. the true values, from its own output) and the combined model's behavior (chosen direction vs. shown direction, split by reward-bias condition, compared to the most-rewarded direction), both from the existing all-conditions, zero-noise test data. Plots specific reward bias conditions of the summary in **Figure 3B**.

- `modelPCplots.py` — Generates PCA plots of the MT and dlPFC modules' activity in motion and reward dimensions, as well as in the top principal components. Uses test data generated by `simulate_combined_model.py`. This population-geometry analysis maps onto **Figure 3** of the paper (MT panels C–E; dlPFC panels F–H).

### MATLAB figure scripts

Located in `matlab_code_data`.

- `modelBehaviorSummary.m` — chosen vs. shown motion direction, relative to reward bias, for the combined model. Approximates **Figure 3B**.
- `modelMicrostimHeatmap.m` / `modelMicrostimHeatmap_binned.m` — choice-frequency difference heatmaps (chosen vs motion direction, relative to stimulated unit's preference), averaged across the 5 stimulated units per module. The `_binned` version bins into 20°-wide bins; the other computes exact per-degree frequencies. Both take a `layer` (`'MT'`/`'DLPFC'`) and `bias` (`0`/`180`) switch to pick which module/relative reward bias condition to load. The binned version maps onto **Figure 4B and C**, model columns.
- `polarPlotsUstimChoiceEx_model.m` — polar histogram of an example unit's choice distribution with microstimulation, for one module.
- `polarPlotsUstimChoiceSummary_model.m` — polar histogram pooling choices across all 5 stimulated units per module (aligned to each unit's own preferred direction). Maps onto **Figure 4A**.
- `polarPlotsUstimChoiceEx_expdata.m` / `polarPlotsUstimChoiceSummary_expdata.m` — the same two analyses, using real monkey microstimulation session data (`experimentdata/mtUstimBehaviorSummary.mat`/`dlpfcUstimBehaviorSummary.mat`). The summary maps onto **Figure 4A**, monkey columns. *Note: `experimentdata/mtUstimBehaviorSummary.mat` and `dlpfcUstimBehaviorSummary.mat` are not included in this repository; per the manuscript's Data Availability statement, this data is available from the corresponding author upon reasonable request.*
- `rewardFunctionPolarPlot.m` — polar plots of the reward function and the MT-like module's target output.
