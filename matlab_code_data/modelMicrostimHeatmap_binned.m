%% load data

layer = 'MT'; % 'MT' or 'DLPFC'
bias = 0; % 180 or 0, reward bias relative to microstimulated site

if bias==180
    bias_str = 'bias180';
elseif bias==0
    bias_str = '';
end

if strcmp(layer, 'MT')
    filepath = append('./modeltestdata/DLPFCcombined_m4_stimMTu[37, 4, 42, 25, 34]', ...
                       bias_str, '_paramsChoicesOutputs.mat');
    load(filepath)
    data = mstimMT;
elseif strcmp(layer, 'DLPFC')
    filepath = append('./modeltestdata/DLPFCcombined_m4_stimDLPFCu[53, 25, 54, 8, 9]', ...
                   bias_str, '_paramsChoicesOutputs.mat');
    data = mstimDLPFC; 
end

%% calculate matrix

bin_size = 20;
motion_bins = -190:bin_size:190;
choice_bins = -190:bin_size:190;
num_units = size(unique(data.unit), 2);
choiceFreqDiffs = zeros(num_units, size(motion_bins, 2) - 1, size(choice_bins, 2) - 1);

for u = 1:num_units

    % center the motion and choice directions around the stimulated unit's 
    % preferred direction
    motion_rel2stim = data.motion_dirs(2*u-1:2*u, :) - data.pref_dir(2*u);
    motion_rel2stim(motion_rel2stim > 180) = motion_rel2stim(motion_rel2stim > 180) - 360;
    motion_rel2stim(motion_rel2stim <= -180) = motion_rel2stim(motion_rel2stim <= -180) + 360;

    choices_rel2stim = data.choices(2*u-1:2*u, :) - data.pref_dir(2*u);
    choices_rel2stim(choices_rel2stim > 180) = choices_rel2stim(choices_rel2stim > 180) - 360;
    choices_rel2stim(choices_rel2stim <= -180) = choices_rel2stim(choices_rel2stim <= -180) + 360;

    for m = 1:size(choiceFreqDiffs, 2)
        
        % get choice probabilities for trials WITHOUT microstimulation
        inds0 = motion_rel2stim(1, :) >= motion_bins(m) & ...
                motion_rel2stim(1, :) < motion_bins(m+1);

        [freqs0, ~] = histcounts(choices_rel2stim(1, inds0), choice_bins, ...
                                 'Normalization', 'probability');

        % get choice probabilities for trials WITH microstimulation
        inds1 = motion_rel2stim(2, :) >= motion_bins(m) & ...
                motion_rel2stim(2, :) < motion_bins(m+1);

        [freqs1, ~] = histcounts(choices_rel2stim(2, inds1), choice_bins, ...
                                 'Normalization', 'probability');

        choiceFreqDiffs(u, m, :) = freqs1 - freqs0;
    end
end

avg_choiceFreqDiffs = squeeze(mean(choiceFreqDiffs,1));

%% plot heatmap

figure
colormap parula
imagesc(avg_choiceFreqDiffs.')
clim([-1,1])
colorbar
labels = -150:50:150;
tick_locs = labels/bin_size + bin_size/2;
set(gca, 'XTick', tick_locs, 'XTickLabel', labels)
set(gca, 'YTick', tick_locs, 'YTickLabel', labels)
title(strcat(layer, ' model, difference of choice frequency (ustim - no ustim)'))
xlabel("distance of visual stimulus from stimulated site's preference")
ylabel('chosen angle')
hold on
plot(1:72, 1:72, 'Color', 'k')
pbaspect([1,1,1])
