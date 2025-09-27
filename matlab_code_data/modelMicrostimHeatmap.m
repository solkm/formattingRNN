%% load data
load('DLPFCcombined_m4_stimMTu[37, 4, 42, 25, 34]_paramsChoicesOutputs.mat')
load('DLPFCcombined_m4_stimDLPFCu[53, 25, 54, 8, 9]_paramsChoicesOutputs.mat')
%% calculate matrix
layer = 'DLPFC'; % 'MT' or 'DLPFC'

if strcmp(layer, 'MT')
    data = mstimMT;
elseif strcmp(layer, 'DLPFC')
    data = mstimDLPFC; 
end

unique_motion = unique(data.motion_dirs);
unique_choice = unique(data.choices);
num_units = size(unique(data.unit),2);
choiceFreqDiffs = zeros(num_units, size(unique_motion,1), size(unique_choice,1));

for u = 1:num_units
    freqs0 = zeros(size(unique_motion,1), size(unique_choice,1));
    freqs1 = zeros(size(unique_motion,1), size(unique_choice,1));
    for m = 1:size(unique_motion,1)
        inds0 = find(data.motion_dirs(2*u-1,:) == unique_motion(m));
        choices_m0 = data.choices(2*u-1, inds0);
        chosen_dirs_m0 = unique(choices_m0);
        for c = 1:size(chosen_dirs_m0,2)
            freqs0(m, chosen_dirs_m0(c)/5+1) = nnz(choices_m0==chosen_dirs_m0(c))/size(inds0,2);
        end

        inds1 = find(data.motion_dirs(2*u,:) == unique_motion(m));
        choices_m1 = data.choices(2*u, inds1);
        chosen_dirs_m1 = unique(choices_m1);
        for c = 1:size(chosen_dirs_m1,2)
            freqs1(m, chosen_dirs_m1(c)/5+1) = nnz(choices_m1==chosen_dirs_m1(c))/size(inds1,2);
        end 
    end
    diff_uncentered = freqs1 - freqs0;
    
    motion_rel2stim = unique_motion - data.pref_dir(2*u);
    motion_rel2stim(motion_rel2stim>180) = motion_rel2stim(motion_rel2stim>180) - 360;
    motion_rel2stim(motion_rel2stim<=-180) = motion_rel2stim(motion_rel2stim<=-180) + 360;
    [~, motion_order] = sort(motion_rel2stim);

    choice_rel2stim = unique_choice - data.pref_dir(2*u);
    choice_rel2stim(choice_rel2stim>180) = choice_rel2stim(choice_rel2stim>180) - 360;
    choice_rel2stim(choice_rel2stim<=-180) = choice_rel2stim(choice_rel2stim<=-180) + 360;
    [~, choice_order] = sort(choice_rel2stim);

    choiceFreqDiffs(u,:,:) = diff_uncentered(motion_order, choice_order);
end
avg_choiceFreqDiffs = squeeze(mean(choiceFreqDiffs,1));

%% plot heatmap

colormap parula
imagesc(avg_choiceFreqDiffs.')
clim([-1,1])
colorbar
labels = -150:50:150;
set(gca, 'XTick',labels/5+36, 'XTickLabel', labels)
set(gca, 'YTick',labels/5+36, 'YTickLabel', labels)
title(strcat(layer, ' model, difference of choice frequency (ustim - no ustim)'))
xlabel("distance of visual stimulus from stimulated site's preference")
ylabel('chosen angle')
hold on
plot(1:72, 1:72, 'Color', 'k')
pbaspect([1,1,1])
