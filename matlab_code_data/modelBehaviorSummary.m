% model behavior summary
addpath('/Users/Sol/Desktop/CohenLab/GeneralCode')
addpath('/Users/Sol/Desktop/CohenLab/GeneralCode/CircStat2012a')
addpath('/Users/Sol/Desktop/CohenLab/DotsBehavior/matlab_code_data/finalModel_testData_plotCode')

load("DLPFCcombined_m4_allCondsNoNoise_coh0.6.mat")
%%
N_trials = size(trialparams, 2);
choices_rel2bias = zeros(1, N_trials);
motiondirs_rel2bias = zeros(1, N_trials);

for ii = 1:N_trials
    bias = double(trialparams{ii}.good_deg);
    shown_deg = double(trialparams{ii}.shown_deg);
    [~, maxind] = max(DLPFCoutput(ii, end, :));
    choice_deg = double((maxind-1)*5);
    
    motiondirs_rel2bias(1, ii) = rad2deg(circ_dist(deg2rad(shown_deg), deg2rad(bias)));
    choices_rel2bias(1, ii) = rad2deg(circ_dist(deg2rad(choice_deg), deg2rad(bias)));
end
%%
plot(-180:180, -180:180, 'k', LineWidth=1.5)
hold on
scatter(motiondirs_rel2bias, choices_rel2bias, 20, 'b', 'filled')

xlabel('Motion direction relative to reward bias')
ylabel('Chosen direction relative to reward bias')
xticks(-150:50:150)
yticks(-150:50:150)
marleneaxes