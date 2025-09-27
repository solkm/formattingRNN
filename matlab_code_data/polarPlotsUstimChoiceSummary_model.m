%% add paths
cd('/Users/Sol/Desktop/CohenLab/DotsBehavior/matlab_code_data') % 'SET_CURRENT_DIRECTORY'
addpath('/Users/Sol/Desktop/CohenLab/GeneralCode/CircStat2012a') % 'PATH_TO_CircStat'
addpath('./finalModel_testData_plotCode') % 'PATH_TO_DATA'

%% load data
load("DLPFCcombined_m4_stimDLPFCu[53, 25, 54, 8, 9]_paramsChoicesOutputs.mat")
load("DLPFCcombined_m4_stimMTu[37, 4, 42, 25, 34]_paramsChoicesOutputs.mat")
%%
area = 1; % 0 for MT 1 for DLPFC
if area==0
    areaColor = '#D95319';
    data = mstimMT;
else
    areaColor = 'b';
    data = mstimDLPFC;
end

choices_mstim = [];
choices_nomstim = [];
shown_dist = 100; % distance from microstim site
sthresh = 20; % max deviation from value above, 0 for exact match
num_units = size(unique(data.unit),2);

for u = 1:num_units
    s = 2*u;
    ns = s-1;
    prefdir = double(data.pref_dir(s));
    sdeg = prefdir + 100;
    if sdeg > 360
        sdeg = sdeg - 360;
    end
    inds_stim = abs(rad2deg(circ_dist(deg2rad(double(data.motion_dirs(s,:))), deg2rad(sdeg)))) < sthresh+1;
    c_s = double(data.choices(s, inds_stim));
    inds_nostim = abs(rad2deg(circ_dist(deg2rad(double(data.motion_dirs(ns,:))), deg2rad(sdeg)))) < sthresh+1;
    c_ns = double(data.choices(ns, inds_nostim));
    
    choices_mstim = [choices_mstim; rad2deg(circ_dist(deg2rad(c_s), deg2rad(prefdir)))'];
    choices_nomstim = [choices_nomstim; rad2deg(circ_dist(deg2rad(c_ns), deg2rad(prefdir)))'];
end

mstim_dir = 0;
bias_deg = 0;
mean_choice_nomstim = rad2deg(circ_mean(deg2rad(choices_nomstim)));
%% polar plot

plot_nomstim_dist = false; % whether or not to plot the distribution of choices without microstim
pax=polaraxes;
binwidth=15;
ch_ms=polarhistogram(pax,deg2rad(double(choices_mstim)), 'Normalization','probability','BinEdges',deg2rad(0:binwidth:360),'FaceColor',areaColor,'FaceAlpha',0.6,'DisplayName','proportion of choices, with \mustim');
hold on
if plot_nomstim_dist
    ch_n=polarhistogram(pax,deg2rad(double(choices_nomstim)), 'Normalization','probability','BinEdges',deg2rad(0:binwidth:360),'FaceColor',[0, 0.8, 0],'FaceAlpha',0.4,'DisplayName','proportion of choices, w/out \mustim');
end
rho=0.4; % edit radius of plot
polarplot(pax,repmat(deg2rad(double(mstim_dir)),1,2), [0, rho], 'LineStyle', '--', 'Color','r', 'Marker','none','LineWidth',1.5);
sx=polarscatter(pax,deg2rad(double(mstim_dir)), rho, 120, 'Marker','X','MarkerEdgeColor', 'r','LineWidth',2,'DisplayName','stimulated direction');

polarplot(pax,repmat(deg2rad(mean_choice_nomstim),1,2), [0, rho], 'LineStyle', '--', 'Color',[0, 0.8, 0], 'Marker','none','LineWidth',1.5);
bx=polarscatter(pax,deg2rad(mean_choice_nomstim), rho, 120, 'Marker','X','MarkerEdgeColor', [0, 0.8, 0],'LineWidth',2,'DisplayName','mean choice w/out \mustim');

rho=rho+0;
bias_thetas=deg2rad(double(bias_deg))-0.15:0.01:deg2rad(double(bias_deg))+0.15;
ga=polarplot(pax,bias_thetas, repmat(rho,size(bias_thetas)), 'LineStyle', '-', 'Color', [1, 0.4, 0.8],'LineWidth',3,'DisplayName','reward bias');
polarplot(pax,bias_thetas-pi, repmat(rho,size(bias_thetas)), 'LineStyle', '-', 'Color', [0.6, 0, 1],'LineWidth',3)

if area==0
    polarplot(pax,repmat(circ_mean(deg2rad(choices_mstim)),1,2), [0, rho], 'LineStyle', '--', 'Color', 'k', 'Marker', 'none', 'LineWidth', 1.5);
    ux=polarscatter(pax,circ_mean(deg2rad(choices_mstim)), rho, 120, 'Marker','X','MarkerEdgeColor', 'k', 'LineWidth', 2, 'DisplayName', 'mean choice with \mustim');
    if plot_nomstim_dist
        legend([ch_ms,ch_n,sx,ux,bx,ga])
    else
        legend([ch_ms,sx,ux,bx,ga])
    end
else
    if plot_nomstim_dist
        legend([ch_ms,ch_n,sx,bx,ga])
    else
        legend([ch_ms,sx,bx,ga])
    end
end

pax.FontSize=14;
rlim([0,rho]);
rticks([0,0.1,0.2,0.3,0.4]);



