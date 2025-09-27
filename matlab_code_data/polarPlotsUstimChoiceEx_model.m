%% add paths
cd('/Users/Sol/Desktop/CohenLab/DotsBehavior/matlab_code_data') % 'SET_CURRENT_DIRECTORY'
addpath('/Users/Sol/Desktop/CohenLab/GeneralCode/CircStat2012a') % 'PATH_TO_CircStat'
addpath('./finalModel_testData_plotCode') % 'PATH_TO_DATA'

%% load data
load("DLPFCcombined_m4_stimDLPFCu[53, 25, 54, 8, 9]_paramsChoicesOutputs.mat")
load("DLPFCcombined_m4_stimMTu[37, 4, 42, 25, 34]_paramsChoicesOutputs.mat")
%% pick an example unit and motion direction

area = 1; % 0 for MT 1 for DLPFC
if area==0
    areaColor = '#D95319';
    data = mstimMT;
    unit = 37;
    motion_dir = 0;
else
    areaColor = 'b';
    data = mstimDLPFC;
    unit = 54;
    motion_dir = 155;
end

idx = find(data.unit == unit);
ns = find(data.motion_dirs(idx(1),:)==motion_dir);
choices_nomstim = data.choices(idx(1),ns);
s = find(data.motion_dirs(idx(2),:)==motion_dir);
choices_mstim = data.choices(idx(2),s);

mstim_dir = data.pref_dir(idx(1));
bias_deg = mstim_dir;
mean_choice_nomstim = rad2deg(circ_mean(deg2rad(double(choices_nomstim'))));
mean_choice_mstim = rad2deg(circ_mean(deg2rad(double(choices_mstim'))));
%% polar plot

plot_nomstim_dist = false; % whether or not to plot the distribution of choices without microstim
pax=polaraxes;
binwidth=15;
ch_ms=polarhistogram(pax,deg2rad(double(choices_mstim)), 'Normalization','probability','BinEdges',deg2rad(0:binwidth:360),'FaceColor',areaColor,'FaceAlpha',0.6,'DisplayName','proportion of choices, with \mustim');
hold on
if plot_nomstim_dist
    ch_n=polarhistogram(pax,deg2rad(double(choices_nomstim)), 'Normalization','probability','BinEdges',deg2rad(0:binwidth:360),'FaceColor',[0, 0.8, 0],'FaceAlpha',0.4,'DisplayName','proportion of choices, w/out \mustim');
end
rho=1; % edit radius of plot
polarplot(pax,repmat(deg2rad(double(mstim_dir)),1,2), [0, rho], 'LineStyle', '--', 'Color','r', 'Marker','none','LineWidth',1.5);
sx=polarscatter(pax,deg2rad(double(mstim_dir)), rho, 120, 'Marker','X','MarkerEdgeColor', 'r','LineWidth',2,'DisplayName','stimulated direction');

polarplot(pax,repmat(deg2rad(mean_choice_nomstim),1,2), [0, rho], 'LineStyle', '--', 'Color',[0, 0.8, 0], 'Marker','none','LineWidth',1.5);
bx=polarscatter(pax,deg2rad(mean_choice_nomstim), rho, 120, 'Marker','X','MarkerEdgeColor', [0, 0.8, 0],'LineWidth',2,'DisplayName','mean choice w/out \mustim');

rho=rho+0;
bias_thetas=deg2rad(double(bias_deg))-0.15:0.01:deg2rad(double(bias_deg))+0.15;
ga=polarplot(pax,bias_thetas, repmat(rho,size(bias_thetas)), 'LineStyle', '-', 'Color', [1, 0.4, 0.8],'LineWidth',3,'DisplayName','reward bias');
polarplot(pax,bias_thetas-pi, repmat(rho,size(bias_thetas)), 'LineStyle', '-', 'Color', [0.6, 0, 1],'LineWidth',3)

if plot_nomstim_dist
    legend([ch_ms,ch_n,sx,bx,ga])
else
    legend([ch_ms,sx,bx,ga])
end

pax.FontSize=14;
rlim([0,rho]);
rticks([0,0.1,0.2,0.3,0.4, 0.5, 0.6]);
