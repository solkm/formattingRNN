%% add paths
cd('/Users/Sol/Desktop/CohenLab/DotsBehavior/matlab_code_data') % 'SET_CURRENT_DIRECTORY'
addpath('/Users/Sol/Desktop/CohenLab/GeneralCode/CircStat2012a') % 'PATH_TO_CircStat'
addpath('./finalModel_testData_plotCode') % 'PATH_TO_DATA'

%% monkey data, pull single examples from beh structure

area = 0; % 0 for MT 1 for DLPFC

if area==0
    load('mtUstimBehaviorSummary.mat')
    areaColor = '#D95319';
    sessData = mtUstim.beh(24).alltrials;
    mstim_dir = 130;
    mstim_ch = 9;
    bias_deg = 150;
    shown_deg = 40;
    inds = sessData(:,4)==mstim_ch & sessData(:,1)==shown_deg & sessData(:,8)==bias_deg;
    choices_mstim = sessData(inds,2);
    disp(['# trials: ', num2str(nnz(inds))])
    inds2 = sessData(:,4)==0 & sessData(:,1)==shown_deg & sessData(:,8)==bias_deg;
    choices_nomstim = sessData(inds2,2);
    mean_choice_nomstim = rad2deg(circ_mean(deg2rad(choices_nomstim)));

else
    load('dlpfcUstimBehaviorSummary.mat')
    areaColor = 'b';
    sessData = dlpfcUstim.beh(13).alltrials;
    mstim_dir = 0;
    mstim_ch = 159;
    bias_deg = 0;
    shown_deg = 140;
    inds = sessData(:,4)==mstim_ch & sessData(:,1)==shown_deg & sessData(:,8)==bias_deg;
    choices_mstim = sessData(inds,2);
    disp(['# trials: ', num2str(nnz(inds))])
    inds2 = sessData(:,4)==0 & sessData(:,1)==shown_deg & sessData(:,8)==bias_deg;
    choices_nomstim = sessData(inds2,2);
    mean_choice_nomstim = rad2deg(circ_mean(deg2rad(choices_nomstim)));

end
%% polar plot

plot_nomstim_dist = false; % whether or not to plot the distribution of choices without microstim
pax=polaraxes;
binwidth=10;
ch_ms=polarhistogram(pax,deg2rad(double(choices_mstim)), 'Normalization','probability','BinEdges',deg2rad(0:binwidth:360),'FaceColor',areaColor,'FaceAlpha',0.6,'DisplayName','proportion of choices, with \mustim');
hold on
if plot_nomstim_dist
    ch_n=polarhistogram(pax,deg2rad(double(choices_nomstim)), 'Normalization','probability','BinEdges',deg2rad(0:binwidth:360),'FaceColor',[0, 0.8, 0],'FaceAlpha',0.4,'DisplayName','proportion of choices, w/out \mustim');
end
rho=0.35; % edit radius of plot
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
