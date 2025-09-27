%% add paths
cd('SET_CURRENT_DIRECTORY')
addpath('PATH_TO_CircStat/CircStat2012a')
addpath('PATH_TO_DATA')

%% load data, MT or DLPFC

area = 1; % 0 for MT 1 for DLPFC

if area==0
    load('mtUstimBehaviorSummary.mat')
    areaUstim = mtUstim;
    areaColor = '#D95319';
else
    load('dlpfcUstimBehaviorSummary.mat')
    areaUstim = dlpfcUstim;
    areaColor = 'b';
end
%% pool trials from beh structure, where microstim aligns with GOOD reward location

choices_mstim = [];
choices_nomstim = [];
shown_deg = 100; % distance from microstim site
sthresh = 20; % max deviation from value above, 0 for exact match
gthresh = 20; % max distance of microstim from reward bias, 0 for exact match
numSess = 0;
for ii = 1:length(allSess)
    sess_beh = areaUstim.beh(ii);
    stimIsGood = zeros(2,2);
    stimIsGood(1,:) = abs(rad2deg(circ_dist(deg2rad(sess_beh.biasDirs),deg2rad(sess_beh.uStimCh1pref)))) < gthresh+1; 
    stimIsGood(2,:) = abs(rad2deg(circ_dist(deg2rad(sess_beh.biasDirs),deg2rad(sess_beh.uStimCh2pref)))) < gthresh+1;
    if nnz(stimIsGood)==0
        continue
    end
    numSess = numSess + 1;
    for ch = 1:2
        for bd = 1:2
            if stimIsGood(ch,bd)==1
                inds = abs(rad2deg(circ_dist(deg2rad(sess_beh.UstimAlignedChoices{ch,bd}(:,1)),deg2rad(shown_deg)))) < sthresh+1;
                choices_mstim = [choices_mstim; sess_beh.UstimAlignedChoices{ch,bd}(inds,2)];
                inds = abs(rad2deg(circ_dist(deg2rad(sess_beh.nonUstimAlignedChoices{ch,bd}(:,1)),deg2rad(shown_deg)))) < sthresh+1;
                choices_nomstim = [choices_nomstim; sess_beh.nonUstimAlignedChoices{ch,bd}(inds,2)];
            end
        end
    end
end
mstim_dir = 0;
bias_deg = 0;
mean_choice_nomstim = rad2deg(circ_mean(deg2rad(choices_nomstim)));
choice_deg = choices_mstim;

%% polar plot
plot_nomstim_dist = false; % whether or not to plot the distribution of choices without microstim
pax=polaraxes;
binwidth=15;
ch_ms=polarhistogram(pax,deg2rad(double(choice_deg)), 'Normalization','probability','BinEdges',deg2rad(0:binwidth:360),'FaceColor',areaColor,'FaceAlpha',0.6,'DisplayName','proportion of choices, with \mustim');
hold on
if plot_nomstim_dist
    ch_n=polarhistogram(pax,deg2rad(double(choices_nomstim)), 'Normalization','probability','BinEdges',deg2rad(0:binwidth:360),'FaceColor',[0, 0.8, 0],'FaceAlpha',0.4,'DisplayName','proportion of choices, w/out \mustim');
end
rho=0.25; % edit radius of plot
polarplot(pax,repmat(deg2rad(double(mstim_dir)),1,2), [0, rho], 'LineStyle', '--', 'Color','r', 'Marker','none','LineWidth',1.5);
sx=polarscatter(pax,deg2rad(double(mstim_dir)), rho, 120, 'Marker','X','MarkerEdgeColor', 'r','LineWidth',2,'DisplayName','stimulated direction');

polarplot(pax,repmat(deg2rad(mean_choice_nomstim),1,2), [0, rho], 'LineStyle', '--', 'Color',[0, 0.8, 0], 'Marker','none','LineWidth',1.5);
bx=polarscatter(pax,deg2rad(mean_choice_nomstim), rho, 120, 'Marker','X','MarkerEdgeColor', [0, 0.8, 0],'LineWidth',2,'DisplayName','mean choice w/out \mustim');

rho=rho+0;
bias_thetas=deg2rad(double(bias_deg))-0.15:0.01:deg2rad(double(bias_deg))+0.15;
ga=polarplot(pax,bias_thetas, repmat(rho,size(bias_thetas)), 'LineStyle', '-', 'Color', [1, 0.4, 0.8],'LineWidth',3,'DisplayName','reward bias');
polarplot(pax,bias_thetas-pi, repmat(rho,size(bias_thetas)), 'LineStyle', '-', 'Color', [0.6, 0, 1],'LineWidth',3)

if area==0
    polarplot(pax,repmat(circ_mean(deg2rad(choice_deg)),1,2), [0, rho], 'LineStyle', '--', 'Color', 'k', 'Marker', 'none', 'LineWidth', 1.5);
    ux=polarscatter(pax,circ_mean(deg2rad(choice_deg)), rho, 120, 'Marker','X','MarkerEdgeColor', 'k', 'LineWidth', 2, 'DisplayName', 'mean choice with \mustim');
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
