cd '/Users/Sol/Desktop/CohenLab/DotsBehavior/matlab_code_data/finalModel_testData_plotCode'
load('DLPFCcombined_m4_allCondsNoNoise_coh0.6.mat')
load('romaO.mat')

[~, maxind] = max(DLPFCoutput(:,end,:), [], 3);
choice_degs = 5*(maxind-1);
N = size(trialparams, 2);
shown_degs = zeros(N,1);
bias_degs = zeros(N,1);
for i=1:N
    shown_degs(i,1) = trialparams{1,i}.shown_deg;
    bias_degs(i,1) = trialparams{1,i}.good_deg;
end
%%
fr = max(DLPFCstatevar, 0);
[coeff,~,~,~,var] = pca(reshape(fr(:,end,:),[size(fr,1), size(fr,3)]));
%%
rewConds = 0:20:359;
PCs = [4, 5, 6];
figure(1)
    hold on
    axis([-25 30  -25 30  -25 30])
    xlabel(sprintf('PC%d', PCs(1)))
    ylabel(sprintf('PC%d', PCs(2)))
    zlabel(sprintf('PC%d', PCs(3)))
for r=1:size(rewConds,2)
    view(3)
    trials = find(bias_degs==rewConds(r));
    traj = reshape(fr(trials,end,:),[size(trials,1), size(fr,3)]) * coeff(:,PCs);
    plot3(traj(:,1), traj(:,2), traj(:,3), LineWidth=1.5, Color=romaO(cast(rewConds(r)/360*256, 'uint8')+1,:));
    drawnow
    title('reward bias',rewConds(r))
    pause(0.2)
end
%% PC 1 and 2 as theta
rewConds = 0:20:359;
figure(1)
    hold on
    xlabel('theta (deg)')
    ylabel('PC3')
    zlabel('PC4')
for r=1:size(rewConds,2)
    view(3)
    trials = find(bias_degs==rewConds(r));
    traj = reshape(fr(trials,end,:),[size(trials,1), size(fr,3)]) * coeff(:,[1,2,3,4]);
    theta = atan2d(traj(:,2),traj(:,1));
    [~, order] = sort(theta);
    plot3(theta(order), traj(order,3), traj(order,4), LineWidth=1.5, Color=romaO(cast(rewConds(r)/360*256, 'uint8')+1,:));
    drawnow
    title('reward bias',rewConds(r))
    pause(0.2)
end

