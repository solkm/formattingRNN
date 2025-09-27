% reward function polar plot
addpath('/Users/Sol/Desktop/CohenLab/GeneralCode/CircStat2012a')
%%
bias_deg = 90;
shown_deg = 0;
accCutoff_deg = 45;
t = linspace(0, 2*pi, 73);

distFromShown = abs(circ_dist(deg2rad(shown_deg), t));
distFromBias = abs(circ_dist(deg2rad(bias_deg), t));
accuracyFunction = zeros(1, length(t));
biasFunction = zeros(1, length(t));
for ii = 1:length(t)
    if distFromShown(ii) <= deg2rad(accCutoff_deg)
        accuracyFunction(1, ii) = 3.5 + 1.5*cos(1.8*distFromShown(ii));
    end
    biasFunction(1, ii) = 1 + 0.75*cos(distFromBias(ii));
end

rewardFunction = accuracyFunction .* biasFunction;
[maxR, argmaxR] = max(rewardFunction);
best_rad = t(argmaxR);
%% dlpfc model - reward function
pax=polaraxes;
baseline = 0;
bias_thetas=deg2rad(bias_deg)-0.14:0.01:deg2rad(bias_deg)+0.14;
polarplot(pax, bias_thetas, (baseline+1)*ones(size(bias_thetas)), 'LineStyle', '-', 'Color', [1, 0.4, 0.8],'LineWidth',6,'DisplayName','reward bias');
hold on
polarscatter(pax, deg2rad(shown_deg),baseline+1,320,'Marker','X','MarkerFaceColor', [0, 176/255, 240/255],'MarkerEdgeColor', [0, 176/255, 240/255],'LineWidth',7)
polarplot(pax, t, baseline + rewardFunction./maxR, 'Color', [179/255, 140/255, 1], 'Linewidth',2)
polarscatter(pax, t, baseline + rewardFunction./maxR, 150, 'MarkerFaceColor', [179/255, 140/255, 1], 'MarkerEdgeColor','k')

rticks([])
pax.FontSize = 16;
%% mt model - motion function (new, MT_broadsharp_m10)
A = 2.0;
k_out = 0.8;
coh = 0.8;
motionFunction = A * (-0.3 + exp(coh*k_out*cos(distFromShown)));

rho = floor(max(motionFunction)+1.0);
pax=polaraxes;
baseline = 0;
bias_thetas=deg2rad(bias_deg)-0.14:0.01:deg2rad(bias_deg)+0.14;
polarplot(pax, bias_thetas, (baseline+rho)*ones(size(bias_thetas)), 'LineStyle', '-', 'Color', [1, 0.4, 0.8],'LineWidth',6,'DisplayName','reward bias');
hold on
polarscatter(pax, deg2rad(shown_deg),baseline+rho,320,'Marker','X','MarkerFaceColor', [0, 176/255, 240/255],'MarkerEdgeColor', [0, 176/255, 240/255],'LineWidth',7)
polarplot(pax, t, baseline + motionFunction, 'Color', [0, 176/255, 240/255], 'LineWidth', 2)
polarscatter(pax, t, baseline + motionFunction, 150, 'MarkerFaceColor', [0, 176/255, 240/255], 'MarkerEdgeColor','k')
polarscatter(pax, deg2rad(bias_deg),baseline+rho, 150, 'MarkerFaceColor', [1, 0.4, 0.8],'MarkerEdgeColor', 'k')
rticks([])
pax.FontSize = 16;

%% mt model - motion function (old)
motionFunction = exp(-0.5*(8/pi * distFromShown).^2);

pax=polaraxes;
baseline = 0;
bias_thetas=deg2rad(bias_deg)-0.14:0.01:deg2rad(bias_deg)+0.14;
polarplot(pax, bias_thetas, (baseline+1)*ones(size(bias_thetas)), 'LineStyle', '-', 'Color', [1, 0.4, 0.8],'LineWidth',6,'DisplayName','reward bias');
hold on
polarscatter(pax, deg2rad(shown_deg),baseline+1,320,'Marker','X','MarkerFaceColor', [0, 176/255, 240/255],'MarkerEdgeColor', [0, 176/255, 240/255],'LineWidth',7)
polarplot(pax, t, baseline + motionFunction, 'Color', [0, 176/255, 240/255], 'LineWidth', 2)
polarscatter(pax, t, baseline + motionFunction, 150, 'MarkerFaceColor', [0, 176/255, 240/255], 'MarkerEdgeColor','k')
polarscatter(pax, deg2rad(bias_deg),baseline+1, 150, 'MarkerFaceColor', [1, 0.4, 0.8],'MarkerEdgeColor', 'k')
rticks([])
pax.FontSize = 16;

