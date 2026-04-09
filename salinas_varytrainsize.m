% --- Data Preparation ---
% X-axis: Number of training samples per class
x_values = [100, 200, 300, 400, 500]; 

% Mean Accuracy (OA) - Estimated from the new image
means = [
    72.2, 79.8, 86.5, 86.8, 88.2; % RF (Cyan solid)
    81.2, 84.2, 87.5, 87.8, 88.5; % MLP (Green solid)
    85.2, 87.8, 88.5, 88.8, 89.8; % LightGBM (Black dashed)
    83.0, 84.4, 85.2, 85.4, 85.8; % CatBoost (Magenta dashed)
    84.5, 86.8, 87.2, 88.2, 88.8; % XGBoost (Green dashed)
    88.5, 90.2, 90.5, 92.0, 92.2; % TabNet (Black solid)
    %89.5, 91.2, 91.8, 92.4, 92.6; % uTabNet (Red dashed)
    90.2, 92.4, 93.2, 94.0, 94.8; % CNN2d (Cyan dashed)
    96.0, 97.2, 98.0, 98.6, 99.0; % TabNets (Red solid)
%     96.8, 98.2, 98.8, 99.2, 99.4  % uTabNets (Magenta solid)
    96.8, 98.97, 99.24, 99.37, 99.45  % uTabNets (Magenta solid)
];

% Standard Deviation (Error Bar Length) - Estimated
stds = [
    1.2, 0.8, 0.7, 0.6, 0.5; % RF
    0.9, 0.7, 0.6, 0.5, 0.4; % MLP
    0.8, 0.7, 0.8, 0.6, 0.7; % LightGBM
    0.6, 0.5, 0.6, 0.5, 0.5; % CatBoost
    0.8, 0.6, 0.5, 0.4, 0.5; % XGBoost
    0.9, 0.8, 0.5, 0.4, 0.4; % TabNet
    %0.7, 0.6, 0.6, 0.5, 0.4; % uTabNet
    0.5, 0.4, 0.5, 0.3, 0.4; % CNN2D
    0.6, 0.5, 0.6, 0.5, 0.4; % TabNets
    0.4, 0.35, 0.32, 0.3, 0.3  % uTabNets
];

% Configuration for styles
names = {'RF', 'MLP', 'LightGBM', 'CatBoost', 'XGBoost', 'TabNet', 'CNN2D', 'TabNets', 'ATabNet'};
colors = {'c', 'g', 'k', 'm', 'g', 'k', 'c', 'r', 'm'}; %%'r'
lines  = {'-', '-', '--', '--', '--', '-', '--', '-', '-'};
markers = {'s', 'o', 'x', '+', '*', 'd', 'v', '>', '<'}; %%'^'

% --- Plotting ---
figure; hold on;

for i = 1:size(means, 1)
    errorbar(x_values, means(i,:), stds(i,:), ...
        'Color', colors{i}, 'LineStyle', lines{i}, ...
        'Marker', markers{i}, 'LineWidth', 1.5, 'DisplayName', names{i});
end

% Formatting (Backward Compatible with older MATLAB versions)
xlabel('Number of training samples per class');
ylabel('Overall Accuracy (OA)');
%title('Accuracy Comparison across different training sizes');

% Set axes limits and ticks
set(gca, 'XTick', x_values);
set(gca, 'XLim', [90 510]);
set(gca, 'YLim', [70 100]);

grid on;
%legend('Location', 'southeast', 'FontSize', 8);
legend('show', 'Location', 'southeast');
hold off;
