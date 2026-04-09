% --- Data Preparation ---
ratios = [10, 20, 30, 40, 50]; 

% Mean Accuracy (OA) - Carefully re-mapped from the image
means = [
    78.0, 80.5, 82.8, 83.5, 84.5; % RF (Cyan solid) - Bottom group
    79.0, 80.8, 81.5, 82.5, 83.2; % MLP (Green solid) - Bottom group
    76.8, 81.2, 83.5, 84.8, 85.5; % LightGBM (Black dashed)
    75.5, 81.8, 84.2, 85.2, 86.8; % CatBoost (Magenta dashed)
    74.0, 78.5, 82.8, 84.5, 85.0; % XGBoost (Green dashed)
    82.5, 86.8, 89.0, 90.0, 91.8; % TabNet (Black solid) - Middle group
    %84.2, 88.2, 90.8, 91.5, 92.8; % uTabNet (Red dashed)
    85.2, 88.5, 91.8, 92.8, 93.8; % CNN2D (Cyan dashed)
    95.2, 97.8, 98.2, 98.6, 99.0; % TabNets (Red solid) - Top group
    96.5, 98.2, 98.8, 99.5, 99.8  % ATabNet (Magenta solid) - Top group
];

% Standard Deviation (Error Bar Lengths)
stds = [
    0.5, 0.4, 0.6, 0.4, 0.5; % RF
    0.8, 0.6, 0.5, 0.5, 0.4; % MLP
    0.6, 0.7, 0.5, 0.6, 0.5; % LightGBM
    0.7, 0.5, 0.6, 0.4, 0.6; % CatBoost
    0.9, 0.8, 0.7, 0.6, 0.5; % XGBoost
    0.8, 0.7, 0.8, 0.7, 0.6; % TabNet
    %0.6, 0.6, 0.7, 0.5, 0.4; % uTabNet
    0.5, 0.5, 0.6, 0.5, 0.4; % CNN3d
    0.4, 0.3, 0.4, 0.3, 0.3; % TabNets
    0.3, 0.2, 0.3, 0.2, 0.2  % uTabNets
];

% Configuration for names and styles
names = {'RF', 'MLP', 'LightGBM', 'CatBoost', 'XGBoost', 'TabNet', 'CNN2D', 'TabNets', 'uTabNets'};
colors = {'c', 'g', 'k', 'm', 'g', 'k', 'c', 'r', 'm'};
lines  = {'-', '-', '--', '--', '--', '-', '--', '-', '-'};
markers = {'s', 'o', 'x', '+', '*', 'd', 'v', '>', '<'};  %, '^'

% --- Plotting ---
figure; hold on;

for i = 1:size(means, 1)
    errorbar(ratios, means(i,:), stds(i,:), ...
        'Color', colors{i}, 'LineStyle', lines{i}, ...
        'Marker', markers{i}, 'LineWidth', 1.5, 'DisplayName', names{i});
end

% Backward-Compatible Formatting
xlabel('Ratio of training samples per class');
ylabel('Overall Accuracy (OA)');
%title('Model Accuracy Comparison (Corrected Mapping)');

% Fix for xticks error in older MATLAB versions
set(gca, 'XTick', ratios);
set(gca, 'XTickLabel', {'10%', '20%', '30%', '40%', '50%'});
set(gca, 'XLim', [8 52]);
%set(gca, 'XLim', [10 52]);
set(gca, 'YLim', [70 100]);

grid on;
%legend('Location', 'southeast', 'FontSize', 8);
legend('show', 'Location', 'southeast');
hold off;
