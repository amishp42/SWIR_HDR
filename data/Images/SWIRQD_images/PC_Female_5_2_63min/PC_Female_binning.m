

% To analyze, plot, and export to CSV:
histograms = tif_histogram_binner('PC_Female_5_2_63min_3_14.tif', 'histogram_data.csv');

% Plot all histograms with consistent bins
for i = 1:length(histograms)
    figure(i);
    bar(histograms(i).edges(1:end-1), histograms(i).counts);
    title(['Histogram for Image ' num2str(i)]);
    xlabel('Pixel Value');
    ylabel('Frequency');
    xlim([1000 5000]); % Fixed x-axis limits
    
    % Add statistics to the plot
    text(0.7, 0.9, sprintf('Min: %.2f\nMax: %.2f\nMean: %.2f', ...
        histograms(i).min_val, ...
        histograms(i).max_val, ...
        histograms(i).mean_val), ...
        'Units', 'normalized');
end

% Optional: Plot all histograms on the same figure
figure;
hold on;
colors = jet(length(histograms));
for i = 1:length(histograms)
    plot(histograms(i).edges(1:end-1), histograms(i).counts, 'Color', colors(i,:));
end
hold off;
title('Overlaid Histograms for All Images');
xlabel('Pixel Value');
ylabel('Frequency');
xlim([1000 5000]);
legend(arrayfun(@(x) ['Image ' num2str(x)], 1:length(histograms), 'UniformOutput', false));