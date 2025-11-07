function image_histograms = tif_histogram_binner(filename, output_csv)
    % Define fixed bin parameters
    start_val = 1000;
    end_val = 5000;
    bin_width = 5;
    edges = start_val:bin_width:end_val;
    
    % Get TIFF info
    tiff_info = imfinfo(filename);
    num_images = length(tiff_info);
    
    % Preallocate structure array
    image_histograms = struct('counts', cell(1, num_images), ...
                            'edges', cell(1, num_images), ...
                            'min_val', cell(1, num_images), ...
                            'max_val', cell(1, num_images), ...
                            'mean_val', cell(1, num_images));
    
    % Create matrix to store all histogram data for CSV
    bin_centers = edges(1:end-1) + bin_width/2;  % Calculate bin centers
    histogram_data = zeros(length(bin_centers), num_images + 1);
    histogram_data(:,1) = bin_centers';  % First column is bin centers
    
    % Process each image
    for i = 1:num_images
        % Read image
        current_image = imread(filename, i);
        current_image = double(current_image);
        
        % Compute statistics
        image_histograms(i).min_val = min(current_image(:));
        image_histograms(i).max_val = max(current_image(:));
        image_histograms(i).mean_val = mean(current_image(:));
        
        % Compute histogram with specified bin edges
        [counts, ~] = histcounts(current_image(:), edges);
        image_histograms(i).counts = counts;
        image_histograms(i).edges = edges;
        
        % Store counts in histogram_data matrix
        histogram_data(:,i+1) = counts';
    end
    
    % Export to CSV if output_csv parameter is provided
    if nargin > 1
        % Create header row
        header = cell(1, num_images + 1);
        header{1} = 'Bin_Center';
        for i = 1:num_images
            header{i+1} = ['Image_' num2str(i)];
        end
        
        % Convert header to table
        T = array2table(histogram_data, 'VariableNames', header);
        
        % Write to CSV
        writetable(T, output_csv);
        fprintf('Histogram data exported to: %s\n', output_csv);
    end
end



