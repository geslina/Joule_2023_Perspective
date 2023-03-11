function [feature] = featurization_helper2(diff_time_data,series_data, min_lim,max_lim)
% Alexis Geslin - UPDATED SEPT 1 2022: Extract a feature between pct_min and pct_max
% (Global pct_min, max)
%   Inputs: array of diff_time values (time spent at the voltage, absI or absP)(roughly constant, but some points are weighted stronger)
%           array of the time series of interest: voltage, current, power
%           pct_min and pct_max are the upper and lower percentile limits
%           to derive the feature.
%   Output: feature representing the percentage of time spent in between the two percentiles.

diff_time_data(:,2) = series_data>=min_lim;
diff_time_data(:,3) = series_data<=max_lim;
diff_time_data(:,4) = diff_time_data(:,1).*diff_time_data(:,2).*diff_time_data(:,3); % Time spent of within V2,3 bound (per row)
feature = sum(diff_time_data(:,4)) / sum(diff_time_data(:,1)); %formula of the Howey features

end