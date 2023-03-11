%% Alexis Geslin March 2023 %%
% batch 1,2 and 3 - Data and code (first section) to load the data are from Severson et 
% al. publication (Severson, K.A., Attia, P.M., Jin, N. et al. Data-driven 
% prediction of battery cycle life before capacity degradation. Nat Energy 4,
%  383–391 (2019))
%  This script generates features following the method published by 
%  Greenbank and Howey (IEEE Transactions on Industrial Informatics, 18,
%  2965-2973 (2021)),used for our work in the publication.
% This script also extracts cycling conditions.

%%
clear; close all; clc
addpath('Matlab_Utils')

load('Data/2017-05-12_batchdata_updated_struct_errorcorrect')

batch1 = batch; 
numBat1 = size(batch1,2);

load('Data/2017-06-30_batchdata_updated_struct_errorcorrect')

%Some batteries continued from the first run into the second. We append 
%those to the first batch before continuing.
add_len = [661, 980, 1059, 207, 481];
summary_var_list = {'cycle','QDischarge','QCharge','IR','Tmax','Tavg',...
    'Tmin','chargetime'};
batch2_idx = [8:10,16:17];
for i=1:5
    batch1(i).cycles(end+1:end+add_len(i)+1) = batch(batch2_idx(i)).cycles;
    batch1(i).summary.cycle(end+1:end+add_len(i)+1) = ...
        batch(batch2_idx(i)).summary.cycle;
    batch1(i).summary.QDischarge(end+1:end+add_len(i)+1) = ...
        batch(batch2_idx(i)).summary.QDischarge;
    batch1(i).summary.QCharge(end+1:end+add_len(i)+1) = ...
        batch(batch2_idx(i)).summary.QCharge;
    batch1(i).summary.IR(end+1:end+add_len(i)+1) = ...
        batch(batch2_idx(i)).summary.IR;
    batch1(i).summary.Tmax(end+1:end+add_len(i)+1) = ...
        batch(batch2_idx(i)).summary.Tmax;
    batch1(i).summary.Tavg(end+1:end+add_len(i)+1) = ...
        batch(batch2_idx(i)).summary.Tavg;
    batch1(i).summary.Tmin(end+1:end+add_len(i)+1) = ...
        batch(batch2_idx(i)).summary.Tmin;
    batch1(i).summary.chargetime(end+1:end+add_len(i)+1) = ...
        batch(batch2_idx(i)).summary.chargetime;
end

batch([8:10,16:17]) = [];
batch2 = batch;
numBat2 = size(batch2,2);
clearvars batch

load('Data/2018-04-12_batchdata_updated_struct_errorcorrect')
batch3 = batch;
batch3(38) = []; %remove channel 46 upfront; there was a problem with 
%the data collection for this channel
numBat3 = size(batch3,2);
endcap3 = zeros(numBat3,1);
clearvars batch
for i = 1:numBat3
    endcap3(i) = batch3(i).summary.QDischarge(end);
end
rind = find(endcap3 > 0.885);
batch3(rind) = [];

%remove the noisy Batch 8 batteries
nind = [3, 40:41];
batch3(nind) = [];
numBat3 = size(batch3,2);

batch_combined = [batch1, batch2, batch3];
numBat = numBat1 + numBat2 + numBat3;

%optionally remove the batteries that do not finish in Batch 1; depending
%on the modeling goal, you may not want to do this step
batch_combined([9,11,13,14,23]) = [];
numBat = numBat - 5;
numBat1 = numBat1 - 5; 

clearvars -except batch_combined numBat1 numBat2 numBat3 numBat

% Output variable
%Extract the number of cycles to 0.88; this is the output variable used in
%modeling for the paper

bat_label = zeros(numBat,1);
for i = 1:numBat
    if batch_combined(i).summary.QDischarge(end) < 0.88
        bat_label(i) = find(batch_combined(i).summary.QDischarge < 0.88,1);
        
    else
        bat_label(i) = size(batch_combined(i).cycles,2) + 1;
    end
end

bat_label = log10(bat_label);


%% Setting up our data and parameters

% begin_cycles = [1 31 61 91];
begin_cycles = [0 1 11 21 31 41 51 61 71 81 91 101 111 121 131 141];
num_cycle =10; %represent to window width the features is calculated upon

% Creating streams and names for intervals. Loading the limits structure
streams = ["current","abs_current","voltage","power","abs_power"];
percentile_handles = ["1_33","1_67","1_99","33_67","33_99","67_99",];

% Structuring containing percentile limits (chosen manually) defining
% intervals where datastreams may fall in
clear LS 

LS.charge.current = get_stream_limits([0 1 2 5]);
LS.charge.abs_current = get_stream_limits([0 1 2 5]);
LS.charge.voltage = get_stream_limits([3.1 3.3 3.5 3.6]);
LS.charge.power = get_stream_limits([0 3.3 7 18]);
LS.charge.abs_power = get_stream_limits([0 3.3 7 18]);

LS.full.current = get_stream_limits([-3.9 -1 1 5]);
LS.full.abs_current = get_stream_limits([0 2 3 5]);
LS.full.voltage = get_stream_limits([2 3 3.3 3.6]);
LS.full.power = get_stream_limits([-7.8 -3 3.3 18]);
LS.full.abs_power = get_stream_limits([0 6 9.9 18]);

LS.discharge.current = get_stream_limits([-4.0005 -4.0001 -3.9999 -3.9995]);
LS.discharge.abs_current = get_stream_limits([3.9995 3.9999 4.0001 4.0005]);
LS.discharge.voltage = get_stream_limits([2.5 3 3.3 3.5]);
LS.discharge.power = get_stream_limits([-14 -13.2 -12 -10]);
LS.discharge.abs_power = get_stream_limits([10 12 13.2 14]);

%just picking one limit 6x2 array and extracting 6.
num_intervals = size(LS.charge.abs_current,1); 
num_features = size(begin_cycles,2).*length(streams).*num_intervals;
my_feats = zeros(numBat,num_features,3);
features_list =[];



%% Creating feature matrix
for  i = 1:numBat
    cell_data = batch_combined(i);
    % printing progress
    if rem(i,10)==0
        disp(i)
    end

    % Every 10cycles, we create new sets of features, based on the next 10
    % cycles. eg, begin_cycles = 31. Then we extract features based on
    % cycles 31-40.
    for j = 1:length(begin_cycles)
        begin_cycle = begin_cycles(j);

        % We then iterate through every stream of data
        for k = 1:length(streams)
            stream = streams(k);

            % For each stream, we calculate 6 features, 1 for each of
            % the 6 intervals in LS
            
            % We thus iterate through all 6 percentile intervals.
            for interval_idx = 1:num_intervals
                percentile_name =percentile_handles(interval_idx);
                
                %for each stream, there will be 6 percentile intervals, 
                %for each begin_cycle, there will be X streams x 6
                %intervals = gives us the formula for feature number: 
                feat_index = interval_idx + (k-1)*num_intervals+(j-1)*length(streams)*num_intervals;

                %adding cycle 1 data, based on only 1 cycle of data. 
                %Added at the beginning of the features matrix
                if begin_cycle==0
                    [my_feats(i,feat_index,1),my_feats(i,feat_index,2),my_feats(i,feat_index,3)]= featurization_helper1(cell_data,begin_cycle+1,stream,LS,interval_idx,1);
                    % this begin_cycle+1 passed in the function is because
                    % begin_cycle = 0, so we want cycle 1.
                else
                    [my_feats(i,feat_index,1),my_feats(i,feat_index,2),my_feats(i,feat_index,3)]= featurization_helper1(cell_data,begin_cycle,stream,LS,interval_idx,num_cycle);
                end

                %creating features handles
                if i==1 
                    feat_to_add = "Cycle_"+num2str(begin_cycle,'%03.f')+'-'+num2str(begin_cycle+9,'%03.f')+'_'+stream+"_"+percentile_name;
                    if begin_cycle == 0
                        feat_to_add = "Cycle_"+num2str(begin_cycle+1,'%03.f')+'-'+num2str(begin_cycle+1,'%03.f')+'_'+stream+"_"+percentile_name;
                    end
                    features_list = [features_list,feat_to_add];
                end
            end                     
        end
    end
end


%% Matrix for cycling conditions
feat = zeros(numBat,4);

for i = 1:numBat 
    policy = batch_combined(i).policy_readable;
    [first_charging,percent,second_charging]=get_policy(policy);
    
    feat(i,1) = first_charging;
    feat(i,2) = second_charging;
    feat(i,3) = percent;
    % adding the average charging rate.
    feat(i,4) = (first_charging * percent + second_charging *(80-percent))/80;


end %end loop through batteries

% description 
feat_description = ["CC1" "CC2" "SOC shift in CC" "avg C charge"];


%% Saving data for Python ML
clear features_matrix
features_matrix.features = my_feats;
features_matrix.cycling_conditions =feat;
features_matrix.labels = bat_label;
features_matrix.features_names = cellstr(features_list);
features_matrix.cycling_conditions_names =cellstr(feat_description);
save('data/features_matrix.mat','-struct','features_matrix');

