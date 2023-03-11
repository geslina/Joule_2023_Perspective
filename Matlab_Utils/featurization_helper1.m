function [red_feature,red_green_feature,green_feature] = featurization_helper1(cell_data,begin_cycle,series_of_interest,lim_struct,interval_idx,num_cycle)
%%% Alexis Geslin - last update: November 2022 update %%% 
%This function returns three features, one based on charge only (red feature), one on the whole
%time-series data (red_green feature) and one based exclusively on the
%discharge part of the curve (green feature). It computes the
%features over the duration of 10 cycles, between min and max pctle
%   Inputs: cell_data is the matlab struct containing among other "cycles"
%           info, which contains t, V, I
%           begin_cycle indicates the cycle number where to start counting
%           for 10 cycles: expected 1,31,61,91 for example
%           Series of interest is either 'abs_current', 'voltage', 'abs_power'
%           min_pct and max_pct are the percentile limits of interest for
%           this feature
%           Num_cycle: define the width of the window to derive the
%           features on: aka how many cycles
%   Ouputs: Three features, one based on charge, on based on charge + discharge curves
%           (red_green) features and the other one based exclusively on the
%           discharge part of the curve (green feature)
%   Requirements: featurization_helper2


diff_time_data =[];
diff_time_data_dis = [];
diff_time_data_ch = [];
series_data = [];
series_data_dis = [];
series_data_ch = [];

if strcmp(series_of_interest,'current')
    for i=0:num_cycle-1
        [ch_ind,full_ind,disch_ind] = get_charge_discharge_indices2(cell_data.cycles(begin_cycle+1+i));
        time_data = cell_data.cycles(begin_cycle+1+i).t;
        diff_time = [0;diff(time_data)];
        diff_time(diff_time>1)=0.0001; 

        diff_time_data = [diff_time_data;diff_time(full_ind)];
        diff_time_data_dis =[diff_time_data_dis;diff_time(disch_ind)];
        diff_time_data_ch =[diff_time_data_ch;diff_time(ch_ind)];

        series_data = [series_data;cell_data.cycles(begin_cycle+1+i).I(full_ind)];
        series_data_dis =[series_data_dis;cell_data.cycles(begin_cycle+1+i).I(disch_ind)];
        series_data_ch = [series_data_ch;cell_data.cycles(begin_cycle+1+i).I(ch_ind)];
    end

elseif strcmp(series_of_interest,'abs_current')
    for i=0:num_cycle-1
        [ch_ind,full_ind,disch_ind] = get_charge_discharge_indices2(cell_data.cycles(begin_cycle+1+i));
        time_data = cell_data.cycles(begin_cycle+1+i).t;
        diff_time = [0;diff(time_data)];
        diff_time(diff_time>1)=0.0001; 

        diff_time_data = [diff_time_data;diff_time(full_ind)];
        diff_time_data_dis =[diff_time_data_dis;diff_time(disch_ind)];
        diff_time_data_ch =[diff_time_data_ch;diff_time(ch_ind)];
        
        series_data = [series_data;abs(cell_data.cycles(begin_cycle+1+i).I(full_ind))];
        series_data_dis =[series_data_dis;abs(cell_data.cycles(begin_cycle+1+i).I(disch_ind))];
        series_data_ch = [series_data_ch;abs(cell_data.cycles(begin_cycle+1+i).I(ch_ind))];
    end

elseif strcmp(series_of_interest, 'voltage')
    for i=0:num_cycle-1
        [ch_ind,full_ind,disch_ind] = get_charge_discharge_indices2(cell_data.cycles(begin_cycle+1+i));
        time_data = cell_data.cycles(begin_cycle+1+i).t;
        diff_time = [0;diff(time_data)];
        diff_time(diff_time>1)=0.0001; 

        diff_time_data = [diff_time_data;diff_time(full_ind)];
        diff_time_data_dis =[diff_time_data_dis;diff_time(disch_ind)];
        diff_time_data_ch =[diff_time_data_ch;diff_time(ch_ind)];
        
        series_data = [series_data;cell_data.cycles(begin_cycle+1+i).V(full_ind)];
        series_data_dis =[series_data_dis;cell_data.cycles(begin_cycle+1+i).V(disch_ind)];
        series_data_ch = [series_data_ch;cell_data.cycles(begin_cycle+1+i).V(ch_ind)];
    end

elseif strcmp(series_of_interest,'power')
    for i=0:num_cycle-1
        [ch_ind,full_ind,disch_ind] = get_charge_discharge_indices2(cell_data.cycles(begin_cycle+1+i));
        time_data = cell_data.cycles(begin_cycle+1+i).t;
        diff_time = [0;diff(time_data)];
        diff_time(diff_time>1)=0.0001; 

        diff_time_data = [diff_time_data;diff_time(full_ind)];
        diff_time_data_dis =[diff_time_data_dis;diff_time(disch_ind)];
        diff_time_data_ch =[diff_time_data_ch;diff_time(ch_ind)];

        power = cell_data.cycles(begin_cycle+i+1).I .* cell_data.cycles(begin_cycle+i+1).V;
        series_data = [series_data;power(full_ind)];
        series_data_dis = [series_data_dis;power(disch_ind)];
        series_data_ch = [series_data_ch;power(ch_ind)];
    end

elseif strcmp(series_of_interest,'abs_power')
    for i=0:num_cycle-1
        [ch_ind,full_ind,disch_ind] = get_charge_discharge_indices2(cell_data.cycles(begin_cycle+1+i));
        time_data = cell_data.cycles(begin_cycle+1+i).t;
        diff_time = [0;diff(time_data)];
        diff_time(diff_time>1)=0.0001; 

        diff_time_data = [diff_time_data;diff_time(full_ind)];
        diff_time_data_dis =[diff_time_data_dis;diff_time(disch_ind)];
        diff_time_data_ch =[diff_time_data_ch;diff_time(ch_ind)];

        power = abs(cell_data.cycles(begin_cycle+i+1).I .* cell_data.cycles(begin_cycle+i+1).V);
        series_data = [series_data;power(full_ind)];
        series_data_dis = [series_data_dis;power(disch_ind)];
        series_data_ch = [series_data_ch;power(ch_ind)];
    end
    
else
    error('Wrong input: series of interest must be current, voltage, power, abs_current, or abs_power')
end

red_feature = featurization_helper2(diff_time_data_ch,series_data_ch,lim_struct.charge.(series_of_interest)(interval_idx,1),lim_struct.charge.(series_of_interest)(interval_idx,2));
red_green_feature = featurization_helper2(diff_time_data,series_data,lim_struct.full.(series_of_interest)(interval_idx,1),lim_struct.full.(series_of_interest)(interval_idx,2));
green_feature = featurization_helper2(diff_time_data_dis,series_data_dis,lim_struct.discharge.(series_of_interest)(interval_idx,1),lim_struct.discharge.(series_of_interest)(interval_idx,2));
end