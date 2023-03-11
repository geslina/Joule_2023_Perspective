function [ch_indices,full_indices,disch_indices] = get_charge_discharge_indices2(cell_data_cycle)
%This function extracts indices corresponding to the charge and to the
%discharge part of the cycling curve, including both CC and CV parts. It
%also returns the full_indices, to correct the glitches in the dataset.
%   Input: cell_data at a given cycle. It contains I, V, t series among
%       others
%   Output: the indices corresponding to the charge portion of the curves
%       and the discharge portion of the curves. Discharge only include points
%       where I =-4A and removing the trailing point
%       The charge part includes both CC1, CC2, CV but remove any points with
%       negative current. 
all = find(cell_data_cycle.t>-5);
CC_dis = find(abs(cell_data_cycle.I+4) < 0.05);
last_dis = CC_dis(end);
afterCC = linspace(last_dis+1,all(end),all(end)-last_dis)';
CV_dis = afterCC(abs(cell_data_cycle.V(afterCC)-2.0)<0.01);
CV_dis(cell_data_cycle.t(CV_dis)-cell_data_cycle.t(last_dis)>10)=[];
disch_indices =  [CC_dis;CV_dis];
full_indices = linspace(1,disch_indices(end),disch_indices(end))';
ch_indices = linspace(1,disch_indices(1),disch_indices(1))';
remove_neg_I = find(cell_data_cycle.I(ch_indices)< -0.05);
ch_indices(remove_neg_I) =[];
end