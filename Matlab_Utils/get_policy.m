function [first_charging,percent,second_charging] = get_policy(policy)
% Given a cell data policy, the function returns the first charging rate,
% the SOC % switch to the second charging and the second charging rate
first_charging = str2num(policy(1:strfind(policy,'(')-2));
second_charging = str2num(policy(strfind(policy,'-')+1:end-1));
%treating policy reporting error:
if policy(end) == 'e'
    second_charging = str2num(policy(strfind(policy,'-')+1:strfind(policy,'n')-3));
elseif policy(end)~='C'
    second_charging = str2num(policy(strfind(policy,'-')+1:end));
end
percent = str2num(policy(strfind(policy,'(')+1:strfind(policy,'%')-1));
end

