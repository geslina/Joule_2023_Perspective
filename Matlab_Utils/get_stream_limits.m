function [stream_limits] = get_stream_limits(stream_percentiles)
% From an array of percentile stream values of size n (like the voltage corresponding
% to 33 percentile of the data), return a n! x 2 array of stream limits
% values

V_down = [];
V_up = [];
for i = 1 :(length(stream_percentiles)-1)
    V_down = [V_down, stream_percentiles(i)*ones(1,length(stream_percentiles)-i)];
    V_up = [V_up,stream_percentiles(i+1:end)];

end
stream_limits = [V_down; V_up]';

end