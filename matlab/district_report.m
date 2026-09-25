function T = district_report(out)
%DISTRICT_REPORT  Turn a drishti_district run into the numbers module 5 asks for.
%   out = sim('drishti_district');
%   T   = district_report(out);
%
%   Prints the final value of every logged statistic: how many patients
%   reached each endpoint, how long each queue grew, and how heavily the
%   reviewers were loaded. Compare the totals with district_sim.m, which
%   computes the same district in base MATLAB.

if ~isprop(out, 'logsout') || isempty(out.logsout)
    fprintf(2, ['No logged signals. Rebuild with build_simulink_model so the\n' ...
                'statistic ports are enabled and marked for logging.\n']);
    T = table();
    return
end

ls = out.logsout;
n = ls.numElements;
name = strings(n, 1);
final = zeros(n, 1);
kind = strings(n, 1);
for i = 1:n
    e = ls{i};
    name(i) = string(e.Name);
    if name(i) == "", name(i) = "(unnamed " + i + ")"; end
    [final(i), kind(i)] = lastValue(e.Values);
end
T = sortrows(table(name, final, kind, ...
    'VariableNames', {'statistic', 'final', 'container'}), 'statistic');
disp(T)

days = str2double(get_param('drishti_district', 'StopTime')) / 24;
fprintf('\nover %g simulated days\n', days);
fprintf(['\nSanity check against the base-MATLAB model:\n' ...
         '   S = district_sim([], 2, 0.6, 0.40)\n' ...
         'Treated should be in the low thousands. Dozens means entities are\n' ...
         'stalling, most likely at the Referrals input switch.\n']);
end

function [d, kind] = lastValue(v)
%LASTVALUE  Final sample of a logged signal, whatever container it arrived in.
%   Simulink has used timeseries, "structure with time" and, in recent
%   releases, timetable. Rather than assume one, take the last sample from
%   whichever this is and report the type so a surprise is visible.
d = NaN;
kind = string(class(v));
try
    if isa(v, 'timeseries')
        x = v.Data;
    elseif istimetable(v)
        x = v{:, 1};
    elseif isstruct(v) && isfield(v, 'Data')
        x = v.Data;
    elseif isnumeric(v)
        x = v;
    else
        x = [];
    end
    if ~isempty(x)
        x = x(:);
        d = double(x(end));
    end
catch
end
end
