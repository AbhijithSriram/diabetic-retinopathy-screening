function probe_simevents()
%PROBE_SIMEVENTS  Print the parameter names this MATLAB release accepts for
%   every SimEvents block build_simulink_model.m uses.
%
%   build_simulink_model sets around twenty block parameters by name. Those
%   names drift between releases, and a wrong one is silently skipped by
%   trySet, so the model builds and then behaves oddly. Run this once on a
%   licensed machine and the real names are known instead of guessed:
%
%       probe_simevents
%
%   Nothing is saved; the scratch model is closed at the end.

blocks = { ...
    'Entity Generator', ...
    'Entity Server', ...
    'Entity Queue', ...
    'Entity Output Switch', ...
    'Entity Input Switch', ...
    'Entity Terminator'};

% the names build_simulink_model.m relies on, per block
wanted = containers.Map( ...
    {'Entity Generator', 'Entity Server', 'Entity Queue', 'Entity Output Switch', ...
     'Entity Input Switch', 'Entity Terminator'}, ...
    {{'TimeSource', 'IntergenerationTimeAction', 'AttributeName', 'AttributeInitialValue', 'GenerateAction'}, ...
     {'Capacity', 'ServiceTimeSource', 'ServiceTimeAction', 'ServiceTime', 'ExitAction'}, ...
     {'AverageWait', 'NumberEntitiesInBlock'}, ...
     {'NumberOutputPorts', 'SwitchingCriterion', 'SwitchAttributeName'}, ...
     {'NumberInputPorts', 'ExitAction'}, ...
     {}});

% Simulink must be up before its libraries can be resolved, and the
% SimEvents library has gone by more than one name across releases.
load_system('simulink');
lib = '';
for cand = {'sldelib', 'simevents', 'simeventslib'}
    try
        load_system(cand{1});
        lib = cand{1};
        break
    catch
    end
end
if isempty(lib)
    fprintf(2, 'Could not load the SimEvents library under any known name.\n');
    fprintf(2, 'Run these three lines and paste the output:\n');
    fprintf(2, '   ver(''simevents'')\n');
    fprintf(2, '   which -all simevents\n');
    fprintf(2, '   simevents   %% opens the library; then: disp(find_system(''type'',''block_diagram''))\n');
    return
end
fprintf('SimEvents library loaded as "%s"\n', lib);

mdl = 'probe_simevents_tmp';
if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl);
cleanup = onCleanup(@() close_system(mdl, 0));

for i = 1:numel(blocks)
    b = blocks{i};
    path = [mdl '/B' num2str(i)];
    try
        add_block([lib '/' b], path);
    catch e
        fprintf(2, '\n=== %s === could not be added from %s: %s\n', b, lib, e.message);
        continue
    end
    names = fieldnames(get_param(path, 'DialogParameters'));
    fprintf('\n=== %s === (%d parameters)\n', b, numel(names));
    need = wanted(b);
    for j = 1:numel(need)
        hit = find(strcmpi(names, need{j}), 1);
        if isempty(hit)
            near = names(contains(lower(names), lower(extractBefore(need{j}, min(5, strlength(need{j}) + 1)))));
            fprintf('  MISSING  %-26s  candidates: %s\n', need{j}, strjoin(near', ', '));
        else
            fprintf('  ok       %-26s\n', names{hit});
        end
    end
    fprintf('  all: %s\n', strjoin(names', ', '));
end
fprintf('\nDone. Paste this output back so the names can be fixed in one pass.\n');
end
