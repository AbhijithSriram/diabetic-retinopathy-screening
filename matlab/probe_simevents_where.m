function probe_simevents_where()
%PROBE_SIMEVENTS_WHERE  Find the SimEvents block library on this install.
%   `ver` lists SimEvents, so the product is installed and its library file
%   is somewhere under matlabroot. This locates it and reports the name that
%   load_system / add_block will accept.

fprintf('=== ver ===\n');
try, disp(ver('simevents')); catch e, fprintf(2, '%s\n', e.message); end

fprintf('=== which -all simevents ===\n');
try, disp(which('simevents', '-all')); catch e, fprintf(2, '%s\n', e.message); end

fprintf('=== library files under matlabroot/toolbox/simevents ===\n');
d = fullfile(matlabroot, 'toolbox', 'simevents');
fprintf('%s exists: %d\n', d, isfolder(d));
if isfolder(d)
    for ext = {'*.slx', '*.mdl', '*.p'}
        f = dir(fullfile(d, '**', ext{1}));
        for i = 1:min(numel(f), 25)
            fprintf('  %s\n', fullfile(f(i).folder, f(i).name));
        end
    end
end

fprintf('=== anything named *simevents* elsewhere under toolbox ===\n');
f = dir(fullfile(matlabroot, 'toolbox', '**', '*simevents*.sl*'));
for i = 1:min(numel(f), 15)
    fprintf('  %s\n', fullfile(f(i).folder, f(i).name));
end

fprintf('=== try the simevents command (opens the library) ===\n');
try
    simevents;
    fprintf('simevents command ran\n');
catch e
    fprintf(2, 'simevents command failed: %s\n', e.message);
end

fprintf('=== block diagrams now in memory ===\n');
disp(find_system('type', 'block_diagram'));

fprintf('\nPaste all of the above back.\n');
end
