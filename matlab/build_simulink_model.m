function mdl = build_simulink_model(paramsFile)
%BUILD_SIMULINK_MODEL  Programmatically build the SimEvents model of a
%   district DR screening programme, parameterised like sim/params.yaml.
%
%   mdl = build_simulink_model('../sim/params.json')   % JSON copy of params.yaml
%
%   Entities are patients (time unit: hours). Attributes:
%     Route    1 recall, 2 tele-review, 3 direct refer, 4 ungradable after 3 tries
%     Refer    1 if the reviewer refers (review route only)
%     Attends  1 if the referred patient reaches the hospital
%   Blocks: Patients -> Capture (sites) -> Triage switch
%     -> Recall | Upload queue -> Uplink -> Review queue -> Reviewers -> decision
%     -> Referrals merge -> Attends? -> Hospital queue -> Hospital slots.
%   Reviewer availability (h/day) and link availability are folded into
%   effective service times; sim/district.py models them explicitly.
%
%   Verified on R2026b, 25 September 2026: builds, runs, and agrees with
%   matlab/district_sim.m (the same model in base MATLAB) to within 1-3%.
%   Parameter names are set through trySet(), which prints any name your
%   release rejects; fix those with get_param(<block>, 'DialogParameters'),
%   and see matlab/README.md for the seven API differences already handled.

if nargin < 1
    paramsFile = fullfile(fileparts(mfilename('fullpath')), '..', 'sim', 'params.json');
end
P = jsondecode(fileread(paramsFile));

% The SimEvents block library is 'sldelib' from R2025b or so; it was
% 'simevents' before that, and `simevents` is now only a function that
% opens the library. Resolve it rather than hard-coding either name.
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
    error('drishti:noSimEvents', ...
          'Could not load the SimEvents block library. Run probe_simevents_where.');
end

mdl = 'drishti_district';
if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl); open_system(mdl);
set_param(mdl, 'StopTime', num2str(P.run.days * 24), 'Solver', 'VariableStepDiscrete');

rate = P.district.annual_screenings / P.district.working_days_per_year / 24;  % per hour, averaged over the day
tri = P.triage;
pUngr = P.capture.intrinsic_ungradable;
routeP = [(1 - pUngr) * tri.recall, (1 - pUngr) * tri.review, (1 - pUngr) * tri.refer, pUngr];
uploadH = P.network.image_mb * P.network.eyes_per_patient * 8 * 1024 / P.network.bandwidth_kbps / 3600 / P.network.online_fraction;
reviewH = P.review.seconds_per_case / 3600 * 24 / P.review.review_hours_per_reviewer_day;
hospitalH = 24 / P.referral.hospital_slots_per_day;

add('simevents/Entity Generator', 'Patients', [40 100]);
trySet('Patients', 'TimeSource', 'MATLAB action');
trySet('Patients', 'IntergenerationTimeAction', sprintf('dt = -log(rand()) / %g;', rate));
% RefIdx and AttIdx are written by the Reviewers and Referrals exit actions
% and read by the switches after them. A SimEvents entity's attribute set is
% fixed when it is created, so both have to be declared here -- an action
% cannot add a field. They are 1-based switch indices, so they start at 1.
trySet('Patients', 'AttributeName', 'Route|Refer|Attends|RefIdx|AttIdx', ...
                   'AttributeInitialValue', '1|0|0|1|1');
% RefIdx and AttIdx are the 1-based port indices the two downstream switches
% read. They used to be computed in exit actions, but Entity Input Switch has
% no actions at all in this release, and both are fixed functions of
% attributes already decided here, so they are set at birth instead.
% SimEvents actions are compiled, not interpreted, so every expression needs
% a provably fixed size. find(u <= c, 1) can in principle return empty, so it
% types as [1 x :?] and cannot be assigned to a scalar attribute. Counting how
% many cumulative bins u has passed is the same thing and is always scalar;
% the min() guards the case where rounding leaves cumsum just under 1.
trySet('Patients', 'GenerateAction', sprintf([ ...
    'u = rand(); c = cumsum([%s]); entity.Route = min(sum(u > c) + 1, 4);\n' ...
    'entity.Refer = double(rand() < %g);\n' ...
    'entity.Attends = double(rand() < %g);\n' ...
    'entity.RefIdx = entity.Refer + 1;\n' ...
    'entity.AttIdx = entity.Attends + 1;'], ...
    num2str(routeP, '%g '), P.review.refer_after_review, P.referral.completion_with_engine));

add('simevents/Entity Server', 'Capture', [180 100]);
trySet('Capture', 'Capacity', num2str(P.district.sites));
trySet('Capture', 'ServiceTimeSource', 'MATLAB action');
m = P.capture.minutes_per_attempt;
% triangular_draw is inlined rather than called: a compiled action cannot
% rely on an ordinary function file being resolvable on the path.
trySet('Capture', 'ServiceTimeAction', sprintf([ ...
    'u = rand(); a = %g; md = %g; b = %g;\n' ...
    'if u < (md - a) / (b - a)\n' ...
    '    dt = (a + sqrt(u * (b - a) * (md - a))) / 60;\n' ...
    'else\n' ...
    '    dt = (b - sqrt((1 - u) * (b - a) * (b - md))) / 60;\n' ...
    'end'], m(1), m(2), m(3)));

add('simevents/Entity Output Switch', 'Triage', [320 100]);
trySet('Triage', 'NumberOutputPorts', '4');
trySet('Triage', 'SwitchingCriterion', 'From attribute');
trySet('Triage', 'SwitchAttributeName', 'Route');

add('simevents/Entity Terminator', 'Recall', [480 20]);
add('simevents/Entity Queue', 'UploadQueue', [480 100]);
add('simevents/Entity Server', 'Uplink', [600 100]);
trySet('Uplink', 'Capacity', num2str(P.district.sites));
% constant service time lives in ServiceTimeValue; ServiceTime does not exist
trySet('Uplink', 'ServiceTimeSource', 'Dialog');
trySet('Uplink', 'ServiceTimeValue', num2str(uploadH));
add('simevents/Entity Queue', 'ReviewQueue', [720 100]);
add('simevents/Entity Server', 'Reviewers', [840 100]);
trySet('Reviewers', 'Capacity', num2str(P.review.reviewers));
trySet('Reviewers', 'ServiceTimeSource', 'Dialog');
trySet('Reviewers', 'ServiceTimeValue', num2str(reviewH));
add('simevents/Entity Output Switch', 'ReviewDecision', [960 100]);
trySet('ReviewDecision', 'NumberOutputPorts', '2');
trySet('ReviewDecision', 'SwitchingCriterion', 'From attribute');
trySet('ReviewDecision', 'SwitchAttributeName', 'RefIdx');
add('simevents/Entity Terminator', 'RecallAfterReview', [1100 60]);

add('simevents/Entity Input Switch', 'Referrals', [1100 200]);
trySet('Referrals', 'NumberInputPorts', '3');
add('simevents/Entity Output Switch', 'AttendsSwitch', [1220 200]);
trySet('AttendsSwitch', 'NumberOutputPorts', '2');
trySet('AttendsSwitch', 'SwitchingCriterion', 'From attribute');
trySet('AttendsSwitch', 'SwitchAttributeName', 'AttIdx');
add('simevents/Entity Terminator', 'LostToFollowUp', [1360 160]);
add('simevents/Entity Queue', 'HospitalQueue', [1360 240]);
add('simevents/Entity Server', 'HospitalSlots', [1480 240]);
trySet('HospitalSlots', 'ServiceTimeSource', 'Dialog');
trySet('HospitalSlots', 'ServiceTimeValue', num2str(hospitalH));
add('simevents/Entity Terminator', 'Treated', [1600 240]);

conn('Patients/1', 'Capture/1');
conn('Capture/1', 'Triage/1');
conn('Triage/1', 'Recall/1');
conn('Triage/2', 'UploadQueue/1');
conn('UploadQueue/1', 'Uplink/1');
conn('Uplink/1', 'ReviewQueue/1');
conn('ReviewQueue/1', 'Reviewers/1');
conn('Reviewers/1', 'ReviewDecision/1');
conn('ReviewDecision/1', 'RecallAfterReview/1');
conn('ReviewDecision/2', 'Referrals/1');
conn('Triage/3', 'Referrals/2');
conn('Triage/4', 'Referrals/3');
conn('Referrals/1', 'AttendsSwitch/1');
conn('AttendsSwitch/1', 'LostToFollowUp/1');
conn('AttendsSwitch/2', 'HospitalQueue/1');
conn('HospitalQueue/1', 'HospitalSlots/1');
conn('HospitalSlots/1', 'Treated/1');

% Statistics. Each one enabled here adds a signal output port to its block;
% without logging them the run animates and reports nothing, which is no use
% for a capacity study. These are the numbers module 5 actually asks for:
% how many patients reach each endpoint, how long the queues get, and how
% hard the reviewers are working.
for q = {'UploadQueue', 'ReviewQueue', 'HospitalQueue'}
    trySet(q{1}, 'AverageWait', 'on');
    trySet(q{1}, 'NumberEntitiesInBlock', 'on');
    trySet(q{1}, 'AverageQueueLength', 'on');
end
for t = {'Recall', 'RecallAfterReview', 'LostToFollowUp', 'Treated'}
    trySet(t{1}, 'NumberEntitiesArrived', 'on');
end
trySet('Reviewers', 'Utilization', 'on');
trySet('Reviewers', 'AverageWait', 'on');
trySet('Capture', 'Utilization', 'on');

% Mark every statistic port for logging, and name it. Without a name the log
% comes back as "unnamed 1..21" and has to be decoded by magnitude.
%
% A block exposes its statistic ports in the order the parameters appear in
% its dialog, not the order they were switched on above, and the statistics
% come BEFORE the entity output port -- so take the first N. (Logging the
% entity port instead yields a struct rather than a number, which is how the
% off-by-one showed up.)
set_param(mdl, 'SignalLogging', 'on', 'SignalLoggingName', 'logsout');
statsOf = containers.Map();
for q = {'UploadQueue', 'ReviewQueue', 'HospitalQueue'}
    statsOf(q{1}) = {'NumberEntitiesInBlock', 'AverageWait', 'AverageQueueLength'};
end
statsOf('Reviewers') = {'AverageWait', 'Utilization'};
statsOf('Capture') = {'Utilization'};
for t = {'Recall', 'RecallAfterReview', 'LostToFollowUp', 'Treated'}
    statsOf(t{1}) = {'NumberEntitiesArrived'};
end
blocks = keys(statsOf);
for i = 1:numel(blocks)
    ph = get_param([mdl '/' blocks{i}], 'PortHandles');
    nm = statsOf(blocks{i});
    if numel(ph.Outport) < numel(nm), continue; end
    for j = 1:numel(nm)
        try
            set_param(ph.Outport(j), 'DataLogging', 'on', ...
                'DataLoggingNameMode', 'Custom', ...
                'DataLoggingName', [blocks{i} '.' nm{j}]);
        catch
            % entity ports cannot be logged; only the statistic ports can
        end
    end
end
save_system(mdl);
fprintf('Built %s.slx. Run: out = sim(''%s'');\n', mdl, mdl);

    function add(blockPath, name, pos)
        % blockPath is written as 'simevents/Entity Server'; swap in whatever
        % the library is actually called on this release.
        parts = split(blockPath, '/');
        add_block([lib '/' parts{end}], [mdl '/' name], 'Position', [pos, pos + [70 40]]);
    end
    function conn(a, b)
        try
            add_line(mdl, a, b, 'autorouting', 'on');
        catch e
            fprintf(2, 'add_line %s -> %s failed: %s\n', a, b, e.message);
        end
    end
    function trySet(blk, varargin)
        % varargin is one or more name/value pairs. Some parameters are
        % validated against each other -- AttributeName and
        % AttributeInitialValue must list the same number of pipe-separated
        % entries -- so those have to be set in one call or neither can ever
        % be applied.
        try
            set_param([mdl '/' blk], varargin{:});
        catch e
            fprintf(2, 'set_param %s.{%s} rejected (%s). Check get_param(''%s/%s'',''DialogParameters'').\n', ...
                blk, strjoin(varargin(1:2:end), ', '), e.message, mdl, blk);
        end
    end
end
