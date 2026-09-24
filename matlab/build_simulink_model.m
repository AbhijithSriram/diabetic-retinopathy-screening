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
%   UNTESTED: written without a MATLAB/SimEvents licence on the build
%   machine. Parameter names are set through trySet(), which prints any
%   name your release rejects; fix those with
%   get_param(<block>, 'DialogParameters'). matlab/district_sim.m runs the
%   same model in base MATLAB as a cross-check.

if nargin < 1
    paramsFile = fullfile(fileparts(mfilename('fullpath')), '..', 'sim', 'params.json');
end
P = jsondecode(fileread(paramsFile));
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
trySet('Patients', 'AttributeName', 'Route|Refer|Attends');
trySet('Patients', 'AttributeInitialValue', '1|0|0');
trySet('Patients', 'GenerateAction', sprintf([ ...
    'u = rand(); c = cumsum([%s]); entity.Route = find(u <= c, 1);\n' ...
    'entity.Refer = double(rand() < %g);\n' ...
    'entity.Attends = double(rand() < %g);'], num2str(routeP, '%g '), P.review.refer_after_review, P.referral.completion_with_engine));

add('simevents/Entity Server', 'Capture', [180 100]);
trySet('Capture', 'Capacity', num2str(P.district.sites));
trySet('Capture', 'ServiceTimeSource', 'MATLAB action');
m = P.capture.minutes_per_attempt;
trySet('Capture', 'ServiceTimeAction', sprintf('dt = triangular_draw(%g, %g, %g) / 60;', m(1), m(2), m(3)));

add('simevents/Entity Output Switch', 'Triage', [320 100]);
trySet('Triage', 'NumberOutputPorts', '4');
trySet('Triage', 'SwitchingCriterion', 'From attribute');
trySet('Triage', 'SwitchAttributeName', 'Route');

add('simevents/Entity Terminator', 'Recall', [480 20]);
add('simevents/Entity Queue', 'UploadQueue', [480 100]);
add('simevents/Entity Server', 'Uplink', [600 100]);
trySet('Uplink', 'Capacity', num2str(P.district.sites));
trySet('Uplink', 'ServiceTime', num2str(uploadH));
add('simevents/Entity Queue', 'ReviewQueue', [720 100]);
add('simevents/Entity Server', 'Reviewers', [840 100]);
trySet('Reviewers', 'Capacity', num2str(P.review.reviewers));
trySet('Reviewers', 'ServiceTime', num2str(reviewH));
add('simevents/Entity Output Switch', 'ReviewDecision', [960 100]);
trySet('ReviewDecision', 'NumberOutputPorts', '2');
trySet('ReviewDecision', 'SwitchingCriterion', 'From attribute');
trySet('ReviewDecision', 'SwitchAttributeName', 'RefIdx');
trySet('Reviewers', 'ExitAction', 'entity.RefIdx = entity.Refer + 1;');
add('simevents/Entity Terminator', 'RecallAfterReview', [1100 60]);

add('simevents/Entity Input Switch', 'Referrals', [1100 200]);
trySet('Referrals', 'NumberInputPorts', '3');
add('simevents/Entity Output Switch', 'AttendsSwitch', [1220 200]);
trySet('AttendsSwitch', 'NumberOutputPorts', '2');
trySet('AttendsSwitch', 'SwitchingCriterion', 'From attribute');
trySet('AttendsSwitch', 'SwitchAttributeName', 'AttIdx');
trySet('Referrals', 'ExitAction', 'entity.AttIdx = entity.Attends + 1;');
add('simevents/Entity Terminator', 'LostToFollowUp', [1360 160]);
add('simevents/Entity Queue', 'HospitalQueue', [1360 240]);
add('simevents/Entity Server', 'HospitalSlots', [1480 240]);
trySet('HospitalSlots', 'ServiceTime', num2str(hospitalH));
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

% queue statistics for the Simulation Data Inspector
for q = {'UploadQueue', 'ReviewQueue', 'HospitalQueue'}
    trySet(q{1}, 'AverageWait', 'on');
    trySet(q{1}, 'NumberEntitiesInBlock', 'on');
end
save_system(mdl);
fprintf('Built %s.slx. Run: out = sim(''%s'');\n', mdl, mdl);

    function add(lib, name, pos)
        add_block(lib, [mdl '/' name], 'Position', [pos, pos + [70 40]]);
    end
    function conn(a, b)
        try
            add_line(mdl, a, b, 'autorouting', 'on');
        catch e
            fprintf(2, 'add_line %s -> %s failed: %s\n', a, b, e.message);
        end
    end
    function trySet(blk, name, val)
        try
            set_param([mdl '/' blk], name, val);
        catch e
            fprintf(2, 'set_param %s.%s rejected (%s). Check get_param(''%s/%s'',''DialogParameters'').\n', blk, name, e.message, mdl, blk);
        end
    end
end
