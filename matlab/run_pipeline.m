function result = run_pipeline(imagePath, onnxDir)
%RUN_PIPELINE  Screen one fundus photograph with the exported ONNX models.
%   result = run_pipeline('photo.jpg', '../runs/onnx_matlab')
%
%   drishti_screen (quality checks -> grader ensemble + Grad-CAM -> lesion
%   maps -> sub-pixel lesion centres -> landmarks -> ICDR rule engine), then
%   the triage policy and a figure. Networks stay cached between calls.
%
%   Toolboxes: Deep Learning Toolbox (+ Converter for ONNX Model Format),
%   Image Processing Toolbox. Verified on R2026b, 25 September 2026: grades a
%   photograph end to end, and check_export agrees with the Python reference
%   (drishti/pipeline.py) to within 0.005 on P(referable).

if nargin < 2, onnxDir = fullfile(fileparts(mfilename('fullpath')), '..', 'runs', 'onnx_matlab'); end
man = jsondecode(fileread(fullfile(onnxDir, 'manifest.json')));
result = drishti_screen(imagePath, onnxDir, true);
if ~result.quality.passed
    result.outcome = 'recapture';
    fprintf('Recapture: %s\n', strjoin(result.quality.instructions, ' '));
    return
end
pref = result.p_referable;

%% triage (policy fitted in Python on the calibration split, then frozen;
% which rule flags may hold a case back from recall is set in configs/triage.json)
tcfg = struct('use_lesion_flag', false, 'use_dme_flag', true);
tfile = fullfile(fileparts(mfilename('fullpath')), '..', 'configs', 'triage.json');
if isfile(tfile), tcfg = jsondecode(fileread(tfile)); end
flag = (tcfg.use_lesion_flag && result.rules.lesion_flag) || (tcfg.use_dme_flag && result.rules.dme_flag);
pol = man.policy;
if pref >= pol.lambda_high
    result.outcome = 'refer';
elseif pref < pol.lambda_low && ~flag
    result.outcome = 'recall';
else
    result.outcome = 'review';
end

%% figure
c = result.rules.counts;
figure('Name', 'DrishtiTrust', 'Color', 'w');
subplot(1, 3, 1); imshow(result.images.photo); title('Photo');
subplot(1, 3, 2); imshow(result.images.lesions);
title(sprintf('Lesions: MA %d HE %d EX %d SE %d', c.MA, c.HE, c.EX, c.SE));
subplot(1, 3, 3); imshow(result.images.gradcam);
title(sprintf('Grad-CAM · P(referable)=%.2f · %s', pref, upper(result.outcome)));
end
