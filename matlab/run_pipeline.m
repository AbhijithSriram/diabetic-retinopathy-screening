function result = run_pipeline(imagePath, onnxDir)
%RUN_PIPELINE  Screen one fundus photograph with the exported ONNX models.
%   result = run_pipeline('photo.jpg', '../runs/onnx_matlab')
%
%   quality checks -> grader ensemble (probabilities, P(referable), Grad-CAM)
%   -> lesion maps -> connected components (lesion counts, sub-pixel MA
%   centres) -> ICDR rule engine -> triage policy -> figure + struct.
%
%   Toolboxes: Deep Learning Toolbox (+ Converter for ONNX Model Format),
%   Image Processing Toolbox. UNTESTED here: no MATLAB licence on the build
%   machine. The Python pipeline (drishti/pipeline.py) is the reference.

if nargin < 2, onnxDir = fullfile(fileparts(mfilename('fullpath')), '..', 'runs', 'onnx_matlab'); end
man = jsondecode(fileread(fullfile(onnxDir, 'manifest.json')));
rgb = imread(imagePath);
if size(rgb, 3) == 1, rgb = repmat(rgb, 1, 1, 3); end

%% quality gate (rules only; the learned gradability model is Python-side)
q = quality_gate(rgb);
result.quality = q;
if ~q.passed
    result.outcome = 'recapture';
    fprintf('Recapture: %s\n', strjoin(q.instructions, ' '));
    return
end

%% grader ensemble (scripts/export_matlab.py -> grader_<run>.onnx, one per member)
crop = drishti_preprocess(rgb, 512, false);        % for display
g = grade_ensemble(rgb, onnxDir, man);
pref = g.p_referable;
result.probs = g.probs;
result.p_referable = pref;
result.member_p_referable = g.members;

% Grad-CAM on log P(grade >= 2) = logsigmoid(l0) + logsigmoid(l1), first member
cam = gradcam_referable(g.net, g.X);
result.gradcam = imresize(cam, [size(crop, 1) size(crop, 2)], 'bilinear');

%% lesions
[big, ~, geomBig] = drishti_preprocess(rgb, man.lesion_size, false); %#ok<ASGLU>
lin = grader_input(big, false);                    % Graham, as drishti/lesions/data.py full_input
seg = importNetworkFromONNX(fullfile(onnxDir, 'lesions.onnx'), 'InputDataFormats', 'BCSS');
X = dlarray(normalise_imagenet(lin, man), 'SSCB');
% flip-TTA averaged in logit space, -30 outside the field of view, then a
% double-precision sigmoid so confident peaks stay distinct (drishti/lesions/infer.py)
L = extractdata(predict(seg, X));
Lf = extractdata(predict(seg, X(:, end:-1:1, :, :)));
L = double(squeeze((L + Lf(:, end:-1:1, :, :)) / 2));  % H x W x 5
fovBig = max(big, [], 3) > 8;
L(repmat(~fovBig, 1, 1, 5)) = -30;
maps = 1 ./ (1 + exp(-L));
names = {'MA', 'HE', 'EX', 'SE'};
minArea = struct('MA', 2, 'HE', 6, 'EX', 4, 'SE', 20, 'OD', 200);
lesions = struct('kind', {}, 'x', {}, 'y', {}, 'area', {}, 'peak', {});
for c = 1:4
    t = man.lesion_thresholds.(names{c});
    peakMin = man.lesion_thresholds.(['peak_' names{c}]);  % confidence filter, calibrated on DDR valid
    P = maps(:, :, c);
    cc = bwconncomp(P >= t, 8);
    st = regionprops(cc, P, 'Area', 'WeightedCentroid', 'MaxIntensity');
    for i = 1:numel(st)
        if st(i).Area < minArea.(names{c}) || st(i).MaxIntensity < peakMin, continue; end
        % weighted centroid = sub-pixel lesion centre (MATLAB 1-based -> 0-based)
        lesions(end + 1) = struct('kind', names{c}, 'x', st(i).WeightedCentroid(1) - 1, ...
            'y', st(i).WeightedCentroid(2) - 1, 'area', st(i).Area, 'peak', st(i).MaxIntensity); %#ok<AGROW>
    end
end
od = [];
ccOD = bwconncomp(maps(:, :, 5) >= man.lesion_thresholds.OD, 8);
if ccOD.NumObjects > 0
    st = regionprops(ccOD, 'Area', 'Centroid');
    [a, k] = max([st.Area]);
    if a >= minArea.OD
        od = [st(k).Centroid - 1, sqrt(a / pi)];
    end
end
result.lesions = lesions;

%% fovea (landmarks.onnx on the 512 crop, as drishti/landmarks/landmarks.py predict)
fovea = [];
lmkFile = fullfile(onnxDir, 'landmarks.onnx');
if isfile(lmkFile)
    lmk = importNetworkFromONNX(lmkFile, 'InputDataFormats', 'BCSS');
    hm = double(squeeze(extractdata(predict(lmk, ...
        dlarray(normalise_imagenet(grader_input(crop), man), 'SSCB')))));  % 512 x 512 x 2, sigmoid
    [fx, fy, fh] = heat_peak(hm(:, :, 2));
    result.fovea = struct('x', fx, 'y', fy, 'confidence', fh);
    if fh >= 0.2  % same confidence gate as drishti/rules/icdr.py assess
        fovea = [fx fy] * man.lesion_size / 512;
    end
end
result.rules = rule_engine(lesions, od, fovea, man.lesion_size);

%% triage (policy fitted in Python on the calibration split, then frozen)
pol = man.policy;
if pref >= pol.lambda_high
    result.outcome = 'refer';
elseif pref < pol.lambda_low && ~result.rules.lesion_flag && ~result.rules.dme_flag
    result.outcome = 'recall';
else
    result.outcome = 'review';
end

%% figure
figure('Name', 'DrishtiTrust', 'Color', 'w');
subplot(1, 3, 1); imshow(crop); title('Photo');
subplot(1, 3, 2); imshow(big); hold on;
cols = struct('MA', [1 .15 .15], 'HE', [.6 0 0], 'EX', [1 .85 0], 'SE', [.9 .9 1]);
for i = 1:numel(lesions)
    plot(lesions(i).x + 1, lesions(i).y + 1, 'o', 'Color', cols.(lesions(i).kind), 'MarkerSize', 6);
end
title(sprintf('Lesions: MA %d HE %d EX %d SE %d', result.rules.counts.MA, result.rules.counts.HE, result.rules.counts.EX, result.rules.counts.SE));
subplot(1, 3, 3); imshow(crop); hold on;
h = imshow(ind2rgb(uint8(255 * result.gradcam), jet(256))); set(h, 'AlphaData', 0.45);
title(sprintf('Grad-CAM · P(referable)=%.2f · %s', pref, upper(result.outcome)));
end

function cam = gradcam_referable(net, X)
[feats, grads] = dlfeval(@camGrad, net, X);
feats = extractdata(feats); grads = extractdata(grads);
w = mean(grads, [1 2]);
cam = max(sum(feats .* w, 3), 0);
cam = imresize(cam, [size(X, 1) size(X, 2)], 'bilinear');
cam = cam / max(cam(:) + eps);
end

function [feats, grads] = camGrad(net, X)
[logits, feats] = forward(net, X);  % export order: logits, features
target = -softplus(-logits(1)) - softplus(-logits(2));  % logsigmoid(l0) + logsigmoid(l1)
grads = dlgradient(target, feats);
end

function y = softplus(x)
y = log(1 + exp(x));
end

function [x, y, h] = heat_peak(hm)
% argmax refined by a 7x7 weighted centroid, 0-based coordinates
[h, idx] = max(hm(:));
[r, c] = ind2sub(size(hm), idx);
r0 = max(1, r - 3); r1 = min(size(hm, 1), r + 3);
c0 = max(1, c - 3); c1 = min(size(hm, 2), c + 3);
w = max(hm(r0:r1, c0:c1), 0);
[cc, rr] = meshgrid(c0:c1, r0:r1);
if sum(w(:)) > 0
    x = sum(w(:) .* cc(:)) / sum(w(:)) - 1;
    y = sum(w(:) .* rr(:)) / sum(w(:)) - 1;
else
    x = c - 1; y = r - 1;
end
end
