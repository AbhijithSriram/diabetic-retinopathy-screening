function r = drishti_screen(rgb, onnxDir, withImages)
%DRISHTI_SCREEN  Screen one fundus photograph with the exported ONNX models.
%   r = drishti_screen(rgb)                 % rgb: image array or file path
%   r = drishti_screen(rgb, onnxDir, withImages)
%
%   quality checks -> grader ensemble (probabilities, P(referable), Grad-CAM)
%   -> lesion maps -> connected components (lesion counts, sub-pixel MA
%   centres) -> landmarks -> ICDR rule engine. No triage and no figure:
%   run_pipeline adds both for interactive use, and the web app
%   (drishti/matlab_backend.py) applies the Python triage policy so there is
%   one source of truth for it.
%
%   Networks are imported once per MATLAB session and cached, so a server
%   pays the import cost on its first photograph only.
%
%   Only scalar structs, numeric arrays and cell arrays of text are returned,
%   because the MATLAB Engine for Python cannot pass struct arrays: lesions
%   come back as an N x 5 matrix [x y area peak kind] (0-based pixel
%   coordinates in the original photograph, kind indexing r.lesion_kinds).

if nargin < 2 || isempty(onnxDir), onnxDir = fullfile(fileparts(mfilename('fullpath')), '..', 'runs', 'onnx_matlab'); end
if nargin < 3, withImages = true; end
man = jsondecode(fileread(fullfile(onnxDir, 'manifest.json')));
if ischar(rgb) || isstring(rgb), rgb = imread(rgb); end
if size(rgb, 3) == 1, rgb = repmat(rgb, 1, 1, 3); end
rgb = im2uint8(rgb);

%% quality gate: framing rules + the learned gradability score (quality.onnx)
% combined as drishti/quality/gate.py check: with a score, the exposure /
% evenness / focus rules only explain a failure, they do not cause one
q = quality_gate(rgb);
crop = drishti_preprocess(rgb, 512, false);
q.gradability = gradability(crop, onnxDir, man);
if ~isnan(q.gradability)
    hard = q.problems(strcmp(q.problems, 'off_centre'));
    soft = q.problems(~strcmp(q.problems, 'off_centre'));
    q.passed = isempty(hard) && q.gradability >= man.quality.threshold;
    if q.passed
        q.problems = {}; q.instructions = {};
    elseif isempty(hard) && isempty(soft)  % failed on the score alone
        q.problems = {'ungradable'};
        q.instructions = {['The picture is not clear enough to read (possible cataract or small pupil). ' ...
            'Retake once; if it fails again the patient needs an in-person eye check.']};
    end
end
r.quality = q;
if withImages, r.images.photo = crop; end
if ~q.passed
    return
end

%% grader ensemble (scripts/export_matlab.py -> grader_<run>.onnx, one per member)
g = grade_ensemble(rgb, onnxDir, man);
r.probs = g.probs;
r.p_referable = g.p_referable;
r.member_p_referable = g.members;
if withImages
    % Grad-CAM on log P(grade >= 2) = logsigmoid(l0) + logsigmoid(l1), first member
    cam = imresize(gradcam_referable(g.net, g.X), [512 512], 'bilinear');
    r.images.gradcam = heat_overlay(crop, min(max(cam, 0), 1));
end

%% lesions
[big, ~, geomBig] = drishti_preprocess(rgb, man.lesion_size, false);
lin = grader_input(big, false);                    % Graham, as drishti/lesions/data.py full_input
seg = cached_net(fullfile(onnxDir, 'lesions.onnx'));
X = on_device(dlarray(normalise_imagenet(lin, man), 'SSCB'));
% flip-TTA averaged in logit space, -30 outside the field of view, then a
% double-precision sigmoid so confident peaks stay distinct (drishti/lesions/infer.py)
L = gather(extractdata(predict(seg, X)));
Lf = gather(extractdata(predict(seg, X(:, end:-1:1, :, :))));
L = double(squeeze((L + Lf(:, end:-1:1, :, :)) / 2));  % H x W x 5
fovBig = max(big, [], 3) > 8;
L(repmat(~fovBig, 1, 1, 5)) = -30;
maps = 1 ./ (1 + exp(-L));
names = {'MA', 'HE', 'EX', 'SE'};
minArea = struct('MA', 2, 'HE', 6, 'EX', 4, 'SE', 20, 'OD', 200);
lesions = struct('kind', {}, 'x', {}, 'y', {}, 'area', {}, 'peak', {});
keptMask = false([size(maps, 1) size(maps, 2) 4]);
for c = 1:4
    t = man.lesion_thresholds.(names{c});
    peakMin = man.lesion_thresholds.(['peak_' names{c}]);  % confidence filter, calibrated on DDR valid
    P = maps(:, :, c);
    cc = bwconncomp(P >= t, 8);
    st = regionprops(cc, P, 'Area', 'WeightedCentroid', 'MaxIntensity');
    km = false(size(P));
    for i = 1:numel(st)
        if st(i).Area < minArea.(names{c}) || st(i).MaxIntensity < peakMin, continue; end
        % weighted centroid = sub-pixel lesion centre (MATLAB 1-based -> 0-based)
        lesions(end + 1) = struct('kind', names{c}, 'x', st(i).WeightedCentroid(1) - 1, ...
            'y', st(i).WeightedCentroid(2) - 1, 'area', st(i).Area, 'peak', st(i).MaxIntensity); %#ok<AGROW>
        km(cc.PixelIdxList{i}) = true;
    end
    keptMask(:, :, c) = km;
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

%% fovea (landmarks.onnx on the 512 crop, as drishti/landmarks/landmarks.py predict)
fovea = [];
lmkFile = fullfile(onnxDir, 'landmarks.onnx');
if isfile(lmkFile)
    lmk = cached_net(lmkFile);
    hm = double(squeeze(gather(extractdata(predict(lmk, ...
        on_device(dlarray(normalise_imagenet(grader_input(crop), man), 'SSCB')))))));  % 512 x 512 x 2, sigmoid
    [fx, fy, fh] = heat_peak(hm(:, :, 2));
    r.fovea = struct('x', fx, 'y', fy, 'confidence', fh);
    if fh >= 0.2  % same confidence gate as drishti/rules/icdr.py assess
        fovea = [fx fy] * man.lesion_size / 512;
    end
end
r.rules = rule_engine(lesions, od, fovea, man.lesion_size);

% lesions in original-photograph coordinates, as drishti/lesions/infer.py to_original
s = geomBig.out / geomBig.side;
r.lesion_kinds = names;
r.lesions = zeros(numel(lesions), 5);
for i = 1:numel(lesions)
    r.lesions(i, :) = [lesions(i).x / s + geomBig.x0, lesions(i).y / s + geomBig.y0, ...
        lesions(i).area / s^2, lesions(i).peak, find(strcmp(names, lesions(i).kind))];
end
if isempty(od), r.optic_disc = nan(1, 3); else, r.optic_disc = od; end

if withImages
    r.images.lesions = imresize(lesion_overlay(big, keptMask, od, fovea, r.rules), [768 768], 'bilinear');
end
end

% ------------------------------------------------------------------ helpers
function net = cached_net(f)
persistent cache
if isempty(cache), cache = containers.Map(); end
if ~isKey(cache, f)
    cache(f) = importNetworkFromONNX(f, 'InputDataFormats', 'BCSS');
end
net = cache(f);
end

function out = heat_overlay(crop, cam)
% as drishti/explain/gradcam.py heat_overlay: jet at 45 % inside the field of view
hm = im2uint8(ind2rgb(uint8(255 * cam), jet(256)));
fov = repmat(max(crop, [], 3) > 8, 1, 1, 3);
out = crop;
mix = uint8(double(crop) * 0.55 + double(hm) * 0.45);
out(fov) = mix(fov);
end

function out = lesion_overlay(img, kept, od, fovea, rules)
% as drishti/lesions/infer.py overlay: outlines of the confident components
% (the ones the rule engine counts), MA as rings; optic disc green, fovea
% circle one disc diameter wide in cyan
colours = [255 40 40; 150 0 0; 255 220 0; 230 230 255];
out = img;
for c = 1:4
    m = kept(:, :, c);
    if ~any(m(:)), continue; end
    if c == 1
        edge = imdilate(m, ones(5)) & ~m;          % tiny MAs drawn as rings
    else
        edge = imdilate(bwperim(m, 8), ones(2));
    end
    out = paint(out, edge, colours(c, :));
end
[xx, yy] = meshgrid(0:size(img, 2) - 1, 0:size(img, 1) - 1);
if ~isempty(od)
    out = paint(out, abs(hypot(xx - od(1), yy - od(2)) - od(3)) < 1.5, [0 200 0]);
end
if ~isempty(fovea)
    out = paint(out, abs(hypot(xx - fovea(1), yy - fovea(2)) - rules.disc_diameter_px) < 1.5, [0 255 255]);
end
end

function img = paint(img, mask, colour)
for k = 1:3
    ch = img(:, :, k);
    ch(mask) = colour(k);
    img(:, :, k) = ch;
end
end

function cam = gradcam_referable(net, X)
[feats, grads] = dlfeval(@camGrad, net, X);
feats = gather(extractdata(feats)); grads = gather(extractdata(grads));
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
