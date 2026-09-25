function q = quality_gate(rgb, cfgPath)
%QUALITY_GATE  Rule-based image checks mirroring drishti/quality/gate.py.
%   Each failed check maps to one instruction a health worker can act on.
%   Thresholds are read from configs/quality.json (fitted in Python on
%   DDR valid). UNTESTED here (no MATLAB licence on the build machine).

if nargin < 2
    cfgPath = fullfile(fileparts(mfilename('fullpath')), '..', 'configs', 'quality.json');
end
cfg = struct('min_fov_fraction', 0.45, 'dark_median', 45, 'bright_median', 200, 'max_clipped', 0.08, 'min_evenness', 0.35, 'min_focus', 0.012);
if isfile(cfgPath)
    c = jsondecode(fileread(cfgPath));
    fn = fieldnames(cfg);
    for i = 1:numel(fn), if isfield(c, fn{i}), cfg.(fn{i}) = c.(fn{i}); end, end
end
instr = struct( ...
    'off_centre', 'The retina is cut off. Ask the patient to look straight at the fixation light and centre the camera on the pupil.', ...
    'too_dark', 'Too dark. Dim the room lights so the pupil stays large, move the camera closer, and retake.', ...
    'too_bright', 'Too bright or washed out. Reduce the flash and tilt slightly to avoid reflections.', ...
    'uneven', 'One side is shadowed. Re-align the camera with the centre of the pupil and retake.', ...
    'blurry', 'Blurry. Ask the patient not to blink or move, hold the camera steady, refocus, and retake.');

[crop, fov] = drishti_preprocess(rgb, 512, false);
m.fov_fraction = mean(fov(:));
[xx, yy] = meshgrid(0:511, 0:511);
circle = (xx - 255.5).^2 + (yy - 255.5).^2 <= 250^2;
m.completeness = sum(fov(:) & circle(:)) / sum(circle(:));
lum = double(rgb2gray(crop));
inner = imerode(fov, strel('square', 15));
v = lum(inner);
m.median = median(v);
m.clipped = mean(v >= 250 | v <= 5);
sec = [];
for i = 0:2
    for j = 0:2
        r = i * 170 + (1:170); c = j * 170 + (1:170);
        blk = lum(r, c); msk = inner(r, c);
        if nnz(msk) > 500, sec(end + 1) = median(blk(msk)); end %#ok<AGROW>
    end
end
m.evenness = min(sec) / max(max(sec), 1);
g = double(crop(:, :, 2));
% cv2.Laplacian(green, ksize=3) does NOT apply the textbook 4-neighbour
% Laplacian -- that is what ksize=1 gives. For ksize>1 OpenCV builds the
% operator from Sobel second derivatives, and for ksize=3 the effective
% kernel is [2 0 2; 0 -8 0; 2 0 2]: twice the magnitude, diagonal rather
% than cross. Using the textbook kernel makes |lap| 2.45x too small, and
% because focus is a variance it comes out 6x too small, so a sharp
% photograph is reported as blurry.
lap = imfilter(g, [2 0 2; 0 -8 0; 2 0 2], 'symmetric');
% Same kernel-size trap as local_average: cv2.GaussianBlur(ksize = 0) on a
% float32 image truncates at 4 sigma (65 taps at sigma 8), imgaussfilt at
% 2 sigma (33). The tighter kernel tracks local detail instead of averaging
% it out, so local_sd comes out too large, lap/(sd+4) too small, and the
% focus score too low -- a sharp photograph gets reported as blurry.
sig = 8;
fs = bitor(round(8 * sig + 1), 1);
blur1 = imgaussfilt(g, sig, 'FilterSize', fs, 'Padding', 'symmetric');
blur2 = imgaussfilt(g.^2, sig, 'FilterSize', fs, 'Padding', 'symmetric');
sd = sqrt(max(blur2 - blur1.^2, 1));
r = lap(inner) ./ (sd(inner) + 4);
m.focus = var(r, 1) / 100;   % var(...,1) normalises by N, as numpy.var does

probs = {};
if m.fov_fraction < cfg.min_fov_fraction || m.completeness < 0.6, probs{end + 1} = 'off_centre'; end
if m.median < cfg.dark_median, probs{end + 1} = 'too_dark';
elseif m.median > cfg.bright_median || m.clipped > cfg.max_clipped, probs{end + 1} = 'too_bright'; end
if m.evenness < cfg.min_evenness, probs{end + 1} = 'uneven'; end
if m.focus < cfg.min_focus, probs{end + 1} = 'blurry'; end
q.metrics = m;
q.problems = probs;
q.instructions = cellfun(@(p) instr.(p), probs, 'UniformOutput', false);
q.passed = isempty(probs);
end
