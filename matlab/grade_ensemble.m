function g = grade_ensemble(rgb, onnxDir, man)
%GRADE_ENSEMBLE  ICDR grade probabilities from the exported grader ensemble.
%   g = grade_ensemble(rgb, onnxDir, man)
%
%   rgb      original fundus photograph (uint8, H x W x 3)
%   onnxDir  folder written by scripts/export_matlab.py
%   man      jsondecode of its manifest.json
%
%   Returns
%     g.probs        1x5 P(ICDR grade 0..4), ensemble mean
%     g.p_referable  P(grade >= 2) = sum of g.probs(3:5)
%     g.members      per-member P(referable), for the report
%     g.net, g.X     first member and its input, for Grad-CAM
%
%   Replays exactly what the Python evaluation did (drishti/grading/train.py
%   and scripts/ensemble.py): each member sees the photograph cropped and
%   Graham-normalised at its own input size, its logits are averaged over the
%   image and its horizontal flip, CORN class probabilities are formed from
%   those logits, and the ensemble is the mean of the members' class
%   probabilities. The temperature is already folded into each ONNX file.
%   scripts/export_matlab.py checks this recipe against the Python
%   predictions before it writes the manifest.

nets = load_graders(onnxDir, man);
m = numel(nets);
P = zeros(m, 5);
for i = 1:m
    sz = man.graders(i).size;
    img = grader_input(drishti_preprocess(rgb, sz, false));
    X = dlarray(normalise_imagenet(img, man), 'SSCB');
    Xf = X(:, end:-1:1, :, :);                    % horizontal flip (width is dim 2)
    l = (member_logits(nets{i}, X) + member_logits(nets{i}, Xf)) / 2;
    P(i, :) = corn_probs(l);
    if i == 1
        g.net = nets{i};
        g.X = X;
    end
end
g.probs = mean(P, 1);
g.p_referable = sum(g.probs(3:5));
g.members = sum(P(:, 3:5), 2)';
end

function l = member_logits(net, X)
[lg, ~] = predict(net, X);                        % outputs: logits, features
l = double(reshape(extractdata(lg), 1, []));
end

function p = corn_probs(l)
% CORN: P(y > k) is the running product of sigmoids, monotone by construction.
s = 1 ./ (1 + exp(-l));
c = cumprod(s);
p = max([1 c] - [c 0], 0);
end

function nets = load_graders(onnxDir, man)
% importNetworkFromONNX takes seconds per network, so the imported
% networks are kept between calls.
persistent cache
if isempty(cache), cache = containers.Map(); end
nets = cell(1, numel(man.graders));
for i = 1:numel(man.graders)
    f = fullfile(onnxDir, man.graders(i).file);
    if ~isKey(cache, f)
        cache(f) = importNetworkFromONNX(f, 'InputDataFormats', 'BCSS');
    end
    nets{i} = cache(f);
end
end
