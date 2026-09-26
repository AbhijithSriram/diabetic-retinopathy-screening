function T = check_export(onnxDir)
%CHECK_EXPORT  Does MATLAB reproduce the Python grader ensemble?
%   T = check_export('../runs/onnx_matlab')
%
%   Runs grade_ensemble on the app's sample photographs and compares
%   P(referable) with the reference values scripts/export_matlab.py wrote
%   into manifest.json (same ONNX files, Python preprocessing). Run this
%   once on any licensed machine before trusting the MATLAB path.
%
%   Tolerance 0.02: the preprocessing is reimplemented (imresize/imgaussfilt
%   vs OpenCV), so agreement is to a grey level or so, not to the bit.
%   A difference above ~0.1 means something is wired wrongly -- usually the
%   output order of an imported network or the flip dimension.

here = fileparts(mfilename('fullpath'));
if nargin < 1, onnxDir = fullfile(here, '..', 'runs', 'onnx_matlab'); end
man = jsondecode(fileread(fullfile(onnxDir, 'manifest.json')));
ref = man.reference;
n = numel(ref);
file = strings(n, 1); python = zeros(n, 1); matlab = zeros(n, 1);
for i = 1:n
    rgb = imread(fullfile(here, '..', 'app', 'samples', ref(i).file));
    g = grade_ensemble(rgb, onnxDir, man);
    file(i) = ref(i).file;
    python(i) = ref(i).p_referable;
    matlab(i) = g.p_referable;
end
T = table(file, python, matlab, abs(python - matlab), 'VariableNames', ...
    {'file', 'python', 'matlab', 'abs_diff'});
disp(T)
if max(T.abs_diff) <= 0.02
    fprintf('PASS: MATLAB matches Python within 0.02 on %d photographs\n', n);
else
    fprintf('CHECK: max difference %.3f exceeds 0.02\n', max(T.abs_diff));
end

% learned gradability (quality.onnx), on the samples and on degraded copies
% of them written by scripts/export_matlab.py; the score is steep on blurred
% photographs, so 0.05 here is a tighter match than it looks
if isfield(man, 'quality')
    qr = man.quality.reference;
    worst = 0;
    for i = 1:numel(qr)
        f = fullfile(here, '..', 'app', 'samples', qr(i).file);
        if contains(qr(i).file, '/'), f = fullfile(onnxDir, qr(i).file); end
        s = gradability(drishti_preprocess(imread(f), 512, false), onnxDir, man);
        worst = max(worst, abs(s - qr(i).gradability));
    end
    if worst <= 0.05
        fprintf('PASS: gradability matches Python within 0.05 on %d photographs (max %.3f)\n', numel(qr), worst);
    else
        fprintf('CHECK: gradability differs by %.3f\n', worst);
    end
end
end
