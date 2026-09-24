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
end
