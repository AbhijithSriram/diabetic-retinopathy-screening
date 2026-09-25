function out = grader_input(crop, useCircle)
%GRADER_INPUT  Graham normalisation of an already-cropped fundus image,
%   exactly as the networks saw it in training.
%   out = grader_input(crop)          grader input   (drishti/data/datasets.py grader_input)
%   out = grader_input(crop, false)   lesion input   (drishti/lesions/data.py full_input)
%
%   crop is the square, un-normalised output of drishti_preprocess(rgb, n, false).
%   The field of view is re-derived from the crop (brightest channel > 8,
%   intersected with the inscribed circle for the grader), and a rim of
%   4*floor(n/128)+1 pixels inside it is set to the neutral 128, because the
%   large blur mixes the black surround into the edge.

if nargin < 2, useCircle = true; end
n = size(crop, 1);
fov = max(crop, [], 3) > 8;
if useCircle
    [xx, yy] = meshgrid(0:n - 1, 0:n - 1);
    c = (n - 1) / 2;
    fov = fov & ((xx - c).^2 + (yy - c).^2 <= (n / 2)^2);
end
I = double(crop);
% Match OpenCV's kernel, not imgaussfilt's default: GaussianBlur(ksize = 0)
% on a float32 image truncates at 4 sigma (ksize = round(8*sigma + 1), odd),
% while imgaussfilt truncates at 2 sigma. Graham multiplies the resulting
% difference by four, so the default costs ~2 grey levels on 28% of pixels.
s = n / 30 / 4;
small = imresize(I, [floor(n / 4) floor(n / 4)], 'box');
small = imgaussfilt(small, s, 'FilterSize', bitor(round(8 * s + 1), 1), 'Padding', 'symmetric');
B = imresize(small, [n n], 'bilinear');
out = uint8(floor(min(max(4 * I - 4 * B + 128, 0), 255)));  % Python truncates, so floor
k = 4 * max(1, floor(n / 128)) + 1;
rim = imerode(fov, strel('square', k));
out(repmat(~rim, 1, 1, 3)) = 128;
end
