function [img, fov, geom] = drishti_preprocess(rgb, outSize, normalise)
%DRISHTI_PREPROCESS  MATLAB port of drishti/preprocess.py (prepare).
%   [img, fov, geom] = drishti_preprocess(rgb, outSize, normalise)
%   1. field-of-view mask: threshold on the brightest channel, keep the
%      largest component, fill holes;
%   2. square crop centred on the FOV (diameter = larger bbox side), padded
%      with zeros, resized to outSize;
%   3. optional Graham normalisation: 4*I - 4*Gauss(I, size/30) + 128.
%   Requires Image Processing Toolbox. UNTESTED in this repository's CI
%   (no MATLAB licence on the build machine): compare against the Python
%   output with matlab/check_parity.m before relying on it.

if nargin < 3, normalise = false; end
rgb = im2uint8(rgb);
[h, w, ~] = size(rgb);
f = 512 / max(h, w);
small = imresize(rgb, f, 'bilinear');
bright = double(max(small, [], 3));
thr = max(12, 0.08 * prctile(bright(:), 99));
m = bright > thr;
m = imopen(m, strel('square', 5));
m = bwareafilt(m, 1);
m = imfill(m, 'holes');
m = imclose(m, strel('square', 9));
mask = imresize(m, [h w], 'nearest');

[ys, xs] = find(mask);
side = max(max(xs) - min(xs) + 1, max(ys) - min(ys) + 1);
cx = (min(xs) - 1 + max(xs)) / 2;  % 0-based centre, as in Python
cy = (min(ys) - 1 + max(ys)) / 2;
x0 = cx - side / 2;  y0 = cy - side / 2;
s = outSize / side;

% affine: processed = (original - [x0 y0]) * s   (0-based pixel coordinates)
tform = affinetform2d([s 0 -x0 * s; 0 s -y0 * s; 0 0 1]);
ref = imref2d([outSize outSize], [-0.5 outSize - 0.5], [-0.5 outSize - 0.5]);
refIn = imref2d([h w], [-0.5 w - 0.5], [-0.5 h - 0.5]);
img = imwarp(rgb, refIn, tform, 'linear', 'OutputView', ref, 'FillValues', 0);
fov = imwarp(uint8(mask), refIn, tform, 'nearest', 'OutputView', ref, 'FillValues', 0) > 0;
img(repmat(~fov, 1, 1, 3)) = 0;

if normalise
    img = graham(img, fov);
end
geom = struct('x0', x0, 'y0', y0, 'side', side, 'out', outSize, 'cx', cx, 'cy', cy, 'radius', side / 2);
end

function out = graham(img, fov)
sigma = size(img, 1) / 30;
I = double(img);
B = local_average(I, sigma);
out = uint8(min(max(4 * I - 4 * B + 128, 0), 255));
rim = imerode(fov, strel('square', 5));
for k = 1:max(1, floor(size(img, 1) / 128)) - 1
    rim = imerode(rim, strel('square', 5));
end
for c = 1:3
    ch = out(:, :, c); ch(~rim) = 128; out(:, :, c) = ch;
end
end

function B = local_average(I, sigma)
% Same as drishti/preprocess.py local_average: blur at 1/4 resolution.
[h, w, ~] = size(I);
small = imresize(I, [floor(h / 4) floor(w / 4)], 'box');
small = imgaussfilt(small, sigma / 4);
B = imresize(small, [h w], 'bilinear');
end
