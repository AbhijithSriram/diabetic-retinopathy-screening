function [img, fov, geom] = drishti_preprocess(rgb, outSize, normalise)
%DRISHTI_PREPROCESS  MATLAB port of drishti/preprocess.py (prepare).
%   [img, fov, geom] = drishti_preprocess(rgb, outSize, normalise)
%   1. field-of-view mask: threshold on the brightest channel, keep the
%      largest component, fill holes;
%   2. square crop centred on the FOV (diameter = larger bbox side), padded
%      with zeros, resized to outSize;
%   3. optional Graham normalisation: 4*I - 4*Gauss(I, size/30) + 128.
%   Requires Image Processing Toolbox. Verified against the Python output on
%   R2026b by check_export (max difference 0.005). imgaussfilt truncates at
%   2 sigma where OpenCV uses 4, so FilterSize is set explicitly below; see
%   matlab/README.md.

if nargin < 3, normalise = false; end
rgb = im2uint8(rgb);
[h, w, ~] = size(rgb);
f = 512 / max(h, w);
small = resize_area(rgb, max(1, round(h * f)), max(1, round(w * f)));  % cv2.INTER_AREA, as preprocess.py fov_mask
bright = double(max(small, [], 3));
thr = max(12, 0.08 * prctile(bright(:), 99));
m = bright > thr;
m = imopen(m, strel('square', 5));
m = bwareafilt(m, 1);
m = imfill(m, 'holes');
m = imclose(m, strel('square', 9));
% cv2.INTER_NEAREST takes source index floor(dst * src/dst) with no
% half-pixel offset; imresize 'nearest' samples pixel centres, which shifts
% the mask edge, and so the crop, by a pixel
ri = min(size(m, 1), floor((0:h - 1) * size(m, 1) / h) + 1);
ci = min(size(m, 2), floor((0:w - 1) * size(m, 2) / w) + 1);
mask = m(ri, ci);

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
%
% The kernel size has to be set explicitly. OpenCV's GaussianBlur, called
% with ksize = 0 on a float32 image, truncates the Gaussian at 4 sigma:
% ksize = round(8*sigma + 1), forced odd. imgaussfilt defaults to 2 sigma,
% roughly half the radius, which is a different filter -- and Graham
% normalisation multiplies the difference by four.
[h, w, ~] = size(I);
s = sigma / 4;
small = imresize(I, [floor(h / 4) floor(w / 4)], 'box');
small = imgaussfilt(small, s, 'FilterSize', bitor(round(8 * s + 1), 1), 'Padding', 'symmetric');
B = imresize(small, [h w], 'bilinear');
end
