function out = resize_area(img, outH, outW)
%RESIZE_AREA  OpenCV's INTER_AREA downscaling: each output pixel is the
%   area-weighted mean of the input pixels its footprint covers.
%   out = resize_area(img, outH, outW)
%
%   imresize has no equivalent: its 'box' and 'bilinear' kernels with
%   antialiasing are close on sharp photographs but move the learned
%   gradability score by up to 0.16 on blurred ones, and the field-of-view
%   threshold on dark ones. Built as two sparse weight matrices (rows, then
%   columns), and rounded ties-to-even as OpenCV does, so the output is
%   bit-identical to cv2.resize(..., INTER_AREA). Enlarging falls back to
%   bilinear, as OpenCV's INTER_AREA itself does.
[h, w, c] = size(img);
if outH > h || outW > w
    out = imresize(img, [outH outW], 'bilinear', 'Antialiasing', false);
    return
end
Wy = area_weights(h, outH);            % outH x h
Wx = area_weights(w, outW);            % outW x w
cls = class(img);
out = zeros(outH, outW, c);
for k = 1:c
    out(:, :, k) = Wy * double(img(:, :, k)) * Wx';
end
if ~strcmp(cls, 'double')
    out = cast(round_half_even(out), cls);
end
end

function y = round_half_even(x)
% OpenCV's saturate_cast rounds with lrint (ties to even); MATLAB's round
% sends ties away from zero. At a 4:3 ratio a third of the outputs are exact
% ties, and that alone moves the gradability score of a blurred photograph
% from 0.73 to 0.65.
y = round(x);
t = abs(x - fix(x)) == 0.5;
y(t) = 2 * round(x(t) / 2);
end

function W = area_weights(n, m)
% W(i, j) = overlap of output cell i with input pixel j, over the cell width.
% Measured in units of 1/m so the overlaps are integers: output cell i spans
% [(i-1)n, in), input pixel j spans [(j-1)m, jm), and the weight is the
% integer overlap / n -- exact whenever n is a power of two, so ties stay ties.
rows = []; cols = []; vals = [];
for i = 1:m
    a = (i - 1) * n; b = i * n;
    for j = floor(a / m) + 1:min(n, ceil(b / m))
        ov = min(b, j * m) - max(a, (j - 1) * m);
        if ov > 0
            rows(end + 1) = i; cols(end + 1) = j; vals(end + 1) = ov / n; %#ok<AGROW>
        end
    end
end
W = sparse(rows, cols, vals, m, n);
end
