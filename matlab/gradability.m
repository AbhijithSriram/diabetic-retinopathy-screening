function s = gradability(crop, onnxDir, man)
%GRADABILITY  Learned gradability score (0..1) from quality.onnx.
%   s = gradability(crop, onnxDir, man)   crop: raw 512 crop from drishti_preprocess
%   Returns NaN when quality.onnx is not in the export.
%
%   Mirrors drishti/pipeline.py _gradability: the raw (not Graham-normalised)
%   crop resized to 384 with OpenCV's INTER_AREA (resize_area.m),
%   ImageNet-normalised, one sigmoid output.
persistent cache
if isempty(cache), cache = containers.Map(); end
s = NaN;
if ~isfield(man, 'quality'), return; end
f = fullfile(onnxDir, man.quality.file);
if ~isfile(f), return; end
if ~isKey(cache, f)
    cache(f) = importNetworkFromONNX(f, 'InputDataFormats', 'BCSS');
end
sz = man.quality.size;
x = resize_area(crop, sz, sz);
X = on_device(dlarray(normalise_imagenet(x, man), 'SSCB'));
s = double(gather(extractdata(predict(cache(f), X))));
end
