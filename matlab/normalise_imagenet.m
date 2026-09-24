function x = normalise_imagenet(img, man)
%NORMALISE_IMAGENET  uint8 RGB -> single, ImageNet mean/std per channel
%   (the normalisation every exported network was trained with).
x = single(img) / 255;
for c = 1:3
    x(:, :, c) = (x(:, :, c) - man.mean(c)) / man.std(c);
end
end
