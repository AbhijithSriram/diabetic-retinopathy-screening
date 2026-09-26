function X = on_device(X)
%ON_DEVICE  Move a network input to the GPU when MATLAB can use one.
%   Needs Parallel Computing Toolbox and a supported GPU; otherwise X is
%   returned unchanged and inference runs on the CPU. Set the environment
%   variable DRISHTI_MATLAB_CPU=1 to force the CPU. Callers gather() what
%   they extract, so results are ordinary arrays either way.
persistent useGpu
if isempty(useGpu)
    useGpu = isempty(getenv('DRISHTI_MATLAB_CPU')) && canUseGPU();
end
if useGpu
    X = gpuArray(X);
end
end
