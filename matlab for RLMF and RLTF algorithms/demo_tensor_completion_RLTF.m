clc;
clear;
close all;
addpath(genpath('Utilities'));
addpath(genpath('Utilities\quality_assess'));

load('.\data\CAVE_sampling_rate_10.mat');
data = double(CAVE_sampling_rate_10);
[n1, n2, n3, ~] = size(data);

GT = squeeze(data(:, :, :, 1));  % groundtruth
GT = normalized(GT);            % Rescale the image into the interval [0, 1].
M = squeeze(data(:, :, :, 2));   % binary mask that indicates the locations of missing and known entries in the image

%% RLTF algorithm for hyperspectral image (HSI) completion (Sampling rate=10%)
opts.lambda1 = 80 * sqrt(max(n1, n2) * n3);
opts.lambda2 = 1e200;  % It should be large enough to have (X.*M=GT.*M) for the tensor completion.
opts.p = 0.5;
opts.varepsilon = 1e-4;
opts.numIter = 150;
temp = 120;
opts.rank = [temp, floor(temp / 4) * ones(1,n3 - 1)];
X = RLTF(GT .* M, M, opts);

%% Performance evaluation
[psnr, ssim, fsim, ergas] = MSIQA(GT .* 255, X .* 255);
fprintf('PSNR achieved by RLTF is %d dB\n', psnr);
fprintf('SSIM achieved by RLTF is %d\n', ssim);
fprintf('FSIM achieved by RLTF is %d\n', fsim);
fprintf('ERGAS achieved by RLTF is %d\n', ergas);
figure(1);
imshow(uint8(X(:, :, 1) .* 255), []);
