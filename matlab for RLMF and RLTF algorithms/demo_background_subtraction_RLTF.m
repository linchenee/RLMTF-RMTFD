clc;
clear all;
close all;
addpath(genpath('Utilities'));
addpath(genpath('Utilities\quality_assess'));

load('.\data\CAVIAR1.mat');
load('.\data\CAVIAR1_GT.mat');
[n1, n2, n3] = size(CAVIAR1);

%% RLTF algorithm for background subtraction
opts.lambda1 = 1 * sqrt(max(n1, n2) * n3);
opts.lambda2 = 0.5;
opts.p = 0.8;
opts.numIter = 150;
opts.rank = [200, ones(1, n3 - 1)];
Y = double(CAVIAR1) ./ 255;  % input
X = RLTF(Y, ones(n1, n2, n3), opts);

%% Performance evaluation
GT = double(repmat(CAVIAR1_GT, [1, 1, n3]));  % groundtruth
[psnr, ssim, fsim, ergas] = MSIQA(GT, X .* 255);
fprintf('PSNR achieved by RLTF is %d dB\n', psnr);
fprintf('SSIM achieved by RLTF is %d\n', ssim);
fprintf('FSIM achieved by RLTF is %d\n', fsim);
fprintf('ERGAS achieved by RLTF is %d\n', ergas);
figure(1);
imshow(uint8(X(:, :, 70) .* 255), []);
