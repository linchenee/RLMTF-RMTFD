clc;
clear;
close all;
addpath(genpath('Utilities'));
addpath(genpath('Utilities\quality_assess'));

GT = imread('.\data\1.jpg');  % groundtruth
SNR = 10;
Y = double(imnoise(uint8(GT), 'salt & pepper', 1/SNR));
[n1, n2, n3] = size(Y);
rate = 0.5;    % sampling rate
M = zeros(n1, n2, n3);   % binary mask that indicates the locations of missing and known entries in the image
omega = rand(n1 * n2 * n3, 1) < rate;
M(omega) = 1;

opts.lambda1 = 8 * sqrt(max(n1, n2));
opts.rank = 24;
opts.p = 0.5;
opts.numIter = 150;
opts.tol = 1e-3;

%% RLMF algorithm for image restoration
for i = 1 : n3
  X1(:, :, i) = RLMF(Y(:, :, i) .* M(:, :, i), M(:, :, i), opts);
end
psnr1 = PSNR(double(GT), X1, ~M);
fprintf('PSNR achieved by RLMF is %d dB\n', psnr1);
figure(1);
imshow(uint8(X1), []);

%% RLTF algorithm for image restoration
opts.lambda1 = 3 * sqrt(max(n1, n2) * n3);
opts.p = 0.5;   
opts.numIter = 150;
temp = 75; % 75-->10dB  100-->20dB
opts.rank = [temp, floor(temp / 6), floor(temp / 12)];
X2 = RLTF(Y .* M, M, opts);
psnr2 = PSNR(double(GT), X2, ~M);
fprintf('PSNR achieved by RLTF is %d dB\n', psnr2);
figure(2);
imshow(uint8(X2), []);
