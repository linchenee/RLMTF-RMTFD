clc;
clear;
close all;
addpath(genpath('util\quality_assess'));

load('Result_CAVE_sampling_rate_10.mat');  % Please first run "demo_tensor_completion_RTFD.py" to obtain this file.
load('.\data\CAVE_sampling_rate_10.mat');

n1 = 256; n2 = 256; n3 = 31;
GT = zeros(n1, n2, n3);  % groundtruth, which has been normalized.
count = 1;
for i = 1 : n1
 for j = 1 : n2
   for k = 1 : n3
     GT(i, j, k) = Y(1, count);
     count = count + 1;
   end
 end
end   

[psnr, ssim, fsim, ergas] = MSIQA(GT .* 255, squeeze(Result));
fprintf('PSNR achieved by RTFD is %d dB\n', psnr);
fprintf('SSIM achieved by RTFD is %d\n', ssim);
fprintf('FSIM achieved by RTFD is %d\n', fsim);
fprintf('ERGAS achieved by RTFD is %d\n', ergas);
