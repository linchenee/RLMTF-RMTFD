clc;
clear;
close all;
addpath(genpath('util\quality_assess'));

load('Result_SBI.mat');  % Please first run "demo_background_subtraction_RTFD.py" to obtain this file.
load('.\data\CAVIAR1_GT.mat');
GT = repmat(double(CAVIAR1_GT), [1, 1, 100]);  % groundtruth

[psnr, ssim, fsim, ergas] = MSIQA(GT, squeeze(Result));
fprintf('PSNR achieved by RTFD is %d dB\n', psnr);
fprintf('SSIM achieved by RTFD is %d\n', ssim);
fprintf('FSIM achieved by RTFD is %d\n', fsim);
fprintf('ERGAS achieved by RTFD is %d\n', ergas);