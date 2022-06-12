function [mse] = mse(A, B)
if ~exist('B', 'var')
 mse = sum(A(:).^2) ./ length(A(:));
else 
 mse = sum((A(:) - B(:)).^2) ./ length(A(:));
end
end