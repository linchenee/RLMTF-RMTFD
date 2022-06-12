function [x] = Shink_Lp(y, alpha, p)
%  Shrinkage operator for the L1 or L2 norm, solving the following optimization problem:
%  min_{x} 0.5*||x-y||_F^2 + alpha*||x||_p^p
% --------------------------------------------------------
% Input:
%  y:         a vector or a matrix
%  alpha:  alpha>0
%  p:         p=1 or p=2 
% --------------------------------------------------------
% Output:
%  x:        a vector or a matrix
% --------------------------------------------------------
% Written by Lin Chen (linchenee@sjtu.edu.cn)
%
  
abs_y=abs(y);
if p == 1
  x = (abs_y - alpha) .* (abs_y > alpha);   
elseif p == 2
  x = 1 / (2 * alpha + 1) .* abs_y;   
end
x = x .* sign(y);
end
