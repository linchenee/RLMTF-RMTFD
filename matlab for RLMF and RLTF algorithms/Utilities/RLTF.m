function [X] = RLTF(Y, M, opts) 
%  RLTF algorithm for image restorarion, solving the following optimization problem:
%  min_{U,V,U1,V1,X}  0.5*lambda1*(||U1||_w,F^2+||V1||_w,F^2)+lambda2*||M@(X-Y)||_p^p   s.t., U1=U, V1=V, U*V=X
% --------------------------------------------------------
% Input:
%  Y:                        n1*n2*n3 tensor
%  M:                       binary mask that indicates the locations of missing and known entries in the tensor 
%  opts.lambda1:    regularization parameter  
%  opts.lambda2:    regularization parameter
%  opts.rank:           tensor multi-rank
%  opts.p:                parameter that controls the weight allocation in the reweighted nuclear norm (0<p<=1)
%  opts.varepsilon:  small constant to avoid dividing by zero in the (re)weighted nuclear norm
%  opts.c:                 compromising constant in the (re)weighted nuclear norm
%  opts.numIter:      iteration number
% --------------------------------------------------------
% Output:
%  X:                        n1*n2*n3 tensor
% --------------------------------------------------------
% Written by Lin Chen (linchenee@sjtu.edu.cn)
%

[n1, n2, n3] = size(Y);

if isfield(opts, 'lambda1');     lambda1 = opts.lambda1;        else lambda1 = sqrt(max(n1, n2) * n3);  end
if isfield(opts, 'lambda2');     lambda2 = opts.lambda2;        else lambda2 = 1;                                   end  
if isfield(opts, 'rank');            rank = opts.rank;                      else rank = 50 * ones(1, n3);                   end 
if isfield(opts, 'p');                 p = opts.p;                                else p = 0.5;                                            end 
if isfield(opts, 'varepsilon');   varepsilon = opts.varepsilon;   else varepsilon = 1e-8;                           end   
if isfield(opts, 'c');                  c = opts.c;                                else c = 1;                                                end 
if isfield(opts, 'numIter');       numIter = opts.numIter;          else numIter = 1e2;                                 end  

omega = find(M);
Yomega = Y(omega);
weight = @(x) c .* (x + varepsilon) .^ (p - 1);
%% Initialization
Xd = dct(ones(n1, n2, n3) .* ~M + Y .* M, [], 3);  % X in the DCT domain
temp = Xd;
for i = 1 : n3
  [u{i}, s{i}, v{i}] = svds(Xd(:, :, i), rank(i)); 
  Ud{i} = u{i} * sqrt(s{i});       % U in the DCT domain
  U1d{i} = Ud{i};                   % U1 in the DCT domain
  Vd{i} = sqrt(s{i}) * v{i}';       % V in the DCT domain
  V1d{i} = Vd{i};                    % V1 in the DCT domain
  W_u{i} = diag(sqrt(s{i}));    % weights in the reweighted Frobenius norm of U 
  W_v{i} = W_u{i};                 % weights in the reweighted Frobenius norm of V 
  Z1{i} = zeros(size(Ud{i}));   % Lagrange multipliers
  Z2{i} = zeros(size(Vd{i}));   % Lagrange multipliers
end
Z3 = zeros(n1, n2, n3);
norm_two = norm(Yomega(:));
mu = 2 / norm_two;  % this one can be tuned
rho = 1.1;   % this one can be tuned

for iter =  1 : numIter
   for i = 1 : n3
      %% Update U and V in the DCT domain
      Ud{i} = (U1d{i} + Z1{i} / mu + temp(:, :, i) * Vd{i}') / (eye(rank(i)) + Vd{i} * Vd{i}');  
      Vd{i} = (eye(rank(i)) + Ud{i}' * Ud{i}) \ (V1d{i} + Z2{i} / mu + Ud{i}' * temp(:, :, i));
      %% Update U1 and V1 in the DCT domain
      [a1{i}, b1{i}, c1{i}] = svd(Ud{i} - Z1{i} / mu, 'econ');
      W_u{i} = diag(b1{i}) ./ (lambda1 / mu .* weight(W_u{i} .* W_v{i}) + 1);
      U1d{i} = a1{i} * diag(W_u{i}) * c1{i}';
      [a2{i}, b2{i}, c2{i}] = svd(Vd{i} - Z2{i} / mu, 'econ');
      W_v{i} = diag(b2{i}) ./ (lambda1 / mu .* weight(W_u{i} .* W_v{i}) + 1);
      V1d{i} = a2{i} * diag(W_v{i}) * c2{i}';
      UVd(:, :, i) = Ud{i} * Vd{i};
      %% Update Lagrange multipliers Z1 and Z2
      Z1{i} = Z1{i} + mu .* (U1d{i} - Ud{i});
      Z2{i} = Z2{i} + mu .* (V1d{i} - Vd{i});
   end
        
   %% Update X
   X_temp = idct(UVd, [], 3) + Z3 / mu;
   X = X_temp;
   X(omega) = Shink_Lp(X_temp(omega) - Yomega, lambda2 / mu, 1) + Yomega;

  %% Update Lagrange multipliers Z3
   Z3 = mu .* (X_temp - X);
   temp = dct(X - Z3 / mu, [], 3);
   mu = mu * rho;
end
end
    