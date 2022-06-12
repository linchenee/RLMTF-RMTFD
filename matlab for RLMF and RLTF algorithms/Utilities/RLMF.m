function [X] = RLMF(Y, M, opts)
%  RLMF algorithm for image restorarion, solving the following optimization problem:
%  min_{U,V,U1,V1,X}  0.5*lambda1*(||U1||_w,F^2+||V1||_w,F^2)+lambda2*||M@(X-Y)||_p^p   s.t., U1=U, V1=V, U*V=X
% --------------------------------------------------------
% Input:
%  Y:                        n1*n2 matrix
%  M:                       binary mask that indicates the locations of missing and known entries in the matrix 
%  opts.lambda1:    regularization parameter  
%  opts.lambda2:    regularization parameter
%  opts.rank:           rank of the matrix
%  opts.p:                parameter that controls the weight allocation in the reweighted nuclear norm (0<p<=1)
%  opts.varepsilon:  small constant to avoid dividing by zero in the (re)weighted nuclear norm
%  opts.c:                 compromising constant in the (re)weighted nuclear norm
%  opts.numIter:      iteration number
%  opts.tol:              termination tolerance
% --------------------------------------------------------
% Output:
%  X:                        n1*n2 matrix
% ---------------------------------------------
% Written by Lin Chen (linchenee@sjtu.edu.cn)
%

[n1, n2] = size(Y);

if isfield(opts, 'lambda1');     lambda1 = opts.lambda1;        else lambda1 = sqrt(max(n1, n2));  end
if isfield(opts, 'lambda2');     lambda2 = opts.lambda2;        else lambda2 = 1;                           end  
if isfield(opts, 'rank');            rank = opts.rank;                      else rank = 25;                                end 
if isfield(opts, 'p');                 p = opts.p;                                else p = 0.5;                                    end 
if isfield(opts, 'varepsilon');   varepsilon = opts.varepsilon;  else varepsilon = 1e-8;                    end   
if isfield(opts, 'c');                  c = opts.c;                                else c = 1;                                        end 
if isfield(opts, 'numIter');       numIter = opts.numIter;          else numIter = 1e2;                         end  
if isfield(opts, 'tol');               tol = opts.tol;                           else tol = 1e-3;                                end  

omega = find(M);
Yomega = Y(omega);
weight = @(x) c .* (x + varepsilon) .^ (p - 1);
%% Initialization
U1 = randn(n1, rank); 
V1 = randn(rank, n2); 
V = V1;
X = U1 * V1; 
[~, s1] = svds(U1, rank); 
W_u = diag(s1);
[~, s2] = svds(V1, rank); 
W_v = diag(s2);
Z1 = zeros(n1, rank);
Z2 = zeros(rank, n2);
Z3 = zeros(n1, n2);
norm_two = norm(Y .* M, 'fro');
mu = 1 / norm_two;  % this one can be tuned
rho = 1.1;  % this one can be tuned

for iter = 1 : numIter
  %% Update U and V
  U = (U1 + Z1 ./mu + (X - Z3 ./ mu) * V') / (eye(rank) + V * V');
  V = (eye(rank)+ U' * U) \ (V1 + Z2 ./ mu + U' * (X - Z3 ./ mu));

  %% Update U1 and V1
  [a1, b1, c1] = svd(U - Z1 ./ mu, 'econ');
  W_u = diag(b1) ./ (lambda1 / mu .* weight(W_u .* W_v) + 1);
  U1 = a1 * diag(W_u) * c1';
  [a2, b2, c2] = svd(V - Z2 ./ mu, 'econ');
  W_v = diag(b2) ./ (lambda1 / mu .* weight(W_u .* W_v) + 1);
  V1 = a2 * diag(W_v) * c2';

  %% Update X
  temp = U * V;
  X = temp + Z3 ./ mu;
  X(omega) = Shink_Lp(X(omega) - Yomega, lambda2 / mu, 1) + Yomega;

  %% Update Lagrange multipliers
  Z1 = Z1 + mu .* (U1 - U); 
  Z2 = Z2 + mu .* (V1 - V); 
  Z3 = Z3 + mu .* (temp - X);     
  mu = mu * rho;

  %% Stop if the termination condition is met
  loss_U = (norm(U - U1, 'fro') / norm(U1, 'fro'))^2;
  loss_V = (norm(V - V1, 'fro') / norm(V1, 'fro'))^2;
  loss_UV = (norm(temp - X, 'fro') / norm(temp, 'fro'))^2;
  if loss_U < tol && loss_V < tol && loss_UV < tol
%     fprintf('Iter %d\n', iter);
    break;
  end    
end
end