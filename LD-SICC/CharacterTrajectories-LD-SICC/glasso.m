function [X, W] = glasso(data, lambda)
%% Graphical Lasso - Friedman et. al, Biostatistics, 2008
% Input:
%   data - n * p matrix with n samples and p variables
%   lambda - thresholding parameter
% Output:
%   X - Concentration matrix
%   W - Covariance matrix
% Written by Quan Liu
%%
data = normlization(data);
p = size(data,2);
S = cov(data);
W = S + lambda * eye(p);
beta = zeros(p) - lambda * eye(p);
eps = 1e-4;
finished = false(p);
while true
    for j = 1 : p
        idx = 1 : p; idx(j) = [];
        beta(idx, j) = lasso(W(idx, idx), S(idx, j), lambda, beta(idx, j));
        W(idx, j) = W(idx,idx) * beta(idx, j);
        W(j, idx) = W(idx, j);
    end
    index = (beta == 0);
    finished(index) = (abs(W(index) - S(index)) <= lambda);
    finished(~index) = (abs(W(~index) -S(~index) + lambda * sign(beta(~index))) < eps);
    if finished
        break;
    end
end
X = zeros(p);
for j = 1 : p
    idx = 1 : p; idx(j) = [];
    X(j,j) = 1 / (W(j,j) - dot(W(idx,j), beta(idx,j)));
    X(idx, j) = -1 * X(j, j) * beta(idx,j);
end
X = sparse(X);
end

%% Data Normalization
function data = normlization(data)
data = bsxfun(@minus, data, mean(data));
data = bsxfun(@rdivide, data, std(data));
end
