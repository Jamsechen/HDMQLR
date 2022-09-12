function [SQvalue, D] = Smooth_Quan_value(X,Y,B,tau,kappa)

D     = Quantile_truncation(X,Y,B,tau,kappa);
[n,q] = size(Y);
A     = Y-X*B;
SQvalue = sum(sum(D.*A))/n - kappa*norm(D,'fro')^2/(2*n);