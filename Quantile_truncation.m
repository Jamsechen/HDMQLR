%% Compute: trancated quantile value [(Y-XB)/(n*kappa)]_tau
% tau--quantile
% kappa--smoothing parameter
% X--predictor matrix
% Y--response matrix

function AT = Quantile_truncation(X,Y,B,tau,kappa)
[n,q] = size(Y);
A     = (Y-X*B)/(kappa);
AT    = A;
for i=1:n
 for j=1:q
     if (A(i,j)>=tau)
       AT(i,j) = tau;
     elseif (A(i,j)<=tau-1)
       AT(i,j) = tau-1;
     end
 end
end