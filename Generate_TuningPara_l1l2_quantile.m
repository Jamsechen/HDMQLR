%% Generate tuning parameters for model 
%%  l1/l2 norm regularized quantile
%Iput: X--Predictor matrix
%      Y--Response matrix
function lambda = Generate_TuningPara_l1l2_quantile(X,Y,tau,delta,et)
[n,p] = size(X);
q     = size(Y,2);
for i=1:p
    X12(i) = norm(X(:,i),2); 
end
for i=1:n
for j=1:q
    if Y(i,j)==0
       YY(i,j) = max(tau,1-tau);
     else
       YY(i,j) = tau-1/2+1/2*sign(Y(i,j));
     end
end
end
lambda_max = max(X12)*norm(YY,'fro')/n;

if (et==2)
   lambda_N = 200;
else
   lambda_N = 100;
end
lambda = lambda_max*delta.^([0:1:(lambda_N-1)]/(lambda_N-1));
