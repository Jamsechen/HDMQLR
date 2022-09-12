clear all; 
n = 500;       p = 500;
q = 100;       s = 0.01*p;

d  = 0.5;       cor = 1;
et = 3;
tau = 0.5;  % quantile

[X,Y,Xv,Yv,B_true] = Generatedata(n,p,q,s,d,et,cor); 
nv    = size(Xv,1);
[p,q] = size(B_true);
B0    = zeros(p,q); 

smooth_type = input('Please choose the smoothing type {"lower" or "upper"}:\n','s')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

quan_para.kappa   = 10^(-3);
quan_para.tau     = tau;
quan_para.maxiter = 5000; 
quan_para.tol     = 10^(-5); 
quan_para.line    = 0.7;
quan_para.B0      = B0;
linesearch        = 1;

delta    = 5*10^(-2);
lambda   = Generate_TuningPara_l1l2_quantile(X,Y,tau,delta,et);
l_lambda = length(lambda);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Smooth quantile %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MSE_Est  = zeros(1,l_lambda);      MSE_Pre  = zeros(1,l_lambda);
Supp_Est = cell(1,l_lambda);       L_Supp   = zeros(1,l_lambda);
Cpu      = zeros(1,l_lambda);      VMSE_Pre = zeros(1,l_lambda);
for i=1:l_lambda
[B_new,Supp_B_new,Time] = Smooth_Quantile(X,Y,lambda(i),linesearch,quan_para,smooth_type);
MSE_Est(i)  = norm(B_new-B_true,'fro')/sqrt(p*q); 
Supp_Est{i} = Supp_B_new;     df = length(Supp_Est{i});     L_Supp(i) = df;
VMSE_Pre(i) = log(Quan_value(Yv-Xv*B_new,tau)/nv) + log(nv)*df/nv;
TP          = length(find(Supp_B_new<=s));         Fp(i)  = L_Supp(i) - TP;
Fn(i)       = s - TP;                              Cpu(i) = Time;
B0          = B_new;
end

%% Pick out the nonzero estimator
 Supp_Id = [ ];
 for k=1:l_lambda
  if (length(L_Supp(k))~=0)
      Supp_Id = [Supp_Id k]; 
  end
 end
 if (length(Supp_Id)==0)
    disp('The tuning parameters are too large!')
 end
 
VMSE_PreA = VMSE_Pre(Supp_Id);
Ind       = find(VMSE_PreA==min(VMSE_PreA));
MSE    = MSE_Est(Ind(1));                     SUPP = L_Supp(Ind(1));
FP     = Fp(Ind(1));                          FN   = Fn(Ind(1));
CPU    = Cpu(Ind(1));

%%
if strcmp(smooth_type,'lower')
lower_Smooth_Results = [MSE,  FP,  FN, CPU]
end
if strcmp(smooth_type,'upper')
upper_Smooth_Results = [MSE,  FP,  FN, CPU]
end
