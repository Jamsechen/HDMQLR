clear all; 
n = 500;     p = 500;
q = 100;     s = 0.01*p;

d   = 0.5;    
et  = 1;
cor = 1;
tau = 0.5;  % quantile

[X,Y,Xv,Yv,B_true] = Generatedata(n,p,q,s,d,et,cor);
nv    = size(Xv,1);
[p,q] = size(B_true);
B0    = zeros(p,q); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

quan_para.sigma   = 0.1;              % Parameter in Augmented Lagrangian 
quan_para.Lagtype = 'change';         % fixed--sigma is fixed; change--sigma changes;
quan_para.gamma   = (1+sqrt(5))/2;    % step size
quan_para.maxiter = 1000;             % The maximum number of iteration in ADMM

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%if(et==1)|(et==3)|(et==4)
%  lambda = [1000:-10:200]/n; 
%end
%if(et==2) 
%  lambda = [5000:-50:800]/n; 
%end
%l_lambda = length(lambda);


delta    = 5*10^(-2); 
lambda   = Generate_TuningPara_l1l2_quantile(X,Y,tau,delta,et);
l_lambda = length(lambda);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Proximal ADMM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

MSE_Est  = zeros(1,l_lambda);    MSE_Pre = zeros(1,l_lambda);
Supp_Est = cell(1,l_lambda);     L_Supp  = zeros(1,l_lambda);
Cpu      = zeros(1,l_lambda);  
for i=1:l_lambda
quan_para.B0 = B0;
[B_new,Supp_B_new,Time] = pADMM_Quantile(X,Y,tau,lambda(i),quan_para);
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
Results = [MSE,  FP,  FN, CPU]

