clear all;
start = datestr(now);       t1 = cputime;
n     = 500;                p  = 1000;
q     = 100;                s  = 0.01*p;

tau = 0.5;  % quantile
et  = 1;    % error type: 1--N(0,1);  2--Cauchy(0,1);  3--exp(1);  4--t(1)
cor = 1;    % correlation type: % for X; 1--d^|i-j|;  2--d 
d   = 0.5;  % correlation

Repeat = 2; % repeat times

for j=1:Repeat
    [X,Y,Xv,Yv,B_true] = Generatedata(n,p,q,s,d,et,cor); 
    nv    = size(Xv,1);
    [p,q] = size(B_true);
    B0    = zeros(p,q); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

quan_para.kappa   = 10^(-3);  % used in smoothong quantile
quan_para.tau     = tau;      % quantile
quan_para.xi      = 10^(-3);  % used in Majorize-Minimize
quan_para.tol     = 10^(-5);  % tollerance error
quan_para.line    = 0.7;      % used in linesearch
quan_para.sigma   = 0.1;      % Parameter in Augmented Lagrangian
quan_para.Lagtype = 'change'; % fixed--sigma is fixed; change--sigma changes;
quan_para.update  = 'new';    % used in sbcd ADMM; 
                              % new--use the new B_{j.} in the next iteration; 
                              % old--use the old B_{j.} in the next iteration;
quan_para.gamma   = (1+sqrt(5))/2; % stepsize
quan_para.maxiter = 5000; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ADMM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

quan_para.maxiter = 100; 
delta    = 5*10^(-2);
lambda   = Generate_TuningPara_l1l2_quantile(X,Y,tau,delta,et); 
l_lambda = length(lambda);        

MSE_Est  = zeros(1,l_lambda);     MSE_Pre  = zeros(1,l_lambda);     
Supp_Est = cell(1,l_lambda);      L_Supp   = zeros(1,l_lambda);     
Cpu      = zeros(1,l_lambda);     VMSE_Pre = zeros(1,l_lambda);

for i=1:l_lambda
    quan_para.B0 = B0;
    [B_new,Supp_B_new,Z_new,Supp_Z_new,Time] = ADMM_Quantile(X,Y,tau,lambda(i),quan_para);
    MSE_Est(i)  = norm(Z_new-B_true,'fro')/sqrt(p*q); 
    Supp_Est{i} = Supp_Z_new;     df = length(Supp_Est{i});     L_Supp(i) = df;
    VMSE_Pre(i) = log(Quan_value(Yv-Xv*Z_new,tau)/nv) + log(nv)*df/nv;
    TP          = length(find(Supp_Z_new<=s));         Fp(i)  = L_Supp(i) - TP;
    Fn(i)       = s - TP;                              Cpu(i) = Time;
    B0          = Z_new;
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
MSE(j)    = MSE_Est(Ind(1));                     SUPP(j) = L_Supp(Ind(1));
FP(j)     = Fp(Ind(1));                          FN(j)   = Fn(Ind(1));
CPU(j)    = Cpu(Ind(1));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Proximal ADMM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

quan_para.maxiter = 1000;
pdelta    = 5*10^(-2); 
plambda   = Generate_TuningPara_l1l2_quantile(X,Y,tau,pdelta,et);
l_plambda = length(plambda);
PMSE_Est  = zeros(1,l_plambda);  PMSE_Pre  = zeros(1,l_plambda);
PSupp_Est = cell(1,l_plambda);   PL_Supp   = zeros(1,l_plambda);
PCpu      = zeros(1,l_plambda); 

for i=1:l_plambda
    quan_para.B0 = B0;
    [PB_new,PSupp_B_new,PTime] = pADMM_Quantile(X,Y,tau,plambda(i),quan_para);
    PMSE_Est(i)  = norm(PB_new-B_true,'fro')/sqrt(p*q);
    PSupp_Est{i} = PSupp_B_new;              Pdf = length(PSupp_Est{i});
    PL_Supp(i)   = Pdf;
    PVMSE_Pre(i) = log(Quan_value(Yv-Xv*PB_new,tau)/nv) + log(nv)*Pdf/nv;
    PTP          = length(find(PSupp_B_new<=s)); 
    PFp(i)       = PL_Supp(i) - PTP;         PFn(i) = s - PTP;
    PCpu(i)      = PTime;                    B0     = PB_new;
end

%% Pick out the nonzero estimator
PSupp_Id = [ ];
for k=1:l_plambda
 if (length(PL_Supp(k))~=0)
     PSupp_Id = [PSupp_Id k]; 
 end
end
if (length(PSupp_Id)==0)
   disp('The tuning parameters are too large!')
end
 
PVMSE_PreA = PVMSE_Pre(PSupp_Id);
PInd       = find(PVMSE_PreA==min(PVMSE_PreA));
PMSE(j)    = PMSE_Est(PInd(1));                PSUPP(j) = PL_Supp(PInd(1));
PFP(j)     = PFp(PInd(1));                     PFN(j)   = PFn(PInd(1));
PCPU(j)    = PCpu(PInd(1));
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Sparse block-coordinate descent (sbcd) ADMM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

quan_para.maxiter = 1000; 
sbcddelta    = 5*10^(-2);
sbcdlambda   = Generate_TuningPara_l1l2_quantile(X,Y,tau,sbcddelta,et);
l_sbcdlambda = length(sbcdlambda);      sbcdMSE_Est  = zeros(1,l_sbcdlambda);     
sbcdMSE_Pre  = zeros(1,l_sbcdlambda);   sbcdSupp_Est = cell(1,l_sbcdlambda);      
sbcdL_Supp   = zeros(1,l_sbcdlambda);   sbcdCpu      = zeros(1,l_sbcdlambda);  

for i=1:l_sbcdlambda
    quan_para.B0 = B0;
    [sbcdB_new,sbcdSupp_B_new,sbcdTime] = sbcdADMM_Quantile(X,Y,tau,sbcdlambda(i),quan_para);
    sbcdMSE_Est(i)  = norm(sbcdB_new-B_true,'fro')/sqrt(p*q);
    sbcdSupp_Est{i} = sbcdSupp_B_new;         sbcddf = length(sbcdSupp_Est{i});
    sbcdL_Supp(i)   = sbcddf;
    sbcdVMSE_Pre(i) = log(Quan_value(Yv-Xv*sbcdB_new,tau)/nv) +log(nv)*sbcddf/nv;
    sbcdTP          = length(find(sbcdSupp_B_new<=s));
    sbcdFp(i)       = sbcdL_Supp(i) - sbcdTP;    sbcdFn(i) = s - sbcdTP;
    sbcdCpu(i)      = sbcdTime;                  B0        = sbcdB_new;
end

%% Pick out the nonzero estimator
 
sbcdSupp_Id = [ ];
for k=1:l_sbcdlambda
 if (length(sbcdL_Supp(k))~=0)
    sbcdSupp_Id = [sbcdSupp_Id k]; 
 end
end
if (length(sbcdSupp_Id)==0)
    disp('The tuning parameters are too large!')
end
sbcdVMSE_PreA = sbcdVMSE_Pre(sbcdSupp_Id);
sbcdInd       = find(sbcdVMSE_PreA==min(sbcdVMSE_PreA));
sbcdMSE(j)    = sbcdMSE_Est(sbcdInd(1));  
sbcdSUPP(j)   = sbcdL_Supp(sbcdInd(1)); 
sbcdFP(j)     = sbcdFp(sbcdInd(1));      
sbcdFN(j)     = sbcdFn(sbcdInd(1));
sbcdCPU(j)    = sbcdCpu(sbcdInd(1));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% lower Smooth quantile %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

smoothtype = 'lower';
quan_para.maxiter = 5000; 
lsqdelta    = 5*10^(-2);
lsqlambda   = Generate_TuningPara_l1l2_quantile(X,Y,tau,lsqdelta,et);
l_lsqlambda = length(lsqlambda);
lsqMSE_Est  = zeros(1,l_lsqlambda);  lsqMSE_Pre  = zeros(1,l_lsqlambda);
lsqSupp_Est = cell(1,l_lsqlambda);   lsqL_Supp   = zeros(1,l_lsqlambda);
lsqCpu      = zeros(1,l_lsqlambda);  lsqVMSE_Pre = zeros(1,l_lsqlambda);
linesearch  = 1;
for i=1:l_lsqlambda
    quan_para.B0 = B0;
    [lsqB_new,lsqSupp_B_new,lsqTime] = Smooth_Quantile(X,Y,lsqlambda(i),linesearch,quan_para,smoothtype);
    lsqMSE_Est(i)  = norm(lsqB_new-B_true,'fro')/sqrt(p*q);
    lsqSupp_Est{i} = lsqSupp_B_new;             lsqdf = length(lsqSupp_Est{i});
    lsqL_Supp(i)   = lsqdf;
    lsqVMSE_Pre(i) = log(Quan_value(Yv-Xv*lsqB_new,tau)/nv) +log(nv)*lsqdf/nv;
    lsqTP          = length(find(lsqSupp_B_new<=s));  
    lsqFp(i)       = lsqL_Supp(i) - lsqTP;      lsqFn(i) = s - lsqTP;                       
    lsqCpu(i)      = lsqTime;                   B0       = lsqB_new;
end

%% Pick out the nonzero estimator
 
lsqSupp_Id = [ ];
 for k=1:l_lsqlambda
  if (length(lsqL_Supp(k))~=0)
     lsqSupp_Id = [lsqSupp_Id k]; 
  end
 end
 if (length(lsqSupp_Id)==0)
    disp('The tuning parameters are too large!')
 end
lsqVMSE_PreA = lsqVMSE_Pre(sbcdSupp_Id);
lsqInd       = find(lsqVMSE_PreA==min(lsqVMSE_PreA));
lsqMSE(j)    = lsqMSE_Est(lsqInd(1));    lsqSUPP(j) = lsqL_Supp(lsqInd(1));
lsqFP(j)     = lsqFp(lsqInd(1));         lsqFN(j)   = lsqFn(lsqInd(1));
lsqCPU(j)    = lsqCpu(lsqInd(1));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% upper Smooth quantile %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

smoothtype = 'upper';
quan_para.maxiter = 5000; 
usqdelta    = 5*10^(-2);
usqlambda   = Generate_TuningPara_l1l2_quantile(X,Y,tau,usqdelta,et);
l_usqlambda = length(usqlambda);
usqMSE_Est  = zeros(1,l_usqlambda);  usqMSE_Pre  = zeros(1,l_usqlambda);
usqSupp_Est = cell(1,l_usqlambda);   usqL_Supp   = zeros(1,l_usqlambda);
usqCpu      = zeros(1,l_usqlambda);  usqVMSE_Pre = zeros(1,l_usqlambda);
linesearch = 1;

for i=1:l_usqlambda
    quan_para.B0 = B0;
    [usqB_new,usqSupp_B_new,usqTime] = Smooth_Quantile(X,Y,usqlambda(i),linesearch,quan_para,smoothtype);
    usqMSE_Est(i)  = norm(usqB_new-B_true,'fro')/sqrt(p*q);
    usqSupp_Est{i} = usqSupp_B_new;             usqdf = length(usqSupp_Est{i});
    usqL_Supp(i)   = usqdf;
    usqVMSE_Pre(i) = log(Quan_value(Yv-Xv*usqB_new,tau)/nv) +log(nv)*usqdf/nv;
    usqTP          = length(find(usqSupp_B_new<=s));
    usqFp(i)       = usqL_Supp(i) - usqTP;                usqFn(i) = s - usqTP;
    usqCpu(i)      = usqTime;                             B0       = usqB_new;
end

%% Pick out the nonzero estimator
usqSupp_Id = [ ];
for k=1:l_usqlambda
 if (length(usqL_Supp(k))~=0)
    usqSupp_Id = [usqSupp_Id k]; 
 end
end
if (length(usqSupp_Id)==0)
   disp('The tuning parameters are too large!')
end
usqVMSE_PreA = usqVMSE_Pre(usqSupp_Id);
usqInd       = find(usqVMSE_PreA==min(usqVMSE_PreA));
usqMSE(j)    = usqMSE_Est(usqInd(1));    usqSUPP(j) = usqL_Supp(usqInd(1));
usqFP(j)     = usqFp(usqInd(1));         usqFN(j)   = usqFn(usqInd(1));
usqCPU(j)    = usqCpu(usqInd(1));


%%%%%%%%%%%%%%%%%%%%%%%%%%% Majorize-Minimize lower Smooth quantile with linesearch %%%%%%%%%%%%%%%%%%%%%%%%%

smooth_type = 'lower';
quan_para.maxiter = 5000; 
MMlsqdelta    = 5*10^(-2);
MMlsqlambda   = Generate_TuningPara_l1l2_quantile(X,Y,tau,MMlsqdelta,et);
l_MMlsqlambda = length(MMlsqlambda);
MMlsqMSE_Est  = zeros(1,l_MMlsqlambda);  MMlsqMSE_Pre  = zeros(1,l_MMlsqlambda);
MMlsqSupp_Est = cell(1,l_MMlsqlambda);   MMlsqL_Supp   = zeros(1,l_MMlsqlambda);
MMlsqCpu      = zeros(1,l_MMlsqlambda);  MMlsqVMSE_Pre = zeros(1,l_MMlsqlambda);
linesearch    = 1;

for i=1:l_MMlsqlambda
    quan_para.B0 = B0;
    [MMlsqB_new,MMlsqSupp_B_new,MMlsqTime] = Smooth_QuantileMML(X,Y,MMlsqlambda(i),linesearch,quan_para,smooth_type);
    MMlsqMSE_Est(i)  = norm(MMlsqB_new-B_true,'fro')/sqrt(p*q);
    MMlsqSupp_Est{i} = MMlsqSupp_B_new;     MMlsqdf = length(MMlsqSupp_Est{i});
    MMlsqL_Supp(i)   = MMlsqdf;
    MMlsqVMSE_Pre(i) = log(Quan_value(Yv-Xv*MMlsqB_new,tau)/nv) +log(nv)*MMlsqdf/nv;
    MMlsqTP          = length(find(MMlsqSupp_B_new<=s));
    MMlsqFp(i)       = MMlsqL_Supp(i) - MMlsqTP;   MMlsqFn(i) = s - MMlsqTP;
    MMlsqCpu(i)      = MMlsqTime;                  B0         = MMlsqB_new;
end

%% Pick out the nonzero estimator
MMlsqSupp_Id = [ ];
for k=1:l_MMlsqlambda
 if (length(MMlsqL_Supp(k))~=0)
    MMlsqSupp_Id = [MMlsqSupp_Id k]; 
 end
end
if (length(MMlsqSupp_Id)==0)
   disp('The tuning parameters are too large!')
end
MMlsqVMSE_PreA = MMlsqVMSE_Pre(MMlsqSupp_Id);
MMlsqInd       = find(MMlsqVMSE_PreA==min(MMlsqVMSE_PreA));
MMlsqMSE(j)    = MMlsqMSE_Est(MMlsqInd(1)); 
MMlsqSUPP(j)   = MMlsqL_Supp(MMlsqInd(1));      
MMlsqFP(j)     = MMlsqFp(MMlsqInd(1));
MMlsqFN(j)     = MMlsqFn(MMlsqInd(1));          
MMlsqCPU(j)    = MMlsqCpu(MMlsqInd(1));


%%%%%%%%%%%%%%%%%%%%%%%%%% Majorize-Minimize upper Smooth quantile with linesearch %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

smooth_type = 'upper';
quan_para.maxiter = 5000; 
MMusqdelta    = 5*10^(-2);
MMusqlambda   = Generate_TuningPara_l1l2_quantile(X,Y,tau,MMusqdelta,et);
l_MMusqlambda = length(MMusqlambda);
MMusqMSE_Est  = zeros(1,l_MMusqlambda);  MMusqMSE_Pre  = zeros(1,l_MMusqlambda);
MMusqSupp_Est = cell(1,l_MMusqlambda);   MMusqL_Supp   = zeros(1,l_MMusqlambda);
MMusqCpu      = zeros(1,l_MMusqlambda);  MMusqVMSE_Pre = zeros(1,l_MMusqlambda);

for i=1:l_MMusqlambda
    quan_para.B0 = B0;
    [MMusqB_new,MMusqSupp_B_new,MMusqTime] = Smooth_QuantileMML(X,Y,MMusqlambda(i),linesearch,quan_para,smooth_type);
    MMusqMSE_Est(i)  = norm(MMusqB_new-B_true,'fro')/sqrt(p*q);
    MMusqSupp_Est{i} = MMusqSupp_B_new;     MMusqdf = length(MMusqSupp_Est{i});
    MMusqL_Supp(i)   = MMusqdf;
    MMusqVMSE_Pre(i) = log(Quan_value(Yv-Xv*MMusqB_new,tau)/nv) +log(nv)*MMusqdf/nv;
    MMusqTP          = length(find(MMusqSupp_B_new<=s));
    MMusqFp(i)       = MMusqL_Supp(i) - MMusqTP;      MMusqFn(i) = s - MMusqTP;
    MMusqCpu(i)      = MMusqTime;                     B0         = MMusqB_new;
end

%% Pick out the nonzero estimator
MMusqSupp_Id = [ ];
for k=1:l_MMusqlambda
 if (length(MMusqL_Supp(k))~=0)
    MMusqSupp_Id = [MMusqSupp_Id k]; 
 end
end
if (length(MMusqSupp_Id)==0)
   disp('The tuning parameters are too large!')
end
MMusqVMSE_PreA = MMusqVMSE_Pre(MMusqSupp_Id);
MMusqInd       = find(MMusqVMSE_PreA==min(MMusqVMSE_PreA));
MMusqMSE(j)    = MMusqMSE_Est(MMusqInd(1));
MMusqSUPP(j)   = MMusqL_Supp(MMusqInd(1));
MMusqFP(j)     = MMusqFp(MMusqInd(1));
MMusqFN(j)     = MMusqFn(MMusqInd(1));
MMusqCPU(j)    = MMusqCpu(MMusqInd(1));
end

%% Mean and standar deviation

%% ADMM 
meanMSE  = mean(MSE);      stdMSE  = std(MSE);
meanSUPP = mean(SUPP);     stdSUPP = std(SUPP);
meanFP   = mean(FP);         stdFP = std(FP);
meanFN   = mean(FN);         stdFN = std(FN);
meanCPU  = mean(CPU);       stdCPU = std(CPU);

%% Proximal ADMM 
meanPMSE  = mean(PMSE);      stdPMSE  = std(PMSE);
meanPSUPP = mean(PSUPP);     stdPSUPP = std(PSUPP);
meanPFP   = mean(PFP);         stdPFP = std(PFP);
meanPFN   = mean(PFN);         stdPFN = std(PFN);
meanPCPU  = mean(PCPU);       stdPCPU = std(PCPU);

%% Sparse block-coordinate descent ADMM 
meansbcdMSE  = mean(sbcdMSE);      stdsbcdMSE  = std(sbcdMSE);
meansbcdSUPP = mean(sbcdSUPP);     stdsbcdSUPP = std(sbcdSUPP);
meansbcdFP   = mean(sbcdFP);         stdsbcdFP = std(sbcdFP);
meansbcdFN   = mean(sbcdFN);         stdsbcdFN = std(sbcdFN);
meansbcdCPU  = mean(sbcdCPU);       stdsbcdCPU = std(sbcdCPU);

%% lower Smooth quantile
meanlsqMSE  = mean(lsqMSE);      stdlsqMSE  = std(lsqMSE);
meanlsqSUPP = mean(lsqSUPP);     stdlsqSUPP = std(lsqSUPP);
meanlsqFP   = mean(lsqFP);         stdlsqFP = std(lsqFP);
meanlsqFN   = mean(lsqFN);         stdlsqFN = std(lsqFN);
meanlsqCPU  = mean(lsqCPU);       stdlsqCPU = std(lsqCPU);

%% upper Smooth quantile
meanusqMSE  = mean(usqMSE);      stdusqMSE  = std(usqMSE);
meanusqSUPP = mean(usqSUPP);     stdusqSUPP = std(usqSUPP);
meanusqFP   = mean(usqFP);         stdusqFP = std(usqFP);
meanusqFN   = mean(usqFN);         stdusqFN = std(usqFN);
meanusqCPU  = mean(usqCPU);       stdusqCPU = std(usqCPU);

%% Majorize-Minimize lower Smooth quantile
meanMMlsqMSE  = mean(MMlsqMSE);      stdMMlsqMSE  = std(MMlsqMSE);
meanMMlsqSUPP = mean(MMlsqSUPP);     stdMMlsqSUPP = std(MMlsqSUPP);
meanMMlsqFP   = mean(MMlsqFP);         stdMMlsqFP = std(MMlsqFP);
meanMMlsqFN   = mean(MMlsqFN);         stdMMlsqFN = std(MMlsqFN);
meanMMlsqCPU  = mean(MMlsqCPU);       stdMMlsqCPU = std(MMlsqCPU);

%% Majorize-Minimize upper Smooth quantile
meanMMusqMSE  = mean(MMusqMSE);      stdMMusqMSE  = std(MMusqMSE);
meanMMusqSUPP = mean(MMusqSUPP);     stdMMusqSUPP = std(MMusqSUPP);
meanMMusqFP   = mean(MMusqFP);         stdMMusqFP = std(MMusqFP);
meanMMusqFN   = mean(MMusqFN);         stdMMusqFN = std(MMusqFN);
meanMMusqCPU  = mean(MMusqCPU);       stdMMusqCPU = std(MMusqCPU);

%% Final rsults
Results =... 
  {   'Methods',     'ADMM',     'pADMM',    'sbcdADMM',      'lSmooth',     'uSmooth',    'MMlSmooth',     'MMuSmooth';...
    'mean(MSE)',    meanMSE,    meanPMSE,    meansbcdMSE,    meanlsqMSE,    meanusqMSE,   meanMMlsqMSE,    meanMMusqMSE;...
    'std(RMSE)',     stdMSE,     stdPMSE,     stdsbcdMSE,     stdlsqMSE,     stdusqMSE,     stdMMlsqMSE,     stdMMusqMSE;...
'mean(Support)',   meanSUPP,   meanPSUPP,   meansbcdSUPP,   meanlsqSUPP,   meanusqSUPP,   meanMMlsqSUPP,   meanMMusqSUPP;...
 'std(Support)',    stdSUPP,    stdPSUPP,    stdsbcdSUPP,    stdlsqSUPP,    stdusqSUPP,    stdMMlsqSUPP,    stdMMusqSUPP;...
     'mean(FP)',     meanFP,     meanPFP,     meansbcdFP,     meanlsqFP,     meanusqFP,     meanMMlsqFP,     meanMMusqFP;...
      'std(FP)',      stdFP,      stdPFP,      stdsbcdFP,      stdlsqFP,      stdusqFP,      stdMMlsqFP,      stdMMusqFP;...
     'mean(FN)',     meanFN,     meanPFN,     meansbcdFN,     meanlsqFN,     meanusqFN,     meanMMlsqFN,     meanMMusqFN;...
      'std(FN)',      stdFN,      stdPFN,      stdsbcdFN,      stdlsqFN,      stdusqFN,      stdMMlsqFN,      stdMMusqFN;...
      'meanCPU',    meanCPU,    meanPCPU,    meansbcdCPU,    meanlsqCPU,    meanusqCPU,    meanMMlsqCPU,    meanMMusqCPU;};
Results
ending = datestr(now); Etime = cputime - t1; 
C = {'Start time', 'End time', 'Elasped time';start, ending, Etime}
