%% Problem--- min { ||Y-XB||_F^2/(2n) + lambda*||B||_{1,2} }
%% approximate as min { ||Y-XZ||_F^2/(2n) + <X^T*(XZ-Y),B-Z> + L_f/2*||B-Z||_F^2 + lambda*||B||_{1,2} } 
%Z--initial point.  
% lambda is a vector

function [B_new,Supp_B_new,Time] = Smooth_QuantileMM(X,Y,lambda,linesearch,para,smooth_type)
time_start = clock;

XT      = X';                      n   = size(X,1);
kappa   = para.kappa;              tau = para.tau;
Maxiter = para.maxiter;            tol = para.tol;
B0      = para.B0;                 xi  = para.xi;
eta     = para.line;

if strcmp(smooth_type,'lower')
   L_f = eigs(XT*X,1,'LM')/(n^2*kappa) + xi;
else
   L_f = eigs(XT*X,1,'LM')/(2*n^2*kappa) + xi;  
end
             
L_old = L_f;              
L_min  = 1e-2*L_f;        B_old = B0;   

if strcmp(smooth_type,'lower')
   [sqValue_B_old,D_B_old] = lSmooth_Quan_value(X,Y,B_old,tau,kappa);
end
if strcmp(smooth_type,'upper')
   [sqValue_B_old,D_B_old] = uSmooth_Quan_value(X,Y,B_old,tau,kappa);
end

XB_old    = X*B_old;
XTD_B_old = XT*D_B_old;

for i=1:Maxiter

%% no linesearch
if (linesearch==0) 
   L = L_old;
   Grad = -XTD_B_old/n;
   G    = B_old - Grad/L;
   [B_new,Supp_B_new] = Shrinkage_Block(G,lambda/L);
   L_new = L;
if strcmp(smooth_type,'lower')
   D_B_new = Grad_lsQuantile(X,Y,B_new,tau,kappa);
end
if strcmp(smooth_type,'upper')
   D_B_new = Grad_usQuantile(X,Y,B_new,tau,kappa);
end
end

%% linesearch

if (linesearch==1) 
    L0 = max(eta*L_old,L_min);
for Lmax=1:100
    L = L0;
    Grad = -XTD_B_old/n;
    G    = B_old - Grad/L;
    [B_temp,Supp_B_temp] = Shrinkage_Block(G,lambda/L);
if strcmp(smooth_type,'lower')
   [sqValue_B_temp,D_B_temp] = lSmooth_Quan_value(X,Y,B_temp,tau,kappa);
 end
if strcmp(smooth_type,'upper')
   [sqValue_B_temp,D_B_temp] = uSmooth_Quan_value(X,Y,B_temp,tau,kappa);
end 
   Q_temp = sqValue_B_temp;
   M_temp = sqValue_B_old + trace(Grad'*(B_temp-B_old)) + L*norm(B_temp-B_old,'fro')^2/2;

if (Q_temp<=M_temp)|(Lmax==100)
   L_new      = min(L_f,max(L,L_min));
   B_new      = B_temp; 
   D_B_new    = D_B_temp;
   Supp_B_new = Supp_B_temp; 
   sqValue_B_new = sqValue_B_temp;
   break;
else
   L0 = min(L/eta,L_f);
end
end
end

%% Check stopping criterion 

XB_new    = X*B_new;
XTD_B_new = XT*D_B_new;
R         = (XTD_B_new-XTD_B_old)/n - L_new*(B_new-B_old);
Cr1       = norm(R,'fro')/(L_new*max(1,norm(B_new,'fro')));
Cr2       = abs(norm(Y-XB_new,'fro')-norm(Y-XB_old,'fro'))/max(1,norm(Y,'fro'));
if (Cr1<=tol)&(Cr2<5*tol)
   break;
else
B_old     = B_new;           XB_old        = XB_new;
XTD_B_old = XTD_B_new;       sqValue_B_old = sqValue_B_new;
end
end

Time = etime(clock,time_start);