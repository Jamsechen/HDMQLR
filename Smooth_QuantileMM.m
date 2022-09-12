%% Problem--- min { ||Y-XB||_F^2/(2n) + lambda*||B||_{1,2} }
%% approximate as min { ||Y-XZ||_F^2/(2n) + <X^T*(XZ-Y),B-Z> + L_f/2*||B-Z||_F^2 + lambda*||B||_{1,2} } 
%Z--initial point.  
% lambda is a vector

function [B_new,Supp_B_new,Time] = Smooth_QuantileMM(X,Y,lambda,para,smooth_type)

time_start = clock;

XT      = X';                      n   = size(X,1);
kappa   = para.kappa;              tau = para.tau;
Maxiter = para.maxiter;            tol = para.tol; 
B0      = para.B0;                 xi  = para.xi;
if strcmp(smooth_type,'lower')
   L = eigs(XT*X,1,'LM')/(n^2*kappa);
else
   L = eigs(XT*X,1,'LM')/(2*n^2*kappa);  
end

B_old  = B0;
if strcmp(smooth_type,'lower')
   D_B_old = Grad_lsQuantile(X,Y,B_old,tau,kappa);
end
if strcmp(smooth_type,'upper')
   D_B_old = Grad_usQuantile(X,Y,B_old,tau,kappa);
end
XB_old    = X*B_old;
XTD_B_old = XT*D_B_old;
for i=1:Maxiter
    Grad = -XTD_B_old/n; 
    G    = B_old - Grad/L;
    [B_new,Supp_B_new] = Shrinkage_Block(G,lambda/L);
 if strcmp(smooth_type,'lower')
    D_B_new  = Grad_lsQuantile(X,Y,B_new,tau,kappa);
 end
 if strcmp(smooth_type,'upper')
    D_B_new  = Grad_usQuantile(X,Y,B_new,tau,kappa);
 end

%% Check stopping criterion 

XB_new    = X*B_new;
XTD_B_new = XT*D_B_new;
R         = (XTD_B_new-XTD_B_old)/n - L*(B_new-B_old);
Cr1       = norm(R,'fro')/(L*max(1,norm(B_new,'fro')));
Cr2       = abs(norm(Y-XB_new,'fro')-norm(Y-XB_old,'fro'))/max(1,norm(Y,'fro'));
if (Cr1<=tol)&(Cr2<5*tol)
   break;
else
B_old  = B_new;    XTD_B_old = XTD_B_new;       
XB_old = XB_new;
end
end

Time = etime(clock,time_start);