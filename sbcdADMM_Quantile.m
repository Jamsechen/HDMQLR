%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [B_new,Supp_B_new,Time] = sbcdADMM_Quantile(X,Y,tau,lambda,quan_para)

tstart = clock;
XT     = X';              Sigma = XT*X;
d      = diag(Sigma);     [n,p] = size(X);    
q      = size(Y,2);

Lag_type = quan_para.Lagtype;       update = quan_para.update;
sigma    = quan_para.sigma;         gamma  = quan_para.gamma;
Max      = quan_para.maxiter;       Max_B  = 100;
alpha    = 1; % 1<=alpha<=1.8
E_abs    = 1e-3;                    E_rel  = 1e-3;               tol_B = 1e-5;

B_old = quan_para.B0;    C_old     = Y - X*B_old;   
V_old = zeros(n,q);      sigma_old = sigma; 
for t = 1:Max   
%% Update B by sparse block-coordinate descent 
XR         = XT*(C_old-Y-V_old/sigma_old);
SigmaB_old = triu(Sigma,1)*B_old ;
for LB = 1:Max_B
    B_temp_new = zeros(p,q);
    Supp_B_temp_new = [ ];
    
if  strcmp(update,'new')

for j=1:p
    gj              = -(SigmaB_old(j,:) + Sigma(j,:)*B_temp_new + XR(j,:))/d(j);
    [bj, Supp_bj]   = shrinkage(gj,lambda/(sigma_old*d(j)));
    B_temp_new(j,:) = bj;
 if (~isempty(Supp_bj))
    Supp_B_temp_new = [Supp_B_temp_new j];
 end
end

end

if strcmp(update,'old') 
   G = -diag(1./d)*(SigmaB_old-diag(d)*B_old + XR);
   [B_temp_new, Supp_B_temp_new] = mshrinkage(G,lambda./(sigma_old*d));
end  
 
SigmaB_temp_new = triu(Sigma,1)*B_temp_new;
RKKT = norm(SigmaB_temp_new-SigmaB_old,'fro')/max(1,norm(B_temp_new,'fro'));
if (RKKT<=tol_B)|(LB ==Max_B)  
   B_new      = B_temp_new;
   Supp_B_new = Supp_B_temp_new;
   SigmaB_old = SigmaB_temp_new;
   break;
else
   B_old = B_temp_new; 
end
end
  XB_new   = X*B_new;
  Y_XB_new = Y-XB_new;
  Y_HXB    = alpha*Y_XB_new + (1-alpha)*C_old; %Y- (alpha*(XB_new) + (1-alpha)*(Y-C_old));
 
%% Update C,V
  C_new = Proximal_quantile(Y_HXB+V_old/sigma_old,1/(n*sigma_old),tau);
  V_new = V_old - gamma*sigma_old*(C_new-Y_HXB);
%% Check stopping critria

   RP  = norm(C_new-Y_XB_new,'fro');
   RD  = sigma_old*norm((XT*(C_old-C_new)),'fro');
   RPB = sqrt(n*q)*E_abs+E_rel*max([norm(XB_new ,'fro'), norm(C_new,'fro'), norm(Y,'fro')]);
   RDB = sqrt(p*q)*E_abs+E_rel*norm(XT*V_new,'fro');

if (RP<RPB)&(RD<RDB)
   break;
else
   B_old = B_new;
   C_old = C_new;
   V_old = V_new;
end
if strcmp(Lag_type,'change')
   if (RP>10*RD)
      sigma_new = 2*sigma_old;
      elseif (RP>RD/10)
             sigma_new = sigma_old;
   else
      sigma_new = sigma_old/2;
   end
elseif strcmp(Lag_type,'fixed')
       sigma_new = sigma_old;
end
   sigma_old = sigma_new;
end

Time = etime(clock,tstart);