%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [B_new,Supp_B_new,Time] = pADMM_Quantile(X,Y,tau,lambda,quan_para)

tstart = clock;
XT     = X';          [n,p] = size(X); 
q      = size(Y,2);   eta   = eigs(XT*X,1,'LM');

Lag_para_type = quan_para.Lagtype;   sigma = quan_para.sigma;   
gamma         = quan_para.gamma;     Max   = quan_para.maxiter;
E_abs         = 1e-3;                E_rel = 1e-3;

alpha      = 1;%1<=alpha<=1.8
B_old      = quan_para.B0;        C_old     = Y - X*B_old;
V_old      = zeros(n,q);          sigma_old = sigma;  
SigmaB_old = XT*(X*B_old);             

for t = 1:Max  
% B-update
    
    G = B_old - 1/eta*SigmaB_old + 1/(sigma_old*eta)*XT*(V_old-sigma_old*(C_old-Y));
    [B_new, Supp_B_new] = shrinkage(G,lambda/(sigma_old*eta));
    XB_new   = X*B_new;
    Y_XB_new = Y - XB_new;
    Y_HXB    = alpha*Y_XB_new + (1-alpha)*C_old; %Y- (alpha*(XB_new) + (1-alpha)*(Y-C_old));
% C-update   
    C_new = Proximal_quantile(Y_HXB + V_old/sigma_old,1/(n*sigma_old),tau);
% Dual variables-update
    V_new = V_old - gamma*sigma_old*(C_new - Y_HXB);

%% Check stopping critria
   SigmaB_new = XT*XB_new;
   RP         = norm(C_new - Y_XB_new,'fro');
   RD         = sigma_old*norm(XT*(C_new-C_old) + eta*(B_new-B_old) - SigmaB_new + SigmaB_old,'fro');
   RPB        = sqrt(n*q)*E_abs + E_rel*max([norm(XB_new,'fro'), norm(C_new,'fro'), norm(Y,'fro')]);
   RDB        = sqrt(p*q)*E_abs + E_rel*norm(XT*V_new,'fro');
if (RP<RPB)&(RD<RDB)
   break;
else
   B_old      = B_new;         C_old      = C_new;
   V_old      = V_new;         SigmaB_old = SigmaB_new;
end
if strcmp(Lag_para_type,'change')
   if (RP>10*RD)
       sigma_new = 2*sigma_old;
       elseif (RP>RD/10)
              sigma_new = sigma_old;
   else
       sigma_new = sigma_old/2;
   end
elseif strcmp(Lag_para_type,'fixed')
       sigma_new = sigma_old;
end
       sigma_old = sigma_new;
end 

Time = etime(clock,tstart);