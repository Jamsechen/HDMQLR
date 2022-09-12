%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [B_new,Supp_B_new,Time] = sbcdADMM_QuantileA(X,Y,tau,lambda,quan_para)

tstart = clock;

Sigma = X'*X;
d     = diag(Sigma);
[n,p] = size(X);    
q     = size(Y,2);

Lag_type = quan_para.Lagtype;
update   = quan_para.update;
sigma    = quan_para.sigma; 
gamma    = quan_para.gamma;
Max      = quan_para.maxiter;
Max_B    = 5;
alpha    = 1; % 1<=alpha<=1.8
E_abs    = 1e-3;    E_rel = 1e-2;    tol_B = 1e-3;

B_old     = quan_para.B0;    
C_old     = zeros(n,q);    
V_old     = zeros(n,q);
sigma_old = sigma; 
for t = 1:Max   
%% Update B by sparse block-coordinate descent 
XR = X'*(C_old -Y-V_old/sigma_old);
for LB = 1:Max_B
    B_temp_new      = B_old;
    Supp_B_temp_new = [ ];
    tic;
if  strcmp(update,'new')
for j=1:p
    gj1 = zeros(1,q);
    for k=1:(j-1)
        gj1 = gj1 + Sigma(j,k)*B_temp_new(k,:);
    end
    gj2 = zeros(1,q);
    for k=(j+1):p
        gj2 = gj2 + Sigma(j,k)*B_temp_new(k,:);
    end
gj = -(gj1 + gj2 + XR(j,:))/d(j);

[bj, Supp_bj]   = shrinkage(gj,lambda/(sigma_old*d(j)));
B_temp_new(j,:) = bj;
if (~isempty(Supp_bj))
Supp_B_temp_new = [Supp_B_temp_new  j];
end
end
end
toc;
if  strcmp(update,'old')
    D = diag(d);  
    G = -diag(1./d)*((Sigma-D)*B_temp_new+XR);
    [B_temp_new, Supp_B_temp_new] = mshrinkage(G,lambda./(sigma_old*d));
end  

RKKT = norm(triu(Sigma,2)*(B_temp_new-B_old),'fro')/max(1,norm(B_temp_new,'fro'));
if (RKKT<=tol_B)|(LB ==2) 
   B_new      = B_temp_new;
   Supp_B_new = Supp_B_temp_new;
   break;
else
   B_old = B_temp_new; 
end
end
  XB_new = X*B_new;
  HXB    = alpha*(XB_new)+(1-alpha)*(Y-C_old);
%% Update C,V
  C_new = Proximal_quantile(Y-HXB+V_old/sigma_old,1/(n*sigma_old),tau);
  XBCY  = HXB + C_new - Y;
  V_new = V_old - gamma*sigma_old*XBCY;

%% Check stopping critria
   RP  = norm(XB_new+C_new-Y,'fro');
   RD  = sigma_old*norm((X'*(C_old-C_new)),'fro');
   RPB = sqrt(n*q)*E_abs+E_rel*max([norm(XB_new ,'fro'), norm(C_new,'fro'), norm(Y,'fro')]);
   RDB = sqrt(p*q)*E_abs+E_rel*norm(X'*V_new,'fro');

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