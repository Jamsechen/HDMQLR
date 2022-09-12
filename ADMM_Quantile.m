%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [B_new,Supp_B_new,Z_new,Supp_Z_new,Time] = ADMM_Quantile(X,Y,tau,lambda,quan_para)

tstart = clock;

XT = X';          [n,p] = size(X);
q  = size(Y,2);

Lag_para_type = quan_para.Lagtype;        sigma = quan_para.sigma; 
gamma         = quan_para.gamma;          Max   = quan_para.maxiter;
E_abs         = 1e-3;                     E_rel = 1e-3;
alpha         = 1;%1<=alpha<=1.8 alpha>1 over-relaxation;  alpha<1 under-relaxation

Z_old     = quan_para.B0;                 C_old = Y - X*quan_para.B0;
U_old     = zeros(p,q);                   V_old = zeros(n,q);
sigma_old = sigma;

for t = 1:Max

% B-update Cholesky decomposition
rho    = 1;
[L,U]  = factor(X, rho);
B_temp = XT*(Y+V_old/sigma_old-C_old) + (Z_old+U_old/sigma_old);

if (n>=p)    % if  X is skinny
    B_new = U\(L\B_temp);
else         % if X is fat
    B_new = B_temp/rho - (XT*(U\(L\(X*B_temp))))/rho^2;
end

% Z-update
HB = alpha*B_new + (1-alpha)*Z_old;
[Z_new, Supp_Z_new] = shrinkage(HB-U_old/sigma_old,lambda/sigma_old);

% C-update
XHB   = X*HB;%
HXB   = alpha*XHB+(1-alpha)*(Y-C_old);
C_new = Proximal_quantile(Y-HXB+V_old/sigma_old,1/(n*sigma_old),tau);

% Dual variables-update
BZ    = HB - Z_new; 
XBCY  = HXB + C_new - Y;
U_new = U_old - gamma*sigma_old*BZ;
V_new = V_old - gamma*sigma_old*XBCY;

%% Check stopping critria
if (alpha==1)
   XB = XHB;
else
   XB = X*B_new;
end
RP  = sqrt(norm(B_new- Z_new,'fro')^2+norm(XB+C_new - Y,'fro')^2); %
RD  = sigma_old*norm(Z_new-Z_old-XT*(C_new-C_old),'fro');
XXB = sqrt(norm(B_new,'fro')^2 +norm(XB,'fro')^2);
AN  = sqrt(norm(Z_new,'fro')^2+norm(C_new,'fro')^2);
RPB = sqrt((n+p)*q)*E_abs+E_rel*max([XXB,AN, norm(Y,'fro')]);
RDB = sqrt(p*q)*E_abs+E_rel*sqrt(norm(U_new,'fro')+norm(XT*V_new,'fro')^2);

if (RP<RPB)&(RD<RDB)
   break;
else
   Z_old = Z_new;
   C_old = C_new;
   U_old = U_new;
   V_old = V_new;
end

%sigma-update

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

Supp_B_new = [ ];
for i=1:p
 if norm(B_new(i,:),2)~=0
    Supp_B_new = [Supp_B_new  i];
 end
end

Time = etime(clock,tstart);