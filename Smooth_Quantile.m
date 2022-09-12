%% Problem--- min { ||Y-XB||_F^2/(2n) + lambda*||B||_{1,2} }
%% approximate as min { ||Y-XZ||_F^2/(2n) + <X^T*(XZ-Y),B-Z> + L_f/2*||B-Z||_F^2 + lambda*||B||_{1,2} } 
%Z--initial point.  
% lambda is a vector

function [B_new,Supp_B_new,Time] = Smooth_Quantile(X,Y,lambda,linesearch,para,smooth_type)
time_start = clock;
[n,p]   = size(X);                       XT  = X';
kappa   = para.kappa;                    tau = para.tau;
Maxiter = para.maxiter;                  tol = para.tol;
B0      = para.B0;
eta     = para.line;
if strcmp(smooth_type,'lower')
   L_f = eigs(XT*X,1,'LM')/(n^2*kappa); 
end  
if strcmp(smooth_type,'upper')
   L_f = eigs(XT*X,1,'LM')/(2*n^2*kappa);
end    
Z_old  = B0;               
t_old  = 1;                             L_old = L_f;              
L_min  = 1e-2*L_f;                      B_old = B0;               
XB_old = X*B_old;

for i=1:Maxiter

%% no linesearch
if (linesearch==0) 
   L = L_old;
if strcmp(smooth_type,'lower')
   D_Z = Grad_lsQuantile(X,Y,Z_old,tau,kappa);
end
if strcmp(smooth_type,'upper')
   D_Z = Grad_usQuantile(X,Y,Z_old,tau,kappa);
end
   Grad = -XT*D_Z/n;
   G    = Z_old - Grad/L;%X'*(Y-X*Z_old)
   [B_new,Supp_B_new] = Shrinkage_Block(G,lambda/L);
   L_new = L;
if strcmp(smooth_type,'lower')
   D_B = Grad_lsQuantile(X,Y,B_new,tau,kappa);
end
if strcmp(smooth_type,'upper')
   D_B = Grad_usQuantile(X,Y,B_new,tau,kappa);
end
end

%% linesearch
if (linesearch==1) 
    L0 = max(eta*L_old,L_min);
for Lmax=1:100
    L = L0;
 if strcmp(smooth_type,'lower')
    [sqValue_Z,D_Z] = lSmooth_Quan_value(X,Y,Z_old,tau,kappa);
 end
 if strcmp(smooth_type,'upper')
    [sqValue_Z,D_Z] = uSmooth_Quan_value(X,Y,Z_old,tau,kappa);
 end
    Grad = -XT*D_Z/n;
    G    = Z_old - Grad/L;
    [B_temp,Supp_B_temp] = Shrinkage_Block(G,lambda/L);
    normB_temp2  = 0;
for j=1:p
    normj        = norm(B_temp(j,:),2);
    normB_temp2  = normB_temp2 + normj;
end
if strcmp(smooth_type,'lower')
   [sqValue_B,D_B] = lSmooth_Quan_value(X,Y,B_temp,tau,kappa);
 end
if strcmp(smooth_type,'upper')
   [sqValue_B,D_B] = uSmooth_Quan_value(X,Y,B_temp,tau,kappa);
end 
    F_temp = sqValue_B + lambda*normB_temp2;
    Q_temp = sqValue_Z + trace(Grad'*(B_temp-Z_old)) + L*norm(B_temp-Z_old,'fro')^2/2 + lambda*normB_temp2;

if (F_temp<=Q_temp)|(Lmax==100)
    L_new      = min(L_f,max(L,L_min));
    B_new      = B_temp; 
    Supp_B_new = Supp_B_temp; 
    break;
else
    L0 = min(L/eta,L_f);
end
end
%Lmax
end

t_new = (1+sqrt(1+4*t_old^2))/2;
c     = (t_old-1)/(t_new);
Z_new = (1+c)*B_new - c*B_old;

% Check stopping criterion 
XB_new = X*B_new;
R      = -XT*D_B/n + L_new*(G-B_new);
Cr1    = norm(R,'fro')/(L_new*max(1,norm(B_new,'fro')));
Cr2    = abs(norm(Y-XB_new,'fro')-norm(Y-XB_old,'fro'))/max(1,norm(Y,'fro'));
if (Cr1<= tol)&(Cr2<5*tol)
   break;
else
B_old  = B_new;      Z_old = Z_new;
t_old  = t_new;      L_old = L_new;
XB_old = XB_new;
end
end
Time = etime(clock,time_start);