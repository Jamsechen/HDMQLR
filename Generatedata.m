%%
  %Generate data for the model Y = b*XB + W,
  %where  Y(Nxq), X(Nx(1+p)), B((1+p)xq)
  %s--number of material predictors; 
  %d--design correlation for error
  %N--total number of data
  %n--the number of data in local machine
  %et--error type
  %cor--correlation type for design matrix
%%
%-------------------------------------------------------------------------------------
function [X,Y,Xv,Yv,B_true] = Generatedata(n,p,q,s,d,et,cor)     
nv = n;
%% generate predictor
 
SIGMA = d*ones(p,p);
if cor == 1 
   for u = 1:p
      for v = 1:p
          SIGMA(u,v)=d^(abs(u-v));
      end
   end
end
if cor == 2 
   for u = 1:p
       SIGMA(u,u) = 1;
   end
end
 MU = zeros(1,p);
 X  = mvnrnd(MU,SIGMA,n); 
 X  = [ones(n,1) X];
 Xv = mvnrnd(MU,SIGMA,nv); 
 Xv = [ones(nv,1) Xv];

%% generate true coefficient matrix
  weight   = 10*[1:s]/s;
  B_true_S = diag(weight)*(2*binornd(1,0.5,[s,q])-1);%rand(s,q);
  B_true   = [B_true_S; zeros(p+1-s,q)];
  Y_temp   = X*B_true; 
  Yv_temp  = Xv*B_true;
 
%% add noise
  if et==1
     Y  = Y_temp  + randn(n,q);  
     Yv = Yv_temp + randn(nv,q);  % Normal error: N(0,1)  
  end

  if et==2
     Y  = Y_temp  + trnd(1,n,q); 
     Yv = Yv_temp + trnd(1,nv,q); % Cauchy error
  end

  if et==3
     Y  = Y_temp  + exprnd(1,n,q); 
     Yv = Yv_temp + exprnd(1,nv,q); % Exponential error: exp(1)
  end

  if et==4
     Y  = Y_temp  + trnd(2,n,q); 
     Yv = Yv_temp + trnd(2,nv,q); % t(2)-distribution error
  end
