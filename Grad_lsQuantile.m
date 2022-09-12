function Grad = Grad_usQuantile(X,Y,B,tau,kappa)

[n,q] = size(Y);
A     = (Y-X*B)/(n*kappa);
Grad  = zeros(n,q);
for i=1:n
 for j=1:q
     u = A(i,j);
     if (u<(tau-1)*kappa)
       Grad(i,j) = tau - 1;
     elseif (A(i,j)<=tau*kappa)
       Grad(i,j) = u;
     else
       Grad(i,j) = tau;
     end
 end
end