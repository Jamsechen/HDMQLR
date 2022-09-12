function Grad = Grad_usQuantile(X,Y,B,tau,kappa)

[n,q] = size(Y);
A     = (Y-X*B)/n;
Grad  = zeros(n,q);
for i=1:n
 for j=1:q
     u = A(i,j);
     if (u<-kappa)
       Grad(i,j) = tau - 1;
     elseif (A(i,j)<=kappa)
       Grad(i,j) = u/(2*kappa) + tau - 1/2;
     else
       Grad(i,j) = tau;
     end
 end
end